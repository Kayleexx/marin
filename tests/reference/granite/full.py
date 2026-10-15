"""Pinned Transformers 4.46.1 FP32 oracle for Granite 3.0 1B-A400M."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, GraniteMoeForCausalLM


def vector(tensor):
    return tensor.detach().float().cpu().tolist()


def quantize_like_marin(model, group=128):
    # Independent FP32 reconstruction of Marin's BF16 row-group maxabs Q8 rule.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim < 2 or "router.layer" in name:
                continue
            if parameter.shape[-1] % group:
                raise ValueError(f"nondivisible Q8 matrix: {name}")
            rows = parameter.data.reshape(-1, group)
            scale = rows.abs().amax(dim=1, keepdim=True) / 127.0
            codes = torch.sign(rows) * torch.floor(rows.abs() / scale.clamp_min(1e-45) + 0.5)
            rows.copy_(codes.clamp(-127, 127) * scale)


def case(model, tokenizer, prompt, steps):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    expert_outputs = []
    layer_outputs = []
    hooks = []
    for layer in model.model.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _args, output: layer_outputs.append(
                    vector(output[0][0, -1])
                )
            )
        )
        hooks.append(
            layer.block_sparse_moe.register_forward_hook(
                lambda _module, _args, output: expert_outputs.append(
                    vector(output[0][0, -1])
                )
            )
        )
    with torch.inference_mode():
        result = model(
            input_ids=torch.tensor([ids]),
            output_hidden_states=True,
            output_router_logits=True,
            use_cache=True,
            return_dict=True,
        )
        layers = [vector(result.hidden_states[0][0, -1])] + layer_outputs
        routers = [vector(t[-1]) for t in result.router_logits]
        top = [
            torch.topk(t[-1], model.config.num_experts_per_tok).indices.cpu().tolist()
            for t in result.router_logits
        ]
        output = {
            "prompt": prompt,
            "prompt_ids": ids,
            "layer_norms": [sum(v * v for v in row) ** 0.5 for row in layers],
            "layer_values": layers,
            "router_values": routers,
            "router_top": top,
            "expert_values": expert_outputs[:len(model.model.layers)],
            "steps": [],
        }
        past = result.past_key_values
        logits = result.logits[0, -1]
        for index in range(steps):
            token = int(torch.argmax(logits).item())
            choices = torch.topk(logits, 5)
            output["steps"].append(
                {
                    "token": token,
                    "top": [
                        [int(i), float(v)]
                        for i, v in zip(choices.indices, choices.values)
                    ],
                    "logits": vector(logits),
                    "router_top": [
                        torch.topk(t[-1], model.config.num_experts_per_tok).indices.cpu().tolist()
                        for t in result.router_logits
                    ],
                    "layer_values": [vector(result.hidden_states[0][0, -1])] + layer_outputs[-len(model.model.layers):] if index < 3 else [],
                    "expert_values": expert_outputs[-len(model.model.layers):] if index < 3 else [],
                }
            )
            if index + 1 < steps:
                result = model(
                    input_ids=torch.tensor([[token]]),
                    past_key_values=past,
                    output_router_logits=True,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )
                past = result.past_key_values
                logits = result.logits[0, -1]
        for hook in hooks:
            hook.remove()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--quantize-int8", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = GraniteMoeForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    if args.quantize_int8:
        quantize_like_marin(model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with a brief greeting."}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a Python function that adds two integers."}],
            tokenize=False,
            add_generation_prompt=True,
        ),
    ]
    if args.long:
        prompts.append(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": ("The workshop builds small wooden clocks. " * 30)
                        + "Summarize the workshop in one sentence.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    result = {"format": 1, "cases": [case(model, tokenizer, p, args.steps) for p in prompts]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result))
    for item in result["cases"]:
        print(len(item["prompt_ids"]), [step["token"] for step in item["steps"]])


if __name__ == "__main__":
    main()
