#!/usr/bin/env python3
"""Official text-backbone reference with exact source weights and FP32 CPU math."""
import argparse
import json
import pathlib
import shutil
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import __version__
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig, Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding
from tokenizers import Tokenizer

assert __version__ == "5.17.0", "review reference changes before changing this pin"
torch.set_num_threads(8)


def float_math(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = lambda x, m=module: F.linear(x.float(), m.weight.float())
        elif isinstance(module, torch.nn.Embedding):
            module.forward = lambda ids, m=module: F.embedding(ids, m.weight).float()
        elif isinstance(module, torch.nn.Conv1d):
            module.float()  # HF causal-conv fallback explicitly casts input to the weight dtype.
    model.eval()
    return model


def load(path, scratch=None):
    config = json.loads((path / "config.json").read_text())
    text = Qwen3_5TextConfig.from_dict(config["text_config"])
    text._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(text)
    missing = set(model.state_dict())
    missing.discard("lm_head.weight")  # Tied to the input embedding below.
    manifest_path = path / "marin-int8.json"
    group = json.loads(manifest_path.read_text())["group"] if manifest_path.exists() else None
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as src:
            for name in src.keys():
                if name.startswith("model.language_model.") and not name.endswith(".scales"):
                    value = src.get_tensor(name)
                    if value.dtype == torch.int8:
                        rows, cols = value.shape
                        scales = src.get_tensor(name + ".scales")
                        assert group and scales.shape == (rows, cols // group)
                        decoded = (value.float().reshape(rows, cols // group, group) * scales[..., None]).reshape(rows, cols)
                        if scratch is None:
                            value = decoded
                        else:
                            # 4B Q8 expands beyond this host's RAM; clean file-backed pages can be reclaimed.
                            mapped = np.memmap(pathlib.Path(scratch) / (name + ".f32"), dtype=np.float32,
                                               mode="w+", shape=(rows, cols))
                            mapped[:] = decoded.numpy()
                            mapped.flush()
                            value = torch.from_numpy(mapped)
                            del mapped
                        del decoded, scales
                    key = name.replace("model.language_model.", "model.", 1)
                    assert key in missing, key
                    model.load_state_dict({key: value}, strict=False, assign=True)
                    missing.remove(key)
                    del value
    assert not missing, missing
    model.tie_weights()
    # Meta initialization does not populate nonpersistent rotary buffers.
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(text)
    assert all(b.device.type != "meta" for b in model.buffers())
    return float_math(model)


def tiny(path, quantizable=False):
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(350)
    text = Qwen3_5TextConfig(vocab_size=320, hidden_size=32 if quantizable else 8, intermediate_size=64 if quantizable else 12,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16 if quantizable else 8,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8 if quantizable else 3,
        linear_value_head_dim=16 if quantizable else 4, linear_conv_kernel_dim=4, max_position_embeddings=1024,
        layer_types=["linear_attention"]*3+["full_attention"], tie_word_embeddings=True,
        eos_token_id=0, rope_parameters={"rope_type":"default","rope_theta":1e7,
            "partial_rotary_factor":.5,"mrope_section":[1,1,0]})
    text._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(text).eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if "norm.weight" in name and "linear_attn.norm.weight" not in name:
                p.copy_(torch.randn_like(p)*.1)
    tensors = {}
    for name,p in model.state_dict().items():
        if name.startswith("model."):
            key = name.replace("model.", "model.language_model.", 1)
            dtype = torch.float32 if name.endswith("A_log") or "linear_attn.norm.weight" in name else torch.bfloat16
            tensors[key] = p.detach().to(dtype).contiguous()
    save_file(tensors, str(path / "model.safetensors"))
    c = text.to_dict(); c["attn_output_gate"] = True
    (path / "config.json").write_text(json.dumps({"model_type":"qwen3_5","text_config":c}))
    shutil.copy(pathlib.Path(__file__).parents[2] / "fixtures/tiny-chat/tokenizer.json", path / "tokenizer.json")


@torch.inference_mode()
def reference(path, out, prompts, steps):
    scratch = tempfile.TemporaryDirectory(dir=out.parent) if (path / "marin-int8.json").exists() else None
    model = load(path, scratch.name if scratch else None)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    cases = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        layers = []
        hooks = [model.model.embed_tokens.register_forward_hook(lambda m,a,y: layers.append(y.float().reshape(-1).tolist()))]
        for layer in model.model.layers:
            hooks.append(layer.register_forward_hook(lambda m,a,y: layers.append(y.float().reshape(-1).tolist())))
        result = model(torch.tensor([ids]), use_cache=True, logits_to_keep=1)
        for hook in hooks: hook.remove()
        case = {"prompt":prompt,"prompt_ids":ids,"layer_values":layers,
                "layer_norms":[float(torch.tensor(x).norm()) for x in layers],"steps":[]}
        cache = result.past_key_values
        sequence = list(ids)
        for i in range(steps):
            logits = result.logits[0,-1].float()
            top = torch.topk(logits,5)
            token = int(logits.argmax())
            case["steps"].append({"token":token,"top":[[int(j),float(v)] for v,j in zip(top.values,top.indices)],"logits":logits.tolist()})
            # Cached decode must agree with a fresh pass under the original tensor tolerances.
            fresh = model(torch.tensor([sequence]),use_cache=False, logits_to_keep=1).logits[0,-1].float()
            diff = (fresh-logits).double()
            normalized = float(diff.abs().max()/fresh.double().abs().max().clamp_min(1))
            relative = float(diff.square().mean().sqrt()/fresh.double().square().mean().sqrt().clamp_min(1e-6))
            assert normalized <=1e-4 and relative<=1e-5, (prompt,i,normalized,relative)
            assert int(fresh.argmax()) == token
            sequence.append(token)
            if i+1 < steps:
                result = model(torch.tensor([[token]]),past_key_values=cache,use_cache=True, logits_to_keep=1)
        print(prompt, "=>", tokenizer.decode([x['token'] for x in case['steps']]),flush=True)
        cases.append(case)
    with out.open("w") as file:
        json.dump({"format":1,"weights":"int8-dequantized" if (path/"marin-int8.json").exists() else "bfloat16","math":"float32","transformers":__version__,"cases":cases},file,separators=(",",":"))
    if scratch:
        scratch.cleanup()


def main():
    p=argparse.ArgumentParser();p.add_argument("model",type=pathlib.Path);p.add_argument("out",type=pathlib.Path)
    p.add_argument("--tiny",action="store_true");p.add_argument("--steps",type=int,default=8)
    p.add_argument("--prefill-lengths", type=int, nargs="+")
    args=p.parse_args()
    if args.tiny: tiny(args.model)
    prompts = [" hello" * n for n in args.prefill_lengths] if args.prefill_lengths else ["The capital of France is","def fibonacci(n):"]
    reference(args.model,args.out,prompts,args.steps)


if __name__=="__main__": main()
