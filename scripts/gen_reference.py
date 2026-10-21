#!/usr/bin/env python3
"""Generates Hugging Face reference outputs that `marin-run verify` compares against.

  gen_reference.py tiny tests/fixtures/tiny-qwen3
  gen_reference.py reference ~/models/Qwen3-1.7B-Base ref.json [--steps 16] [--prompt TEXT ...]
  gen_reference.py tiny-chat tests/fixtures/tiny-chat tests/fixtures/tiny-qwen3 ~/models/Qwen3-1.7B

Weights stay in bf16 on disk and in memory; each layer upcasts to float32 on the fly,
so the math matches marin (float32) without needing 2x the RAM.
"""
import argparse
import json
import pathlib
import shutil

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

TOP_K = 5
DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village",
]
TINY_PROMPTS = ["the cat sat on the mat", "hello world", "naïve café ☕"]
CORPUS = [
    "the cat sat on the mat and the dog sat on the log",
    "hello world, this is a small tokenizer for tests",
    "the capital of france is paris and the capital of italy is rome",
]


def load_float32_math(path):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = lambda x, m=module: F.linear(x.float(), m.weight.float())
        elif isinstance(module, torch.nn.Embedding):
            module.forward = lambda ids, m=module: F.embedding(ids, m.weight).float()
    return model


@torch.no_grad()
def run_case(model, tokenizer, prompt, steps):
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    layers = model.config.num_hidden_layers
    hidden = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
    case = {
        "prompt": prompt,
        "prompt_ids": ids,
        "layer_norms": [float(hidden[i][0, -1].norm()) for i in range(layers)],
        "steps": [],
    }
    sequence = list(ids)
    for _ in range(steps):
        logits = model(torch.tensor([sequence])).logits[0, -1]
        top = torch.topk(logits, TOP_K)
        token = int(torch.argmax(logits))
        case["steps"].append(
            {"token": token, "top": [[int(i), float(v)] for v, i in zip(top.values, top.indices)]}
        )
        sequence.append(token)
    return case


def write_reference(model_dir, tokenizer, out, prompts, steps):
    model = load_float32_math(model_dir)
    cases = [run_case(model, tokenizer, p, steps) for p in prompts]
    reference = {
        "format": 1,
        "weights": "bfloat16",
        "math": "float32",
        "top_k": TOP_K,
        "cases": cases,
    }
    pathlib.Path(out).write_text(json.dumps(reference, indent=1) + "\n")


def build_tiny(out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    tokenizer.save(str(out / "tokenizer.json"))

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=320,  # larger than the tokenizer's 300 on purpose, like the real model
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        initializer_range=0.3,
        eos_token_id=0,
        bos_token_id=0,
    )
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.2 * torch.randn_like(param))
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    write_reference(out, tokenizer, out / "reference.json", TINY_PROMPTS, steps=6)


def build_tiny_chat(out_dir, tiny_dir, official_dir):
    """The tiny random model with a ChatML tokenizer and the official Qwen3 chat metadata."""
    from tokenizers import AddedToken

    out, tiny, official = pathlib.Path(out_dir), pathlib.Path(tiny_dir), pathlib.Path(official_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(tiny / "model.safetensors", out / "model.safetensors")
    config = json.loads((tiny / "config.json").read_text())
    config["max_position_embeddings"] = 1024  # random weights have no position table, so a longer window is free
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=295, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(CORPUS + ["hi there", "who are you?", "2+2?"], trainer)
    tokenizer.add_tokens([AddedToken("<think>", special=False), AddedToken("</think>", special=False)])
    tokenizer.save(str(out / "tokenizer.json"))
    ids = {t: tokenizer.token_to_id(t) for t in specials}
    official_config = json.loads((official / "tokenizer_config.json").read_text())
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "chat_template": official_config["chat_template"]},
            indent=1,
        )
        + "\n"
    )
    (out / "generation_config.json").write_text(
        json.dumps(
            {"bos_token_id": ids["<|endoftext|>"], "eos_token_id": [ids["<|im_end|>"], ids["<|endoftext|>"]]}, indent=1
        )
        + "\n"
    )
    print(ids, "vocab", tokenizer.get_vocab_size())


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("tiny").add_argument("out_dir")
    chat = sub.add_parser("tiny-chat")
    chat.add_argument("out_dir")
    chat.add_argument("tiny_dir")
    chat.add_argument("official_dir")
    ref = sub.add_parser("reference")
    ref.add_argument("model_dir")
    ref.add_argument("out")
    ref.add_argument("--steps", type=int, default=16)
    ref.add_argument("--prompt", action="append")
    args = parser.parse_args()
    if args.mode == "tiny":
        build_tiny(args.out_dir)
    elif args.mode == "tiny-chat":
        build_tiny_chat(args.out_dir, args.tiny_dir, args.official_dir)
    else:
        tokenizer = Tokenizer.from_file(str(pathlib.Path(args.model_dir) / "tokenizer.json"))
        write_reference(args.model_dir, tokenizer, args.out, args.prompt or DEFAULT_PROMPTS, args.steps)


if __name__ == "__main__":
    main()
