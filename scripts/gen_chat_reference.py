#!/usr/bin/env python3
"""Hugging Face reference for the `marin` chat path (Qwen3 non-thinking mode, float32 math).

  gen_chat_reference.py ~/models/Qwen3-1.7B tests/reference/qwen3-chat-differential.json

For every conversation and turn it records: the exact prompt token ids from
apply_chat_template(add_generation_prompt=True, enable_thinking=False), the greedy reply ids, and the
distribution HF would sample from at the first reply step with temperature 0.7, top_k 20, top_p 0.8.
"""
import json
import pathlib
import sys

import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_reference import load_float32_math  # noqa: E402

SYSTEM = "You are Marin, a private assistant that runs on the user's own machine. Be friendly, clear and concise."
CONVERSATIONS = [
    ("hi", ["hi"]),
    ("how's life?", ["how's life?"]),
    ("what do you do?", ["what do you do?"]),
    ("greeting then slang", ["hi", "wsg"]),
    ("greeting then how's life", ["hi", "how's life?"]),
    ("alice", ["my name is alice", "what name did i tell you?"]),
    ("secret number", ["the secret number is 417", "what was the secret number?"]),
]
SAMPLING = {"temperature": 0.7, "top_k": 20, "top_p": 0.8}
MAX_NEW = 64
STOP = [151645, 151643]


def sampling_distribution(logits):
    scores = logits.unsqueeze(0)
    ids = torch.zeros((1, 1), dtype=torch.long)
    for warper in (
        TemperatureLogitsWarper(SAMPLING["temperature"]),
        TopKLogitsWarper(SAMPLING["top_k"]),
        TopPLogitsWarper(SAMPLING["top_p"]),
    ):
        scores = warper(ids, scores)
    probs = torch.softmax(scores[0], dim=-1)
    keep = torch.nonzero(probs > 0).flatten()
    return sorted(([int(i), float(probs[i])] for i in keep), key=lambda p: -p[1])


@torch.no_grad()
def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_float32_math(model_dir)
    cases = []
    for name, user_turns in CONVERSATIONS:
        messages = [{"role": "system", "content": SYSTEM}]
        turns = []
        for user in user_turns:
            messages.append({"role": "user", "content": user})
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
            prompt_ids = encoded["input_ids"]
            logits = model(prompt_ids).logits[0, -1]
            generated = model.generate(prompt_ids, do_sample=False, max_new_tokens=MAX_NEW, eos_token_id=STOP, pad_token_id=151643)
            reply_ids = [int(i) for i in generated[0, prompt_ids.shape[1]:]]
            stopped = bool(reply_ids) and reply_ids[-1] in STOP
            if stopped:
                reply_ids = reply_ids[:-1]
            reply = tokenizer.decode(reply_ids, skip_special_tokens=False)
            turns.append({
                "user": user,
                "prompt_ids": [int(i) for i in prompt_ids[0]],
                "greedy_ids": reply_ids,
                "greedy_text": reply,
                "stopped_on_eos": stopped,
                "step0_distribution": sampling_distribution(logits),
            })
            messages.append({"role": "assistant", "content": reply})
        cases.append({"name": name, "turns": turns})
    pathlib.Path(out).write_text(json.dumps({
        "format": 1, "system": SYSTEM, "sampling": SAMPLING, "max_new_tokens": MAX_NEW, "stop_tokens": STOP, "cases": cases,
    }, indent=1, ensure_ascii=False) + "\n")
    for case in cases:
        for turn in case["turns"]:
            top = ", ".join(f"{tokenizer.decode([i])!r} {p:.2f}" for i, p in turn["step0_distribution"][:4])
            print(f"   first-token odds at sampling settings: {top}")
            print(f"{turn['user']!r:36} -> {len(turn['greedy_ids']):3} tokens, eos={turn['stopped_on_eos']}: {turn['greedy_text'][:70]!r}")


if __name__ == "__main__":
    main()
