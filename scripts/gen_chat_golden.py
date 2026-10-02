#!/usr/bin/env python3
"""Records what Hugging Face's apply_chat_template produces for Qwen3 (non-thinking mode),
so the Rust formatter in crates/chat can be checked against the official behavior.

  gen_chat_golden.py ~/models/Qwen3-1.7B tests/reference/qwen3-chat-golden.json
"""
import json
import pathlib
import sys

from transformers import AutoTokenizer

SYSTEM = "You are Marin, a private assistant running on the user's own machine."
CONVERSATIONS = {
    "single user turn": [{"role": "user", "content": "hi"}],
    "system and user": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "who are you?"}],
    "multi turn": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "explain what a mutex is briefly"},
        {"role": "assistant", "content": "A mutex lets only one thread use a resource at a time."},
        {"role": "user", "content": "2+2?"},
    ],
    "reasoning in an earlier reply is dropped": [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>\nsome plan\n</think>\n\nThe answer."},
        {"role": "user", "content": "q2"},
    ],
    "multiline and unicode": [{"role": "user", "content": "line one\nline two\n\nnaïve café ☕ 你好"}],
}


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cases = []
    for name, messages in CONVERSATIONS.items():
        kwargs = dict(add_generation_prompt=True, enable_thinking=False)
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        assert tokenizer.encode(text, add_special_tokens=False) == list(ids)
        cases.append({"name": name, "messages": messages, "text": text, "ids": list(ids)})
    meta = {
        "eos_token": tokenizer.eos_token,
        "special_tokens": {t: tokenizer.convert_tokens_to_ids(t) for t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]},
    }
    pathlib.Path(out).write_text(json.dumps({"format": 1, "meta": meta, "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
