#!/usr/bin/env python3
"""Regenerate CPU FP32 fixtures using official Transformers 5.17.0 modules.
Run with the project's HF reference environment; no model payload download.
"""
import hashlib
import inspect
import json
import pathlib
import struct

import torch
from transformers import __version__
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

OUT = pathlib.Path(__file__).with_name("blocks.json")
torch.set_num_threads(1)
torch.manual_seed(35)
source = pathlib.Path(inspect.getfile(hf))
assert __version__ == "5.17.0", "review reference changes before regenerating"


def values(x):
    return x.detach().float().reshape(-1).tolist()


def inputs(n, h, offset=0):
    return (torch.sin(torch.arange(n * h, dtype=torch.float32) * .31 + offset)
            + .17 * torch.cos(torch.arange(n * h, dtype=torch.float32) * .71)).reshape(1, n, h)


def config(ratio=1, wide=False):
    return hf.Qwen3_5TextConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=256 if wide else 8,
        linear_num_key_heads=2, linear_num_value_heads=2 * ratio,
        linear_key_head_dim=128 if wide else 3,
        linear_value_head_dim=128 if wide else 4, linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6, layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                         "partial_rotary_factor": .25 if wide else .5,
                         "mrope_section": [11, 11, 10] if wide else [1, 1, 0]},
    )


def build(c, idx):
    c._attn_implementation = "eager"
    block = hf.Qwen3_5DecoderLayer(c, idx).float().eval()
    with torch.no_grad():
        for i, (name, p) in enumerate(block.named_parameters()):
            v = torch.sin(torch.arange(p.numel()).float() * .47 + i * .37).reshape(p.shape)
            if name.endswith("A_log"):
                p.copy_(-2 + v * .2)
            elif name == "linear_attn.norm.weight":
                p.copy_(1 + v * .2)
            else:
                p.copy_(v * (.2 if "conv1d" in name else .13))
    return block


def run(block, c, idx, x, cache):
    start = cache.get_seq_length(3) if idx == 3 else 0
    # Linear position unused; for hybrid the caller supplies explicit full-block positions.
    pos = torch.arange(start, start + x.shape[1]).reshape(1, -1).expand(3, 1, -1)
    emb = hf.Qwen3_5TextRotaryEmbedding(c)(x, pos)
    mask = torch.zeros(1, 1, x.shape[1], start + x.shape[1])
    mask.masked_fill_(torch.arange(start + x.shape[1])[None, :] >
                      torch.arange(start, start + x.shape[1])[:, None], float("-inf"))
    return block(x, position_embeddings=emb, attention_mask=mask if idx == 3 else None,
                 past_key_values=cache)


def case(block, c, idx, n, prefix_len):
    cache = DynamicCache(config=c)
    prefix = inputs(prefix_len, c.hidden_size, .91)
    if prefix_len:
        run(block, c, idx, prefix, cache)
    expected = {}
    handles = []
    def record(name, x):
        expected[name] = values(x)
    def hook(name):
        return lambda _, args, out: record(name, out)
    names = {"input_layernorm": "input_norm", "post_attention_layernorm": "post_norm",
             "mlp.gate_proj": "mlp_gate", "mlp.up_proj": "mlp_up", "mlp.down_proj": "mlp"}
    if idx == 0:
        names.update({"linear_attn.in_proj_qkv": "qkv", "linear_attn.in_proj_z": "z",
                      "linear_attn.in_proj_a": "a", "linear_attn.in_proj_b": "b",
                      "linear_attn.norm": "gated", "linear_attn.out_proj": "mixer"})
    else:
        names.update({"self_attn.q_proj": "q_gate", "self_attn.k_proj": "k_proj",
                      "self_attn.v_proj": "v_proj", "self_attn.q_norm": "q_norm",
                      "self_attn.k_norm": "k_norm", "self_attn.o_proj": "mixer"})
    for name, module in block.named_modules():
        if name in names:
            handles.append(module.register_forward_hook(hook(names[name])))
    handles.append(block.mlp.down_proj.register_forward_pre_hook(lambda _, args: record("mlp_act", args[0])))
    if idx == 3:
        handles.append(block.self_attn.o_proj.register_forward_pre_hook(lambda _, args: record("gated", args[0])))
    original = {name: getattr(hf, name) for name in
                ["causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
                 "torch_recurrent_gated_delta_rule", "apply_rotary_pos_emb", "eager_attention_forward"]}
    def conv(fn, *args, **kwargs):
        full_out = original[fn](*args, **kwargs)
        out = full_out
        out = out[:, :, -n:]
        record("conv", out.transpose(1, 2))
        q, k, v = out.transpose(1, 2).split([block.linear_attn.key_dim] * 2 + [block.linear_attn.value_dim], -1)
        for name, data in [("query", q), ("key", k), ("value", v)]: record(name, data)
        return full_out
    def delta(fn, *args, **kwargs):
        record("g", kwargs["g"]); record("beta", kwargs["beta"])
        out, state = original[fn](*args, **kwargs)
        record("core", out)
        return out, state
    def rope(*args, **kwargs):
        q, k = original["apply_rotary_pos_emb"](*args, **kwargs)
        record("q_rope", q.transpose(1, 2)); record("k_rope", k.transpose(1, 2))
        return q, k
    def attend(*args, **kwargs):
        out, attn = original["eager_attention_forward"](*args, **kwargs)
        record("core", out)
        return out, attn
    hf.causal_conv1d_fn = lambda *a, **k: conv("causal_conv1d_fn", *a, **k)
    hf.causal_conv1d_update = lambda *a, **k: conv("causal_conv1d_update", *a, **k)
    hf.torch_chunk_gated_delta_rule = lambda *a, **k: delta("torch_chunk_gated_delta_rule", *a, **k)
    hf.torch_recurrent_gated_delta_rule = lambda *a, **k: delta("torch_recurrent_gated_delta_rule", *a, **k)
    hf.apply_rotary_pos_emb = rope
    hf.eager_attention_forward = attend
    x = inputs(n, c.hidden_size, .27)
    try:
        out = run(block, c, idx, x, cache)
        record("output", out)
        record("residual", x + torch.tensor(expected["mixer"]).reshape_as(x))
    finally:
        for name, fn in original.items(): setattr(hf, name, fn)
        for h in handles: h.remove()
    state = cache.layers[idx]
    return {"prefix": values(prefix), "input": values(x), "expected": expected,
            "conv_state": values(state.conv_states[0]) if idx == 0 else [],
            "recurrent": values(state.recurrent_states[0]) if idx == 0 else [],
            "keys": values(state.keys.transpose(1, 2)) if idx == 3 else [],
            "values": values(state.values.transpose(1, 2)) if idx == 3 else []}


@torch.no_grad()
def main():
    blocks = []
    for ratio, wide in [(1, False), (2, False), (1, True)]:
        c = config(ratio, wide)
        for idx in [0, 3]:
            block = build(c, idx)
            cases = [case(block, c, idx, n, 4 if n in (2, 65) else 0)
                     for n in ([1, 65] if wide else [1, 2, 4, 16, 65, 128])]
            cases.append(case(block, c, idx, 1, 4))
            blocks.append({"config": c.to_dict(), "kind": c.layer_types[idx],
                           "weights": {name: values(p) for name, p in block.named_parameters()}, "cases": cases})
    c = config()
    chain = [build(c, idx) for idx in range(4)]
    # Reuse the same validated linear weights to exercise four independent state owners.
    for block in chain[:3]: block.load_state_dict(chain[0].state_dict())
    x = inputs(65, c.hidden_size, .27)
    hybrid = []
    cache = DynamicCache(config=c)
    pos = torch.arange(65).reshape(1, -1).expand(3, 1, -1)
    emb = hf.Qwen3_5TextRotaryEmbedding(c)(x, pos)
    mask = torch.full((1, 1, 65, 65), float("-inf")).triu(1)
    for block in chain:
        x = block(x, position_embeddings=emb, attention_mask=mask if block.block_type == "full_attention" else None, past_key_values=cache)
        hybrid.append(values(x))
    result = {"transformers": __version__, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "model_revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c", "weights_dtype": "float32",
              "blocks": blocks, "hybrid": hybrid}
    payload = bytearray()
    def packed(x):
        offset = len(payload)
        payload.extend(struct.pack(f"<{len(x)}f", *x))
        return {"offset": offset, "len": len(x)}
    for b in blocks:
        b["weights"] = {name: packed(x) for name, x in b["weights"].items()}
        for c in b["cases"]:
            c["expected"] = {name: packed(x) for name, x in c["expected"].items()}
            for name in ["prefix", "input", "conv_state", "recurrent", "keys", "values"]:
                c[name] = packed(c[name])
    result["hybrid"] = [packed(x) for x in hybrid]
    result["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    OUT.with_name("values.f32").write_bytes(payload)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(OUT, OUT.stat().st_size, "bytes;", sum(len(b['cases']) for b in blocks), "cases")


if __name__ == "__main__": main()
