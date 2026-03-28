"""QKV partial RoPE kernel config."""

from __future__ import annotations

import torch

import references


def input_generator(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:  # noqa: ARG001
    torch.manual_seed(seed)
    batch = size["batch"]
    seq_len = size["seq_len"]
    q_heads = size["q_heads"]
    kv_heads = size["kv_heads"]
    head_dim = size["head_dim"]
    nope_dim = size["nope_dim"]
    num_heads = q_heads + 2 * kv_heads
    rope_dim = head_dim - nope_dim

    qkv = torch.randn(batch, seq_len, num_heads, head_dim,
                       dtype=torch.bfloat16, device=device)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    return {
        "qkv": qkv, "cos": cos, "sin": sin,
        "q_heads": q_heads, "kv_heads": kv_heads, "nope_dim": nope_dim,
    }


def reference_fn(inputs: dict) -> torch.Tensor:
    return references.qkv_part_rope_ref(**inputs)


def flops_fn(size: dict) -> int:
    return (
        size["batch"] * size["seq_len"]
        * (size["q_heads"] + size["kv_heads"])
        * (size["head_dim"] - size["nope_dim"])
        * 6
    )


def bytes_fn(size: dict, dtype: torch.dtype) -> int:  # noqa: ARG001  # dtype unused: byte count is hardcoded for bf16
    return (
        size["batch"] * size["seq_len"]
        * (size["q_heads"] + 2 * size["kv_heads"])
        * size["head_dim"] * 2 * 2
        + size["seq_len"]
        * ((size["head_dim"] - size["nope_dim"]) // 2)
        * 4 * 2
    )
