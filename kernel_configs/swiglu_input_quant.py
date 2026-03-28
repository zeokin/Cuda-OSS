"""SwiGLU + input FP8 quantization kernel config."""

from __future__ import annotations

import torch

import references

from ._utils import dtype_bytes


def input_generator(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N = size["M"], size["N"]
    x = torch.randn(M, N * 2, dtype=dtype, device=device)
    return {"x": x}


def reference_fn(inputs: dict):
    return references.swiglu_input_quant_ref(inputs["x"])


def flops_fn(size: dict) -> int:
    return 13 * size["M"] * size["N"]


def bytes_fn(size: dict, dtype: torch.dtype) -> int:
    eb = dtype_bytes(dtype)
    return (
        size["M"] * size["N"] * 2 * eb
        + size["M"] * size["N"] * eb
        + size["M"] * size["N"] * 2 * 1
        + (size["N"] * 2 // 128) * size["M"] * 4
    )
