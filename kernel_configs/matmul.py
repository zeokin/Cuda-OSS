"""Matmul kernel config: input generator, reference, flops/bytes functions."""

from __future__ import annotations

import torch

import references

from ._utils import dtype_bytes


def input_generator(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N, K = size["M"], size["N"], size["K"]
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    return {"a": a, "b": b}


def reference_fn(inputs: dict) -> torch.Tensor:
    return references.matmul_ref(inputs["a"], inputs["b"])


def flops_fn(size: dict) -> int:
    return 2 * size["M"] * size["N"] * size["K"]


def bytes_fn(size: dict, dtype: torch.dtype) -> int:
    eb = dtype_bytes(dtype)
    return (size["M"] * size["K"] + size["K"] * size["N"] + size["M"] * size["N"]) * eb
