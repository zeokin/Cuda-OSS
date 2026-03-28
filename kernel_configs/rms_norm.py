"""RMS norm kernel config: input generator, reference, flops/bytes functions."""

from __future__ import annotations

import torch

import references

from ._utils import dtype_bytes


def input_generator(size: dict, dtype: torch.dtype, device: str, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    M, N = size["M"], size["N"]
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    return {"x": x, "weight": weight}


def reference_fn(inputs: dict) -> torch.Tensor:
    return references.rms_norm_ref(inputs["x"], inputs["weight"])


def flops_fn(size: dict) -> int:
    return 6 * size["M"] * size["N"]


def bytes_fn(size: dict, dtype: torch.dtype) -> int:
    eb = dtype_bytes(dtype)
    return (2 * size["M"] * size["N"] + size["N"]) * eb
