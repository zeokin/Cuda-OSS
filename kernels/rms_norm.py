"""Baseline RMS normalization kernel (Triton).

One Triton program per row. Each program loads the full row,
reduces to a single RMS scalar, then writes the row back with
the normalization and per-column weight applied.

Computation is in fp32 internally regardless of input dtype, to
match the standard RMS-norm precision discipline (bf16/fp16 squares
underflow).
"""

import torch
import triton
import triton.language as tl

KERNEL_TYPE = "rms_norm"


@triton.jit
def _rms_norm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xm,
    stride_ym,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N
    rms = tl.sqrt(mean_sq + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x / rms) * w

    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.dim() == 2 and weight.dim() == 1 and weight.shape[0] == x.shape[1]
    M, N = x.shape

    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    _rms_norm_kernel[grid](
        x, weight, y,
        x.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def get_inputs() -> dict:
    torch.manual_seed(0)
    M, N = 2048, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    return {"x": x, "weight": weight}


def get_flops() -> int:
    M, N = 2048, 4096
    return 6 * M * N


def get_bytes() -> int:
    M, N = 2048, 4096
    return (2 * M * N + N) * 2
