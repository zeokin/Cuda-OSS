"""Reference implementations for correctness verification.

Each submodule provides a PyTorch-native implementation that the optimized
kernel is checked against. Do NOT modify these files during experiments.
"""

from .dsa_forward import dsa_forward_ref
from .matmul import matmul_ref
from .qkv_part_rope import qkv_part_rope_ref
from .rms_norm import rms_norm_ref
from .swiglu_input_quant import swiglu_input_quant_ref

__all__ = [
    "matmul_ref",
    "rms_norm_ref",
    "swiglu_input_quant_ref",
    "qkv_part_rope_ref",
    "dsa_forward_ref",
]
