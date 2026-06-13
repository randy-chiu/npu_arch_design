"""Host-side NPU compiler package."""

from .attention import build_attention_mask_plan, build_attention_plan_from_manifest, build_attention_prefill_plan_s8_d8
from .k_stream import plan_matmul_k_stream
from .phase0 import compile_graph

__all__ = [
    "build_attention_mask_plan",
    "build_attention_plan_from_manifest",
    "build_attention_prefill_plan_s8_d8",
    "compile_graph",
    "plan_matmul_k_stream",
]
