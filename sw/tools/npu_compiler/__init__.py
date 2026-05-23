"""Host-side NPU compiler package."""

from .k_stream import plan_matmul_k_stream
from .phase0 import compile_graph

__all__ = ["compile_graph", "plan_matmul_k_stream"]
