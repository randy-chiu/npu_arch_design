"""Host-side NPU compiler package."""

from .phase0 import compile_graph

__all__ = ["compile_graph"]
