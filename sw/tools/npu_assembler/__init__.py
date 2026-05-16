"""Host-side NPU assembler package."""

from .phase0 import encode_program, encode_uop

__all__ = ["encode_program", "encode_uop"]
