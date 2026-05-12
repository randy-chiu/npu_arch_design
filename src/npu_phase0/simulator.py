"""Functional simulator for the Phase 0 JSON micro-op format."""

from __future__ import annotations

import copy
import math
from typing import Any

from .golden import matmul
from .isa import validate_program


class FunctionalSimulator:
    def __init__(self, arch: dict[str, Any]):
        self.arch = arch
        self.reset()

    def reset(self) -> None:
        self.dram: dict[str, Any] = {}
        self.buffers: dict[str, Any] = {}
        self.scalars: dict[str, Any] = {}
        self.counters = {
            "instructions": 0,
            "dma_transfers": 0,
            "dma_elements": 0,
            "mac_ops": 0,
            "vector_ops": 0,
        }

    def run(self, artifact: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        self.reset()
        self.dram.update(copy.deepcopy(inputs))
        program = artifact["program"]
        validate_program(program, self.arch)

        for inst in program:
            self.counters["instructions"] += 1
            op = inst["op"]
            if op == "LOAD":
                self.buffers[inst["buffer"]] = copy.deepcopy(self.dram[inst["tensor"]])
                self._count_dma(self.buffers[inst["buffer"]])
            elif op == "STORE":
                self.dram[inst["tensor"]] = copy.deepcopy(self.buffers[inst["buffer"]])
                self._count_dma(self.buffers[inst["buffer"]])
            elif op == "MATMUL":
                a = self.buffers[inst["a"]]
                b = self.buffers[inst["b"]]
                self.buffers[inst["out"]] = matmul(a, b)
                shape = inst["shape"]
                self.counters["mac_ops"] += shape["m"] * shape["n"] * shape["k"]
            elif op == "VREDMAX":
                self.scalars[inst["dst"]] = [max(row) for row in self.buffers[inst["src"]]]
                self.counters["vector_ops"] += self._num_elements(self.buffers[inst["src"]])
            elif op == "VSUB":
                src = self.buffers[inst["src"]]
                scalar = self.scalars[inst["scalar"]]
                self.buffers[inst["dst"]] = [
                    [float(v) - float(scalar[i]) for v in row] for i, row in enumerate(src)
                ]
                self.counters["vector_ops"] += self._num_elements(src)
            elif op == "VEXP":
                src = self.buffers[inst["src"]]
                self.buffers[inst["dst"]] = [[math.exp(float(v)) for v in row] for row in src]
                self.counters["vector_ops"] += self._num_elements(src)
            elif op == "VREDSUM":
                self.scalars[inst["dst"]] = [sum(row) for row in self.buffers[inst["src"]]]
                self.counters["vector_ops"] += self._num_elements(self.buffers[inst["src"]])
            elif op == "VDIV":
                src = self.buffers[inst["src"]]
                scalar = self.scalars[inst["scalar"]]
                self.buffers[inst["dst"]] = [
                    [float(v) / float(scalar[i]) for v in row] for i, row in enumerate(src)
                ]
                self.counters["vector_ops"] += self._num_elements(src)
            elif op == "HALT":
                break
            else:
                raise ValueError(f"unsupported op: {op}")

        return {
            "dram": copy.deepcopy(self.dram),
            "counters": dict(self.counters),
        }

    def _count_dma(self, value: Any) -> None:
        self.counters["dma_transfers"] += 1
        self.counters["dma_elements"] += self._num_elements(value)

    def _num_elements(self, value: Any) -> int:
        if isinstance(value, list):
            return sum(self._num_elements(item) for item in value)
        return 1

