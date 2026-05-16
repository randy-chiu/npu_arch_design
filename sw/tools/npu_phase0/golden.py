"""CPU golden operators for Phase 0."""

from __future__ import annotations

import math
from typing import List, Union


Number = Union[float, int]
Matrix = List[List[Number]]


def matmul(a: Matrix, b: Matrix) -> list[list[int]]:
    if not a or not b or not b[0]:
        raise ValueError("matmul inputs must be non-empty 2D matrices")
    m = len(a)
    k = len(a[0])
    if any(len(row) != k for row in a):
        raise ValueError("left matrix is ragged")
    if any(len(row) != len(b[0]) for row in b):
        raise ValueError("right matrix is ragged")
    if len(b) != k:
        raise ValueError("matmul shape mismatch")

    n = len(b[0])
    out: list[list[int]] = []
    for i in range(m):
        row: list[int] = []
        for j in range(n):
            acc = 0
            for kk in range(k):
                acc += int(a[i][kk]) * int(b[kk][j])
            row.append(acc)
        out.append(row)
    return out


def softmax(x: Matrix) -> list[list[float]]:
    if not x:
        raise ValueError("softmax input must be non-empty")
    out: list[list[float]] = []
    for row in x:
        if not row:
            raise ValueError("softmax rows must be non-empty")
        max_v = max(float(v) for v in row)
        exps = [math.exp(float(v) - max_v) for v in row]
        denom = sum(exps)
        out.append([v / denom for v in exps])
    return out


def assert_close(actual: Matrix, expected: Matrix, abs_tol: float) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"row count mismatch: {len(actual)} != {len(expected)}")
    for i, (a_row, e_row) in enumerate(zip(actual, expected)):
        if len(a_row) != len(e_row):
            raise AssertionError(f"column count mismatch at row {i}")
        for j, (a, e) in enumerate(zip(a_row, e_row)):
            if abs(float(a) - float(e)) > abs_tol:
                raise AssertionError(f"value mismatch at ({i}, {j}): {a} != {e}")
