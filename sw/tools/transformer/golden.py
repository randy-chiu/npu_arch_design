from __future__ import annotations


TILE_M = 8
TILE_N = 8
TILE_K = 8
TILE_WORDS = TILE_M * TILE_N


def matmul_i8_i32(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    if not a or not b:
        raise ValueError("matmul inputs must be non-empty")
    k_dim = len(a[0])
    if any(len(row) != k_dim for row in a):
        raise ValueError("A rows must have a consistent K dimension")
    if len(b) != k_dim:
        raise ValueError("B row count must match A K dimension")
    n_dim = len(b[0])
    if any(len(row) != n_dim for row in b):
        raise ValueError("B rows must have a consistent N dimension")
    return [
        [sum(int(a[row][k]) * int(b[k][col]) for k in range(k_dim)) for col in range(n_dim)]
        for row in range(len(a))
    ]


def flatten_tile(tile: list[list[int]]) -> list[int]:
    return [int(value) for row in tile for value in row]


def deterministic_i8_matrix(rows: int, cols: int, seed: int) -> list[list[int]]:
    matrix: list[list[int]] = []
    for row in range(rows):
        values = []
        for col in range(cols):
            raw = (row * 17 + col * 31 + seed * 13) % 17
            values.append(raw - 8)
        matrix.append(values)
    return matrix


def tile_k_stream(a: list[list[int]], b: list[list[int]]) -> dict[str, list]:
    if len(a) != TILE_M or len(b[0]) != TILE_N:
        raise ValueError("current fixture generator expects one 8x8 output tile")
    if len(a[0]) != len(b):
        raise ValueError("A and B K dimensions differ")
    k_dim = len(a[0])
    if k_dim % TILE_K != 0:
        raise ValueError("K dimension must be a multiple of 8 for current K-stream fixtures")

    a_stream = []
    b_stream = []
    k_offsets = []
    for k_offset in range(0, k_dim, TILE_K):
        k_offsets.append(k_offset)
        a_chunk = [row[k_offset : k_offset + TILE_K] for row in a]
        b_chunk = b[k_offset : k_offset + TILE_K]
        a_stream.append(flatten_tile(a_chunk))
        b_stream.append(flatten_tile(b_chunk))

    expected_c = flatten_tile(matmul_i8_i32(a, b))
    return {
        "k_chunks": len(k_offsets),
        "k_offsets": k_offsets,
        "a_stream": a_stream,
        "b_stream": b_stream,
        "expected_c": expected_c,
    }
