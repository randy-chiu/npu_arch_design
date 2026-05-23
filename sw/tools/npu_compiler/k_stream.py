"""K-axis streaming matmul planning helpers for the Phase 0 NPU."""

from __future__ import annotations

from typing import Any

from npu_phase0.golden import matmul


def plan_matmul_k_stream(
    a: list[list[int]],
    b: list[list[int]],
    *,
    n_offset: int = 0,
    k_offsets: list[int] | None = None,
    max_chunks: int | None = None,
    require_nonzero: bool = False,
    tile_m: int = 8,
    tile_n: int = 8,
    tile_k: int = 8,
) -> dict[str, Any]:
    """Build packed K-stream chunks for one logical output N tile.

    The current hardware executes one physical ``tile_m x tile_k`` by
    ``tile_k x tile_n`` matmul per chunk, then accumulates all chunk results
    into the same output tile.
    """

    _validate_inputs(a, b, n_offset=n_offset, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    if k_offsets is None:
        selected_offsets = list(range(0, len(b), tile_k))
    else:
        selected_offsets = list(k_offsets)

    chunks = []
    expected_c = [[0 for _ in range(tile_n)] for _ in range(tile_m)]
    for k_offset in selected_offsets:
        if k_offset < 0 or k_offset + tile_k > len(b) or k_offset % tile_k != 0:
            raise ValueError(f"invalid k_offset {k_offset} for K={len(b)} and tile_k={tile_k}")

        a_tile = [row[k_offset : k_offset + tile_k] for row in a]
        b_tile = [row[n_offset : n_offset + tile_n] for row in b[k_offset : k_offset + tile_k]]
        expected_tile = matmul(a_tile, b_tile)
        if require_nonzero and not _is_useful_nonzero_chunk(a_tile, expected_tile):
            continue

        chunks.append(
            {
                "k_offset": k_offset,
                "a_tile": a_tile,
                "b_tile": b_tile,
                "expected_tile": expected_tile,
            }
        )
        for row in range(tile_m):
            for col in range(tile_n):
                expected_c[row][col] += expected_tile[row][col]

        if max_chunks is not None and len(chunks) >= max_chunks:
            break

    if max_chunks is not None and len(chunks) < max_chunks:
        raise ValueError(f"could not plan {max_chunks} K-stream chunks")
    if not chunks:
        raise ValueError("could not plan any K-stream chunks")

    return {
        "m": tile_m,
        "n": tile_n,
        "k_step": tile_k,
        "n_offset": n_offset,
        "k_chunks": len(chunks),
        "input0_words": tile_m * tile_k,
        "input1_words": tile_k * tile_n,
        "output_words": tile_m * tile_n,
        "k_offsets": [chunk["k_offset"] for chunk in chunks],
        "chunks": chunks,
        "a_stream": [chunk["a_tile"] for chunk in chunks],
        "b_stream": [chunk["b_tile"] for chunk in chunks],
        "expected_c": expected_c,
    }


def _validate_inputs(
    a: list[list[int]],
    b: list[list[int]],
    *,
    n_offset: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
        raise ValueError("tile dimensions must be positive")
    if len(a) != tile_m:
        raise ValueError(f"A must have {tile_m} rows, got {len(a)}")
    if not b:
        raise ValueError("B must not be empty")
    if len(b) % tile_k != 0:
        raise ValueError(f"K={len(b)} must be divisible by tile_k={tile_k}")
    if n_offset < 0 or n_offset % tile_n != 0:
        raise ValueError(f"n_offset must be a non-negative multiple of tile_n={tile_n}")

    k_size = len(b)
    for row_index, row in enumerate(a):
        if len(row) < k_size:
            raise ValueError(f"A row {row_index} has {len(row)} columns, expected at least {k_size}")
    for row_index, row in enumerate(b):
        if len(row) < n_offset + tile_n:
            raise ValueError(
                f"B row {row_index} has {len(row)} columns, expected at least {n_offset + tile_n}"
            )


def _is_useful_nonzero_chunk(a_tile: list[list[int]], expected_tile: list[list[int]]) -> bool:
    return any(value != 0 for row in a_tile for value in row) and any(
        value != 0 for row in expected_tile for value in row
    )
