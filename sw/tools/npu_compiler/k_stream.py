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


def plan_tiled_matmul(
    a: list[list[int]],
    b: list[list[int]],
    *,
    tile_m: int = 8,
    tile_n: int = 8,
    tile_k: int = 8,
) -> dict[str, Any]:
    """Lower one logical matmul into one K-stream job per output tile.

    Boundary tiles are zero-filled. The returned jobs are Compiler output; the
    Runtime is expected only to bind and submit them.
    """

    if not a or not b or not a[0] or not b[0]:
        raise ValueError("matmul inputs must be non-empty")
    m_size = len(a)
    k_size = len(a[0])
    n_size = len(b[0])
    if any(len(row) != k_size for row in a):
        raise ValueError("A rows must have a consistent K dimension")
    if len(b) != k_size or any(len(row) != n_size for row in b):
        raise ValueError("B shape must match A K dimension")

    padded_m = _round_up(m_size, tile_m)
    padded_n = _round_up(n_size, tile_n)
    padded_k = _round_up(k_size, tile_k)
    padded_a = [
        [int(a[row][col]) if row < m_size and col < k_size else 0 for col in range(padded_k)]
        for row in range(padded_m)
    ]
    padded_b = [
        [int(b[row][col]) if row < k_size and col < n_size else 0 for col in range(padded_n)]
        for row in range(padded_k)
    ]
    expected = matmul(a, b)
    jobs = []
    for m_offset in range(0, padded_m, tile_m):
        a_rows = padded_a[m_offset : m_offset + tile_m]
        for n_offset in range(0, padded_n, tile_n):
            k_plan = plan_matmul_k_stream(
                a_rows,
                padded_b,
                n_offset=n_offset,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
            )
            valid_m = min(tile_m, m_size - m_offset)
            valid_n = min(tile_n, n_size - n_offset)
            valid_k = min(tile_k, k_size % tile_k or tile_k)
            jobs.append(
                {
                    "job_index": len(jobs),
                    "descriptor_op": "matmul_k_stream",
                    "m_offset": m_offset,
                    "n_offset": n_offset,
                    "valid_m": valid_m,
                    "valid_n": valid_n,
                    "last_k_valid": valid_k,
                    **k_plan,
                }
            )
    physical_invocations = len(jobs) * (padded_k // tile_k)
    return {
        "logical_shape": {"m": m_size, "n": n_size, "k": k_size},
        "tile_shape": {"m": tile_m, "n": tile_n, "k": tile_k},
        "output_tile_jobs": jobs,
        "output_tile_count": len(jobs),
        "physical_tile_invocations": physical_invocations,
        "theoretical_matrix_cycles": physical_invocations * tile_k,
        "useful_mac_ops": m_size * n_size * k_size,
        "issued_mac_capacity": physical_invocations * tile_m * tile_n * tile_k,
        "expected_c": expected,
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


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("tile dimensions must be positive")
    return ((value + multiple - 1) // multiple) * multiple
