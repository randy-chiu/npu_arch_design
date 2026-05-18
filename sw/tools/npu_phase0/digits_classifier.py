"""Deterministic 8x8 digit-classifier workload for Phase 0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .golden import matmul


IMAGE_SIZE = 8
PIXELS = IMAGE_SIZE * IMAGE_SIZE
BATCH_ROWS = 8
CLASS_COLUMNS = 16
REAL_CLASSES = 10
FOREGROUND = 3
BACKGROUND = -1
TILE_M = 8
TILE_N = 8
TILE_K = 8


_DIGIT_GLYPHS = {
    0: [
        "01111110",
        "11000011",
        "11000111",
        "11001011",
        "11010011",
        "11100011",
        "11000011",
        "01111110",
    ],
    1: [
        "00011000",
        "00111000",
        "01111000",
        "00011000",
        "00011000",
        "00011000",
        "00011000",
        "01111110",
    ],
    2: [
        "01111110",
        "11000011",
        "00000011",
        "00000110",
        "00011100",
        "00110000",
        "01100000",
        "11111111",
    ],
    3: [
        "01111110",
        "11000011",
        "00000011",
        "00111110",
        "00000011",
        "00000011",
        "11000011",
        "01111110",
    ],
    4: [
        "00000110",
        "00001110",
        "00011110",
        "00110110",
        "01100110",
        "11111111",
        "00000110",
        "00000110",
    ],
    5: [
        "11111111",
        "11000000",
        "11000000",
        "11111110",
        "00000011",
        "00000011",
        "11000011",
        "01111110",
    ],
    6: [
        "00111110",
        "01100000",
        "11000000",
        "11111110",
        "11000011",
        "11000011",
        "11000011",
        "01111110",
    ],
    7: [
        "11111111",
        "00000011",
        "00000110",
        "00001100",
        "00011000",
        "00110000",
        "00110000",
        "00110000",
    ],
    8: [
        "01111110",
        "11000011",
        "11000011",
        "01111110",
        "11000011",
        "11000011",
        "11000011",
        "01111110",
    ],
    9: [
        "01111110",
        "11000011",
        "11000011",
        "11000011",
        "01111111",
        "00000011",
        "00000110",
        "01111100",
    ],
}


def classifier_graph() -> dict[str, Any]:
    return {
        "tensors": {
            "A": {"shape": [BATCH_ROWS, PIXELS], "dtype": "int8"},
            "W": {"shape": [PIXELS, CLASS_COLUMNS], "dtype": "int8"},
        },
        "ops": [{"type": "matmul", "a": "A", "b": "W", "out": "Logits"}],
    }


def rtl_tile_graph() -> dict[str, Any]:
    return {
        "tensors": {
            "A": {"shape": [TILE_M, TILE_K], "dtype": "int8"},
            "B": {"shape": [TILE_K, TILE_N], "dtype": "int8"},
        },
        "ops": [{"type": "matmul", "a": "A", "b": "B", "out": "C"}],
    }


def tiny_mlp_graph() -> dict[str, Any]:
    return {
        "description": "Tiny MLP graph. Matmul ops are NPU-visible; relu and argmax are CPU/tool-side for now.",
        "tensors": {
            "A": {"shape": [BATCH_ROWS, PIXELS], "dtype": "int8"},
            "W1": {"shape": [PIXELS, CLASS_COLUMNS], "dtype": "int8"},
            "Hidden": {"shape": [BATCH_ROWS, CLASS_COLUMNS], "dtype": "int32"},
            "HiddenRelu": {"shape": [BATCH_ROWS, CLASS_COLUMNS], "dtype": "int32"},
            "W2": {"shape": [CLASS_COLUMNS, CLASS_COLUMNS], "dtype": "int8"},
            "Logits": {"shape": [BATCH_ROWS, CLASS_COLUMNS], "dtype": "int32"},
        },
        "ops": [
            {"type": "matmul", "a": "A", "b": "W1", "out": "Hidden", "placement": "npu"},
            {"type": "relu_requantize", "x": "Hidden", "out": "HiddenInt8", "placement": "cpu"},
            {"type": "matmul", "a": "HiddenInt8", "b": "W2", "out": "Logits", "placement": "npu"},
            {"type": "argmax", "x": "Logits", "classes": REAL_CLASSES, "out": "Predicted", "placement": "cpu"},
        ],
    }


def classifier_inputs(label: int) -> dict[str, list[list[int]]]:
    return {
        "A": activation_batch(label),
        "W": classifier_weights(),
    }


def classifier_inputs_from_image(path: Path) -> dict[str, list[list[int]]]:
    return {
        "A": activation_batch_from_image(path),
        "W": classifier_weights(),
    }


def glyph_rows(label: int) -> list[str]:
    if label not in _DIGIT_GLYPHS:
        raise ValueError(f"unsupported digit label: {label}")
    return list(_DIGIT_GLYPHS[label])


def load_pgm_image(path: Path) -> list[list[int]]:
    data = path.read_bytes()
    tokens = _pgm_tokens(data)
    if not tokens or tokens[0] != "P2":
        raise ValueError(f"{path} must be an ASCII PGM/P2 image")
    if len(tokens) < 4:
        raise ValueError(f"{path} is missing PGM header fields")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    values = [int(token) for token in tokens[4:]]
    if width != IMAGE_SIZE or height != IMAGE_SIZE:
        raise ValueError(f"{path} must be {IMAGE_SIZE}x{IMAGE_SIZE}, got {width}x{height}")
    if max_value <= 0:
        raise ValueError(f"{path} has invalid max value {max_value}")
    if len(values) != width * height:
        raise ValueError(f"{path} contains {len(values)} pixels, expected {width * height}")
    return [values[row * width : (row + 1) * width] for row in range(height)]


def image_to_glyph_rows(path: Path) -> list[str]:
    pixels = load_pgm_image(path)
    threshold = 127
    return ["".join("1" if value > threshold else "0" for value in row) for row in pixels]


def activation_batch(label: int) -> list[list[int]]:
    image = flatten_digit(label)
    return [image] + [[0 for _ in range(PIXELS)] for _ in range(BATCH_ROWS - 1)]


def activation_batch_from_image(path: Path) -> list[list[int]]:
    image = [
        FOREGROUND if pixel == "1" else BACKGROUND
        for row in image_to_glyph_rows(path)
        for pixel in row
    ]
    return [image] + [[0 for _ in range(PIXELS)] for _ in range(BATCH_ROWS - 1)]


def classifier_weights() -> list[list[int]]:
    columns = [flatten_digit(label) for label in range(REAL_CLASSES)]
    columns.extend([[0 for _ in range(PIXELS)] for _ in range(CLASS_COLUMNS - REAL_CLASSES)])
    return [[columns[col][row] for col in range(CLASS_COLUMNS)] for row in range(PIXELS)]


def tiny_mlp_inputs_from_image(path: Path) -> dict[str, list[list[int]]]:
    return {
        "A": activation_batch_from_image(path),
        "W1": classifier_weights(),
        "W2": tiny_mlp_fc2_weights(),
    }


def tiny_mlp_fc2_weights() -> list[list[int]]:
    return [
        [1 if row == col and col < REAL_CLASSES else 0 for col in range(CLASS_COLUMNS)]
        for row in range(CLASS_COLUMNS)
    ]


def relu(x: list[list[int]]) -> list[list[int]]:
    return [[max(0, int(value)) for value in row] for row in x]


def relu_requantize(x: list[list[int]], scale_shift: int = 2) -> list[list[int]]:
    return [[min(127, max(0, int(value) >> scale_shift)) for value in row] for row in x]


def flatten_digit(label: int) -> list[int]:
    if label not in _DIGIT_GLYPHS:
        raise ValueError(f"unsupported digit label: {label}")
    return [
        FOREGROUND if pixel == "1" else BACKGROUND
        for row in _DIGIT_GLYPHS[label]
        for pixel in row
    ]


def reference_logits(label: int) -> list[list[int]]:
    inputs = classifier_inputs(label)
    return matmul(inputs["A"], inputs["W"])


def reference_logits_from_image(path: Path) -> list[list[int]]:
    inputs = classifier_inputs_from_image(path)
    return matmul(inputs["A"], inputs["W"])


def tiny_mlp_reference_logits_from_image(path: Path) -> list[list[int]]:
    inputs = tiny_mlp_inputs_from_image(path)
    hidden = matmul(inputs["A"], inputs["W1"])
    hidden_int8 = relu_requantize(hidden)
    return matmul(hidden_int8, inputs["W2"])


def lower_classifier_to_rtl_tiles(inputs: dict[str, list[list[int]]]) -> list[dict[str, Any]]:
    a = inputs["A"]
    w = inputs["W"]
    if len(a) != BATCH_ROWS or any(len(row) != PIXELS for row in a):
        raise ValueError("classifier activation tensor must have shape 8x64")
    if len(w) != PIXELS or any(len(row) != CLASS_COLUMNS for row in w):
        raise ValueError("classifier weight tensor must have shape 64x16")

    jobs: list[dict[str, Any]] = []
    graph = rtl_tile_graph()
    for n0 in range(0, CLASS_COLUMNS, TILE_N):
        for k0 in range(0, PIXELS, TILE_K):
            a_tile = [row[k0 : k0 + TILE_K] for row in a]
            b_tile = [row[n0 : n0 + TILE_N] for row in w[k0 : k0 + TILE_K]]
            jobs.append(
                {
                    "n_offset": n0,
                    "k_offset": k0,
                    "graph": graph,
                    "inputs": {"A": a_tile, "B": b_tile},
                }
            )
    return jobs


def lower_matmul_to_rtl_tiles(
    a: list[list[int]],
    b: list[list[int]],
) -> list[dict[str, Any]]:
    if len(a) != TILE_M or any(len(row) % TILE_K for row in a):
        raise ValueError("left matrix must have 8 rows and K multiple of 8")
    if not b or len(b) % TILE_K or any(len(row) % TILE_N for row in b):
        raise ValueError("right matrix must have K multiple of 8 and N multiple of 8")
    if len(a[0]) != len(b):
        raise ValueError("matmul shape mismatch")
    k_total = len(b)
    n_total = len(b[0])
    jobs: list[dict[str, Any]] = []
    graph = rtl_tile_graph()
    for n0 in range(0, n_total, TILE_N):
        for k0 in range(0, k_total, TILE_K):
            a_tile = [row[k0 : k0 + TILE_K] for row in a]
            b_tile = [row[n0 : n0 + TILE_N] for row in b[k0 : k0 + TILE_K]]
            jobs.append(
                {
                    "n_offset": n0,
                    "k_offset": k0,
                    "graph": graph,
                    "inputs": {"A": a_tile, "B": b_tile},
                }
            )
    return jobs


def accumulate_tile_output(
    logits: list[list[int]],
    tile: list[list[int]],
    n_offset: int,
) -> None:
    if len(tile) != TILE_M or any(len(row) != TILE_N for row in tile):
        raise ValueError("tile output must have shape 8x8")
    for i in range(TILE_M):
        for j in range(TILE_N):
            logits[i][n_offset + j] += int(tile[i][j])


def predict_label(logits: list[list[int]]) -> int:
    if not logits or len(logits[0]) < REAL_CLASSES:
        raise ValueError("classifier logits must include row 0 and 10 class columns")
    class_logits = logits[0][:REAL_CLASSES]
    return max(range(REAL_CLASSES), key=lambda idx: class_logits[idx])


def _pgm_tokens(data: bytes) -> list[str]:
    tokens: list[str] = []
    for raw_line in data.decode("ascii").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            tokens.extend(line.split())
    return tokens


__all__ = [
    "BATCH_ROWS",
    "CLASS_COLUMNS",
    "PIXELS",
    "REAL_CLASSES",
    "activation_batch",
    "activation_batch_from_image",
    "accumulate_tile_output",
    "classifier_graph",
    "classifier_inputs",
    "classifier_inputs_from_image",
    "classifier_weights",
    "flatten_digit",
    "glyph_rows",
    "image_to_glyph_rows",
    "load_pgm_image",
    "lower_classifier_to_rtl_tiles",
    "lower_matmul_to_rtl_tiles",
    "predict_label",
    "reference_logits",
    "reference_logits_from_image",
    "relu",
    "relu_requantize",
    "rtl_tile_graph",
    "tiny_mlp_fc2_weights",
    "tiny_mlp_graph",
    "tiny_mlp_inputs_from_image",
    "tiny_mlp_reference_logits_from_image",
]
