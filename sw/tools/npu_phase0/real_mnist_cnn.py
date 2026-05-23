"""Reference runner for an open-source pretrained MNIST CNN."""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Any

from .digits_classifier import TILE_K, TILE_M, TILE_N, lower_matmul_to_rtl_tiles


MODEL_WEIGHTS_PATH = Path("test/external/mnist_cnn/mnist-cnn.safetensors")
MODEL_README_PATH = Path("test/external/mnist_cnn/README.md")
TEST_IMAGES_PATH = Path("test/external/mnist/t10k-images-idx3-ubyte.gz")
TEST_LABELS_PATH = Path("test/external/mnist/t10k-labels-idx1-ubyte.gz")


def real_mnist_cnn_graph() -> dict[str, Any]:
    return {
        "description": "Open-source pretrained MNIST CNN from cmaeti/mnist-cnn.",
        "source": "https://huggingface.co/cmaeti/mnist-cnn",
        "weights": str(MODEL_WEIGHTS_PATH),
        "parameters": {
            "conv1.weight": {"shape": [32, 1, 3, 3], "dtype": "float32"},
            "conv1.bias": {"shape": [32], "dtype": "float32"},
            "conv2.weight": {"shape": [64, 32, 3, 3], "dtype": "float32"},
            "conv2.bias": {"shape": [64], "dtype": "float32"},
            "fc1.weight": {"shape": [128, 9216], "dtype": "float32"},
            "fc1.bias": {"shape": [128], "dtype": "float32"},
            "fc2.weight": {"shape": [10, 128], "dtype": "float32"},
            "fc2.bias": {"shape": [10], "dtype": "float32"},
        },
        "tensors": {
            "Image": {"shape": [1, 28, 28], "dtype": "float32"},
            "Conv1": {"shape": [32, 26, 26], "dtype": "float32"},
            "Conv1Relu": {"shape": [32, 26, 26], "dtype": "float32"},
            "Conv2": {"shape": [64, 24, 24], "dtype": "float32"},
            "Conv2Relu": {"shape": [64, 24, 24], "dtype": "float32"},
            "Pool": {"shape": [64, 12, 12], "dtype": "float32"},
            "Flat": {"shape": [9216], "dtype": "float32"},
            "Fc1": {"shape": [128], "dtype": "float32"},
            "Fc1Relu": {"shape": [128], "dtype": "float32"},
            "Logits": {"shape": [10], "dtype": "float32"},
            "Predicted": {"shape": [], "dtype": "int64"},
        },
        "ops": [
            {
                "type": "conv2d",
                "x": "Image",
                "weight": "conv1.weight",
                "bias": "conv1.bias",
                "out": "Conv1",
                "input_shape": [1, 28, 28],
                "weight_shape": [32, 1, 3, 3],
                "bias_shape": [32],
                "output_shape": [32, 26, 26],
                "kernel_shape": [3, 3],
                "strides": [1, 1],
                "pads": [0, 0, 0, 0],
            },
            {"type": "relu", "x": "Conv1", "out": "Conv1Relu", "input_shape": [32, 26, 26], "output_shape": [32, 26, 26]},
            {
                "type": "conv2d",
                "x": "Conv1Relu",
                "weight": "conv2.weight",
                "bias": "conv2.bias",
                "out": "Conv2",
                "input_shape": [32, 26, 26],
                "weight_shape": [64, 32, 3, 3],
                "bias_shape": [64],
                "output_shape": [64, 24, 24],
                "kernel_shape": [3, 3],
                "strides": [1, 1],
                "pads": [0, 0, 0, 0],
            },
            {"type": "relu", "x": "Conv2", "out": "Conv2Relu", "input_shape": [64, 24, 24], "output_shape": [64, 24, 24]},
            {
                "type": "maxpool2d",
                "x": "Conv2Relu",
                "kernel": [2, 2],
                "stride": [2, 2],
                "input_shape": [64, 24, 24],
                "output_shape": [64, 12, 12],
                "out": "Pool",
            },
            {"type": "flatten", "x": "Pool", "input_shape": [64, 12, 12], "output_shape": [9216], "out": "Flat"},
            {
                "type": "linear",
                "x": "Flat",
                "weight": "fc1.weight",
                "bias": "fc1.bias",
                "out": "Fc1",
                "input_shape": [9216],
                "weight_shape": [128, 9216],
                "bias_shape": [128],
                "output_shape": [128],
                "in_features": 9216,
                "out_features": 128,
            },
            {"type": "relu", "x": "Fc1", "out": "Fc1Relu", "input_shape": [128], "output_shape": [128]},
            {
                "type": "linear",
                "x": "Fc1Relu",
                "weight": "fc2.weight",
                "bias": "fc2.bias",
                "out": "Logits",
                "input_shape": [128],
                "weight_shape": [10, 128],
                "bias_shape": [10],
                "output_shape": [10],
                "in_features": 128,
                "out_features": 10,
            },
            {"type": "argmax", "x": "Logits", "out": "Predicted", "input_shape": [10], "output_shape": []},
        ],
        "quantization": {
            "phase0_default": {
                "boundary": "selected float tensor and float layer weights are converted to signed int8 before Phase 0 matmul",
                "activation": {"scheme": "symmetric", "granularity": "per-tensor", "dtype": "int8"},
                "weight": {"scheme": "symmetric", "granularity": "per-tensor", "dtype": "int8"},
                "accumulator": {"dtype": "int32"},
                "bias": {"domain": "dequantized-float"},
            }
        },
    }


def real_mnist_cnn_op(name: str) -> dict[str, Any]:
    graph = real_mnist_cnn_graph()
    for op in graph["ops"]:
        if op.get("out") == name:
            return op
    raise ValueError(f"real MNIST CNN graph has no op producing {name}")


def validate_real_mnist_cnn_graph(weights: dict[str, Any] | None = None) -> None:
    graph = real_mnist_cnn_graph()
    tensors = graph["tensors"]
    params = graph["parameters"]

    if weights is not None:
        for name, meta in params.items():
            if name not in weights:
                raise ValueError(f"missing model weight: {name}")
            if list(weights[name].shape) != meta["shape"]:
                raise ValueError(f"{name} shape mismatch: {list(weights[name].shape)} != {meta['shape']}")

    for op in graph["ops"]:
        op_type = op["type"]
        if "input_shape" in op and op["input_shape"] != tensors[op["x"]]["shape"]:
            raise ValueError(f"{op['out']} input shape does not match tensor {op['x']}")
        if "output_shape" in op and op["output_shape"] != tensors[op["out"]]["shape"]:
            raise ValueError(f"{op['out']} output shape does not match tensor table")
        if op_type == "conv2d":
            if op["weight_shape"] != params[op["weight"]]["shape"]:
                raise ValueError(f"{op['out']} weight shape does not match parameter table")
            if op["bias_shape"] != params[op["bias"]]["shape"]:
                raise ValueError(f"{op['out']} bias shape does not match parameter table")
            if _conv2d_output_shape(op["input_shape"], op["weight_shape"], op["strides"], op["pads"]) != op["output_shape"]:
                raise ValueError(f"{op['out']} conv2d output shape is inconsistent")
        elif op_type == "maxpool2d":
            if _maxpool2d_output_shape(op["input_shape"], op["kernel"], op["stride"]) != op["output_shape"]:
                raise ValueError(f"{op['out']} maxpool output shape is inconsistent")
        elif op_type == "flatten":
            if [_num_elements(op["input_shape"])] != op["output_shape"]:
                raise ValueError(f"{op['out']} flatten output shape is inconsistent")
        elif op_type == "linear":
            if op["weight_shape"] != params[op["weight"]]["shape"]:
                raise ValueError(f"{op['out']} weight shape does not match parameter table")
            if op["bias_shape"] != params[op["bias"]]["shape"]:
                raise ValueError(f"{op['out']} bias shape does not match parameter table")
            if op["input_shape"] != [op["in_features"]]:
                raise ValueError(f"{op['out']} linear input shape does not match in_features")
            if op["output_shape"] != [op["out_features"]]:
                raise ValueError(f"{op['out']} linear output shape does not match out_features")
            if op["weight_shape"] != [op["out_features"], op["in_features"]]:
                raise ValueError(f"{op['out']} linear weight shape is inconsistent")
            if op["bias_shape"] != [op["out_features"]]:
                raise ValueError(f"{op['out']} linear bias shape is inconsistent")


def numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def load_safetensors_f32(path: Path = MODEL_WEIGHTS_PATH) -> dict[str, Any]:
    import numpy as np

    data = path.read_bytes()
    header_len = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_len])
    payload_base = 8 + header_len
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            raise ValueError(f"{name} has unsupported dtype {meta['dtype']}")
        start, end = meta["data_offsets"]
        raw = data[payload_base + start : payload_base + end]
        tensors[name] = np.frombuffer(raw, dtype="<f4").reshape(meta["shape"]).copy()
    return tensors


def load_mnist_images(path: Path = TEST_IMAGES_PATH):
    import numpy as np

    data = gzip.decompress(path.read_bytes())
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"{path} is not an IDX image file")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def load_mnist_labels(path: Path = TEST_LABELS_PATH):
    import numpy as np

    data = gzip.decompress(path.read_bytes())
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"{path} is not an IDX label file")
    return np.frombuffer(data, dtype=np.uint8, offset=8).reshape(count)


def forward_intermediates(image, weights: dict[str, Any], normalize: bool = False) -> dict[str, Any]:
    import numpy as np

    x = image.astype(np.float32) / 255.0
    if normalize:
        x = (x - 0.1307) / 0.3081
    x = x.reshape(1, 28, 28)

    x = np.maximum(_conv2d_valid(x, weights["conv1.weight"], weights["conv1.bias"]), 0.0)
    x = np.maximum(_conv2d_valid(x, weights["conv2.weight"], weights["conv2.bias"]), 0.0)
    pool = _maxpool2d_2x2(x)
    flat = pool.reshape(-1)
    fc1 = weights["fc1.weight"] @ flat + weights["fc1.bias"]
    fc1_relu = np.maximum(fc1, 0.0)
    logits = weights["fc2.weight"] @ fc1_relu + weights["fc2.bias"]
    return {
        "pool": pool,
        "flat": flat,
        "fc1": fc1,
        "fc1_relu": fc1_relu,
        "logits": logits,
    }


def forward_logits(image, weights: dict[str, Any], normalize: bool = False):
    return forward_intermediates(image, weights, normalize=normalize)["logits"]


def predict(image, weights: dict[str, Any], normalize: bool = False) -> int:
    import numpy as np

    return int(np.argmax(forward_logits(image, weights, normalize=normalize)))


def fc2_npu_inputs_from_activation(fc1_relu, weights: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    validate_real_mnist_cnn_graph(weights)
    op = real_mnist_cnn_op("Logits")
    in_features = int(op["in_features"])
    out_features = int(op["out_features"])
    class_columns = _round_up(out_features, TILE_N)
    fc1_relu = np.asarray(fc1_relu, dtype=np.float32)
    activation_scale = _symmetric_scale(float(np.max(np.abs(fc1_relu))))
    weight_scale = _symmetric_scale(float(np.max(np.abs(weights["fc2.weight"]))))
    q_activation = np.clip(np.round(fc1_relu * activation_scale), -128, 127).astype(np.int32)
    q_weight = np.clip(np.round(weights["fc2.weight"] * weight_scale), -128, 127).astype(np.int32)

    a = [[0 for _ in range(in_features)] for _ in range(TILE_M)]
    a[0] = [int(value) for value in q_activation.tolist()]
    w = [[0 for _ in range(class_columns)] for _ in range(in_features)]
    for k in range(in_features):
        for cls in range(out_features):
            w[k][cls] = int(q_weight[cls, k])
    return {
        "A": a,
        "W": w,
        "activation_scale": activation_scale,
        "weight_scale": weight_scale,
        "bias": [float(value) for value in weights["fc2.bias"].tolist()],
        "real_columns": out_features,
        "padded_columns": class_columns,
    }


def fc2_quantized_logits_from_int32(acc: list[list[int]], npu_inputs: dict[str, Any]) -> list[float]:
    scale = float(npu_inputs["activation_scale"]) * float(npu_inputs["weight_scale"])
    return [float(acc[0][idx]) / scale + float(npu_inputs["bias"][idx]) for idx in range(npu_inputs["real_columns"])]


def lower_fc2_to_rtl_tiles(npu_inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return lower_matmul_to_rtl_tiles(npu_inputs["A"], npu_inputs["W"])


def fc1_npu_inputs_from_flat(flat, weights: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    validate_real_mnist_cnn_graph(weights)
    op = real_mnist_cnn_op("Fc1")
    in_features = int(op["in_features"])
    out_features = int(op["out_features"])
    padded_out_features = _round_up(out_features, TILE_N)
    flat = np.asarray(flat, dtype=np.float32)
    if len(flat) != in_features:
        raise ValueError(f"fc1 flat activation must have {in_features} elements")

    activation_scale = _symmetric_scale(float(np.max(np.abs(flat))))
    weight_scale = _symmetric_scale(float(np.max(np.abs(weights["fc1.weight"]))))
    q_activation = np.clip(np.round(flat * activation_scale), -128, 127).astype(np.int32)
    q_weight = np.clip(np.round(weights["fc1.weight"] * weight_scale), -128, 127).astype(np.int32)

    a = [[0 for _ in range(in_features)] for _ in range(TILE_M)]
    a[0] = [int(value) for value in q_activation.tolist()]
    w = [[0 for _ in range(padded_out_features)] for _ in range(in_features)]
    for k in range(in_features):
        for out_idx in range(out_features):
            w[k][out_idx] = int(q_weight[out_idx, k])
    return {
        "A": a,
        "W": w,
        "activation_scale": activation_scale,
        "weight_scale": weight_scale,
        "bias": [float(value) for value in weights["fc1.bias"].tolist()],
        "real_columns": out_features,
        "padded_columns": padded_out_features,
    }


def fc1_logical_matmul_graph(npu_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "tensors": {
            "A": {"shape": [len(npu_inputs["A"]), len(npu_inputs["A"][0])], "dtype": "int8"},
            "B": {"shape": [len(npu_inputs["W"]), len(npu_inputs["W"][0])], "dtype": "int8"},
        },
        "ops": [{"type": "matmul", "a": "A", "b": "B", "out": "C"}],
    }


def fc1_relu_from_int32(acc: list[list[int]], npu_inputs: dict[str, Any]) -> list[float]:
    scale = float(npu_inputs["activation_scale"]) * float(npu_inputs["weight_scale"])
    return [
        max(0.0, float(acc[0][idx]) / scale + float(npu_inputs["bias"][idx]))
        for idx in range(npu_inputs["real_columns"])
    ]


def _symmetric_scale(max_abs: float) -> float:
    if max_abs <= 0.0:
        return 1.0
    return 127.0 / max_abs


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _conv2d_output_shape(input_shape: list[int], weight_shape: list[int], strides: list[int], pads: list[int]) -> list[int]:
    in_channels, in_h, in_w = input_shape
    out_channels, weight_channels, kernel_h, kernel_w = weight_shape
    if in_channels != weight_channels:
        raise ValueError("conv2d channel mismatch")
    pad_top, pad_left, pad_bottom, pad_right = pads
    out_h = (in_h + pad_top + pad_bottom - kernel_h) // strides[0] + 1
    out_w = (in_w + pad_left + pad_right - kernel_w) // strides[1] + 1
    return [out_channels, out_h, out_w]


def _maxpool2d_output_shape(input_shape: list[int], kernel: list[int], stride: list[int]) -> list[int]:
    channels, height, width = input_shape
    return [channels, (height - kernel[0]) // stride[0] + 1, (width - kernel[1]) // stride[1] + 1]


def _num_elements(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _conv2d_valid(x, weight, bias):
    import numpy as np

    channels, height, width = x.shape
    out_channels, weight_channels, kernel_h, kernel_w = weight.shape
    if channels != weight_channels:
        raise ValueError(f"conv channel mismatch: x={channels}, weight={weight_channels}")
    out = np.empty((out_channels, height - kernel_h + 1, width - kernel_w + 1), dtype=np.float32)
    for oy in range(out.shape[1]):
        for ox in range(out.shape[2]):
            patch = x[:, oy : oy + kernel_h, ox : ox + kernel_w]
            out[:, oy, ox] = (weight * patch).sum(axis=(1, 2, 3)) + bias
    return out


def _maxpool2d_2x2(x):
    channels, height, width = x.shape
    if height % 2 or width % 2:
        raise ValueError("2x2 maxpool requires even spatial dimensions")
    return x.reshape(channels, height // 2, 2, width // 2, 2).max(axis=(2, 4))


__all__ = [
    "MODEL_README_PATH",
    "MODEL_WEIGHTS_PATH",
    "TEST_IMAGES_PATH",
    "TEST_LABELS_PATH",
    "fc1_logical_matmul_graph",
    "fc1_npu_inputs_from_flat",
    "fc1_relu_from_int32",
    "fc2_npu_inputs_from_activation",
    "fc2_quantized_logits_from_int32",
    "forward_logits",
    "forward_intermediates",
    "load_mnist_images",
    "load_mnist_labels",
    "load_safetensors_f32",
    "lower_fc2_to_rtl_tiles",
    "numpy_available",
    "predict",
    "real_mnist_cnn_graph",
    "real_mnist_cnn_op",
    "validate_real_mnist_cnn_graph",
]
