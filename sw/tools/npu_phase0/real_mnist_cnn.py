"""Reference runner for an open-source pretrained MNIST CNN."""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Any

from .digits_classifier import BATCH_ROWS, CLASS_COLUMNS, lower_matmul_to_rtl_tiles


MODEL_WEIGHTS_PATH = Path("test/external/mnist_cnn/mnist-cnn.safetensors")
MODEL_README_PATH = Path("test/external/mnist_cnn/README.md")
TEST_IMAGES_PATH = Path("test/external/mnist/t10k-images-idx3-ubyte.gz")
TEST_LABELS_PATH = Path("test/external/mnist/t10k-labels-idx1-ubyte.gz")


def real_mnist_cnn_graph() -> dict[str, Any]:
    return {
        "description": "Open-source pretrained MNIST CNN from cmaeti/mnist-cnn.",
        "source": "https://huggingface.co/cmaeti/mnist-cnn",
        "weights": str(MODEL_WEIGHTS_PATH),
        "tensors": {
            "Image": {"shape": [1, 28, 28], "dtype": "float32"},
            "Conv1": {"shape": [32, 26, 26], "dtype": "float32"},
            "Conv2": {"shape": [64, 24, 24], "dtype": "float32"},
            "Pool": {"shape": [64, 12, 12], "dtype": "float32"},
            "Fc1": {"shape": [128], "dtype": "float32"},
            "Logits": {"shape": [10], "dtype": "float32"},
        },
        "ops": [
            {"type": "conv2d", "x": "Image", "weight": "conv1.weight", "bias": "conv1.bias", "out": "Conv1"},
            {"type": "relu", "x": "Conv1", "out": "Conv1Relu"},
            {"type": "conv2d", "x": "Conv1Relu", "weight": "conv2.weight", "bias": "conv2.bias", "out": "Conv2"},
            {"type": "relu", "x": "Conv2", "out": "Conv2Relu"},
            {"type": "maxpool2d", "x": "Conv2Relu", "kernel": [2, 2], "stride": [2, 2], "out": "Pool"},
            {"type": "flatten", "x": "Pool", "out": "Flat"},
            {"type": "linear", "x": "Flat", "weight": "fc1.weight", "bias": "fc1.bias", "out": "Fc1"},
            {"type": "relu", "x": "Fc1", "out": "Fc1Relu"},
            {"type": "linear", "x": "Fc1Relu", "weight": "fc2.weight", "bias": "fc2.bias", "out": "Logits"},
            {"type": "argmax", "x": "Logits", "out": "Predicted"},
        ],
    }


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

    activation_scale = _symmetric_scale(float(np.max(np.abs(fc1_relu))))
    weight_scale = _symmetric_scale(float(np.max(np.abs(weights["fc2.weight"]))))
    q_activation = np.clip(np.round(fc1_relu * activation_scale), -128, 127).astype(np.int32)
    q_weight = np.clip(np.round(weights["fc2.weight"] * weight_scale), -128, 127).astype(np.int32)

    a = [[0 for _ in range(128)] for _ in range(BATCH_ROWS)]
    a[0] = [int(value) for value in q_activation.tolist()]
    w = [[0 for _ in range(CLASS_COLUMNS)] for _ in range(128)]
    for k in range(128):
        for cls in range(10):
            w[k][cls] = int(q_weight[cls, k])
    return {
        "A": a,
        "W": w,
        "activation_scale": activation_scale,
        "weight_scale": weight_scale,
        "bias": [float(value) for value in weights["fc2.bias"].tolist()],
    }


def fc2_quantized_logits_from_int32(acc: list[list[int]], npu_inputs: dict[str, Any]) -> list[float]:
    scale = float(npu_inputs["activation_scale"]) * float(npu_inputs["weight_scale"])
    return [float(acc[0][idx]) / scale + float(npu_inputs["bias"][idx]) for idx in range(10)]


def lower_fc2_to_rtl_tiles(npu_inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return lower_matmul_to_rtl_tiles(npu_inputs["A"], npu_inputs["W"])


def _symmetric_scale(max_abs: float) -> float:
    if max_abs <= 0.0:
        return 1.0
    return 127.0 / max_abs


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
]
