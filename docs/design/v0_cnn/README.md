# V0/CNN Design Docs

This directory contains V0/CNN-specific design notes. They remain useful as
regression and historical implementation contracts, but they are not the main
driver for Transformer-oriented NPU v1 architecture choices.

| Document | Scope |
| --- | --- |
| `fc1_k_streaming_matmul.md` | Real MNIST CNN `fc1` K-axis streaming matmul contract |
| `k_stream_ping_pong_buffer.md` | K-streaming ping-pong overlap design and measured bottleneck |
| `quantization_strategy.md` | Selected-layer quantization boundary used by current CNN regression |
