# Phase 0 RTL

This directory contains a hand-written minimal RTL implementation for
`npu_v0`. It is not yet a generated RTL system. It exists so the architecture
project has a real hardware target early.

## Implemented Hardware

Top module:

```text
npu_v0_top
```

Capabilities:

- 8x8 INT8 matrix multiply.
- 8-element INT8 softmax approximation.
- Simple host write/read interface.
- Start/done control.

## Host Memory Map

All addresses are word addresses on a 32-bit host data bus.

| Address range | Purpose |
| --- | --- |
| `0x000` - `0x03f` | Matrix A, one signed INT8 per word |
| `0x100` - `0x13f` | Matrix B, one signed INT8 per word |
| `0x200` - `0x23f` | Matrix C, one signed INT32 per word |
| `0x300` - `0x307` | Softmax input X, one signed INT8 per word |
| `0x380` - `0x387` | Softmax output Y, Q0.8 unsigned probability per word |

Control:

| Signal | Meaning |
| --- | --- |
| `op=0` | Run matmul |
| `op=1` | Run softmax |
| `start=1` | Launch operation |
| `done=1` | Operation complete |

## Softmax Approximation

The Phase 0 RTL softmax block computes:

1. max over 8 signed INT8 inputs.
2. subtract max.
3. approximate exp using a tiny LUT for deltas `0` through `-8`.
4. normalize to Q0.8 using integer division.

This is intentionally small and FPGA-friendly. The software simulator still
uses fp32 softmax; Phase 1 should add a shared fixed-point softmax model so RTL
and simulator use identical numeric semantics.

