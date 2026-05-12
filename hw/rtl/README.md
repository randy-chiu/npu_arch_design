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
- 8-element INT8 softmax through explicit vector micro-ops.
- Micro-op sequencer for the Phase 0 ISA subset.
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
| `0x400` - `0x40f` | Instruction memory, one 32-bit micro-op per word |

## Micro-Op Encoding

The RTL smoke test uses a compact temporary encoding:

```text
[31:28] opcode
[27:24] arg0
[23:20] arg1
[19:0]  reserved
```

Opcodes:

| Opcode | Instruction |
| --- | --- |
| `0x1` | `LOAD` |
| `0x2` | `STORE` |
| `0x3` | `MATMUL` |
| `0x4` | `VREDMAX` |
| `0x5` | `VSUB` |
| `0x6` | `VEXP` |
| `0x7` | `VREDSUM` |
| `0x8` | `VDIV` |
| `0xf` | `HALT` |

Tensor IDs:

| ID | Tensor |
| --- | --- |
| `0x0` | `A` |
| `0x1` | `B` |
| `0x2` | `C` |
| `0x3` | `X` |
| `0x4` | `Y` |

Buffer IDs:

| ID | Buffer |
| --- | --- |
| `0x0` | `spad_a` |
| `0x1` | `spad_b` |
| `0x2` | `acc` |
| `0x3` | `vec` |

Control:

| Signal | Meaning |
| --- | --- |
| `op` | Reserved in Phase 0; execution is driven by instruction memory |
| `start=1` | Launch operation |
| `done=1` | Operation complete |

## Softmax Approximation

The Phase 0 RTL softmax sequence computes:

1. max over 8 signed INT8 inputs.
2. subtract max.
3. approximate exp using a tiny LUT for deltas `0` through `-8`.
4. normalize to Q0.8 using integer division.

This is intentionally small and FPGA-friendly. The software simulator still
uses fp32 softmax; Phase 1 should add a shared fixed-point softmax model so RTL
and simulator use identical numeric semantics.
