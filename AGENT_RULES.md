# Mandatory Rules For All Module Agents

These rules apply to every human and AI agent working on this NPU project.

## 1. The Architecture Spec Is The Source Of Truth

Canonical Phase 0 spec:

```text
arch/configs/npu_v0.jsonc
```

Canonical SoC spec:

```text
arch/configs/soc_v0.jsonc
```

Canonical NPU wrapper register spec:

```text
arch/configs/npu_wrapper_v0.jsonc
```

Do not hard-code architecture facts, SoC address-map facts, or NPU-wrapper
register/window offsets in compiler, simulator, RTL, runtime, firmware, or
tests if they belong in the relevant spec.

Each architecture fact must have exactly one canonical representation. Do not
duplicate opcode maps, instruction field layouts, tensor IDs, buffer IDs,
memory-map constants, fixture paths, verification lengths, or numerical
tolerances in multiple source files. Generate downstream constants or metadata
from the canonical source instead.

## 2. ISA Changes Require Full-System Updates

If ISA opcodes, fields, semantics, encoding, or legality rules change, the same
change must update all affected components:

- spec validator
- compiler emission
- functional simulator
- cycle model when available
- RTL control/decode/datapath
- runtime program metadata
- golden or tolerance rules
- tests and documentation

An ISA change is not accepted until the closed-loop verification flow passes.

## 3. Hardware Spec Changes Require Re-Verification

Every spec change must be followed by:

```text
make validate-arch
make demo
make test
```

If RTL-visible behavior changes, also run the RTL simulation target on a machine
with Icarus Verilog or an equivalent SystemVerilog simulator:

```text
make npu-core-sim
```

If SoC-visible launch, register, bus, or wrapper behavior changes, also run:

```text
make soc-sim
make cpu-soc-sim
```

## 4. Keep Phase 0 Minimal

Phase 0 only supports:

- `matmul`
- `softmax`
- `matmul -> softmax`

Allowed micro-ops:

- `LOAD`
- `STORE`
- `MATMUL`
- `VREDMAX`
- `VSUB`
- `VEXP`
- `VREDSUM`
- `VDIV`
- `HALT`

Do not add unrelated operators or advanced architecture features until this
minimal loop is stable.

## 5. Research Does Not Directly Change The Architecture

Reference research lives under:

```text
references/
```

New ideas become implementation only through a reviewed spec change and passing
verification.

## 6. Record Collaboration Decisions

Every meaningful human/AI collaboration turn must update:

```text
docs/collaboration_journal.md
```

Record goals, decisions, architectural reasoning, validation outcomes, and
open risks. Do not record secrets, passwords, tokens, or noisy patch-level edit
history.

See the detailed rulebook at:

```text
docs/work_rules.md
```
