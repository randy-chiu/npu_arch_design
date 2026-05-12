# Mandatory Rules For All Module Agents

These rules apply to every human and AI agent working on this NPU project.

## 1. The Architecture Spec Is The Source Of Truth

Canonical Phase 0 spec:

```text
arch/configs/npu_v0.jsonc
```

Do not hard-code architecture facts in compiler, simulator, RTL, runtime, or
tests if they belong in the spec.

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
make rtl-sim
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

See the detailed rulebook at:

```text
docs/work_rules.md
```

