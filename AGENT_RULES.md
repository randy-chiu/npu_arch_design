# Mandatory Project Rules

These rules apply to every human and AI agent working on this NPU project.

## 1. Specs Are The Source Of Truth

Keep architecture facts in the relevant canonical spec and generate downstream
constants from those specs whenever practical.

Canonical specs:

```text
arch/configs/npu_v0.jsonc
arch/configs/npu_transformer_v1.jsonc
arch/configs/soc_v0.jsonc
arch/configs/npu_wrapper_v0.jsonc
```

Do not duplicate opcode maps, descriptor fields, register offsets, tensor IDs,
fixed-point contracts, or verification tolerances across RTL, firmware, tools,
tests, and docs.

## 2. Reuse Existing RTL Before Adding New Modules

For new requirements, first check whether an existing RTL module, datapath,
primitive, scheduler path, or runtime descriptor should be extended. Prefer
evolving shared blocks such as matrix, vector, reduction, SFU, accumulator,
wrapper, and firmware/runtime paths over creating temporary one-off modules.

Temporary modules are allowed only when the existing boundary cannot reasonably
represent the requirement. In that case, document the temporary scope and the
plan to merge, replace, or retire it.

## 3. Design Documentation Comes Before Module Coding

Before coding or materially changing any RTL module, write or update the
module's detailed design document. The document must cover:

- the mathematical or architectural requirement;
- why the existing module is reused or changed;
- interface, data type, fixed-point, mode, rounding, and saturation behavior;
- golden/RTL consistency rules;
- verification tests and PPA reporting impact;
- known limitations and follow-up conditions.

## 4. Cross-Layer Changes Must Be Complete

Any ISA, descriptor, register, datatype, memory-layout, numerical-contract, or
RTL-visible behavior change must update all affected layers:

- architecture/config specs;
- RTL datapath/control;
- firmware/runtime descriptors;
- fixture and workload generators;
- golden models and tolerances;
- perf/PPA reporting;
- tests and design docs.

Partial cross-layer changes are not accepted.

## 5. Verification Must Match The Blast Radius

Run the smallest relevant tests while iterating, then run the full affected
flow before handing work back.

Common gates:

```text
make npu-core-sim
make primitive-engines-sim
make ppa-l0-report WORKLOAD_PROFILE=transformer
make test
```

If SoC-visible launch, register, bus, wrapper, descriptor, or firmware behavior
changes, run a CPU-to-SoC RTL path such as:

```text
make cpu-soc-transformer
```

Report any skipped or unavailable verification clearly.

## 6. Record Important Decisions

For meaningful architecture or collaboration decisions, update:

```text
docs/collaboration_journal.md
```

Record goals, decisions, reasoning, validation outcomes, and open risks. Do not
record secrets or noisy patch-level history.
