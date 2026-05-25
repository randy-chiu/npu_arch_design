# Architecture Specifications

This directory holds human-reviewable architecture contracts and rationale.
Machine-consumed constants remain in `arch/configs/` and are the source for
generated RTL/software metadata.

Planned active specifications:

```text
npu_core_v0.md
memory_system_v0.md
instruction_set_v0.md
dataflow_v0.md
ppa_model_v0.md
transformer_requirements_v0.md
```

The current PPA and Transformer contracts begin in:

```text
docs/design/ppa_methodology.md
docs/design/transformer_workloads.md
```

They can be promoted or split into specification files as the corresponding
machine-readable configuration fields stabilize.
