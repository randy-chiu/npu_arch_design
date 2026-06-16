# Repository Agent Contract

This file is the mandatory entry point for every AI-assisted task in this
repository. Detailed contracts remain in the linked documents; do not duplicate
or redefine them here.

## Required Startup

Before planning or editing:

1. Read this file.
2. Inspect `git status` and preserve unrelated user changes.
3. Read `docs/design/transformer/next_steps.md` for the active package.
4. Read the owning design document and architecture spec for the affected area.
5. For architecture or PPA work, read `docs/work_rules.md` and
   `docs/design/ppa_methodology.md`.

## North Star

The project explores an NPU architecture for representative LLM inference PPA.
The objective is not operator count or a large software stack. Software exists
to generate representative schedules, submit hardware experiments, verify
results, and expose trustworthy PPA.

Every architecture iteration must follow:

```text
representative workload and measured bottleneck
  -> spec-first design and explicit expected tradeoff
  -> implementation
  -> functional and compatibility regression
  -> candidate-versus-retained-baseline PPA evidence
  -> accept, revise, or reject
```

RTL functionality alone is not an accepted architecture optimization.

## Mandatory Rules

- Design before code. Explain the problem, why the current mechanism is
  inadequate, the proposed mechanism, ownership, expected PPA benefit/cost,
  and acceptance tests in the owning design document. Obtain user review
  before coding a new architecture mechanism.
- Spec first. Hardware-visible architecture, ISA, descriptor, interface,
  resource, or latency changes update canonical `arch/` specs/configs before
  RTL and downstream consumers.
- Hardware first. Do not build general graph importers, dynamic runtimes,
  allocators, or broad software frameworks unless a measured hardware
  experiment requires them.
- Representative functionality before local optimization. Complete and
  measure the currently selected end-to-end workload baseline before accepting
  an isolated operator optimization, unless that optimization is required to
  make the baseline executable.
- Do not hide executable-workload stages in CPU or fixture preprocessing.
  Model-only gaps must stay explicit until the NPU RTL executes them.
- Tiling belongs to the Compiler/planner. Runtime is a thin address binder and
  submitter; Wrapper does not tile, fuse, or parse graphs.
- Keep one fact in one owning document. Update existing module/design documents
  instead of creating overlapping design files.
- Preserve comparable baselines and report unfavorable regressions.
- Use parallel PPA evidence views, not upgrade levels:
  `rtl_workload_view`, `mapped_area_timing_view`, `activity_power_view`, and
  `physical_implementation_view`.
- Never combine evidence from different architecture variants, RTL/config
  revisions, synthesis tops, or incompatible workload manifests.

## Completion Gate

Before claiming completion:

1. Run `make check-workflow`.
2. Run focused tests and the relevant end-to-end regression.
3. Run `make test` for non-trivial cross-module changes.
4. For performance-affecting work, regenerate the relevant PPA report and
   record theoretical versus measured behavior, baseline delta, and costs.
5. Update the owning design document and `next_steps.md` with actual status,
   evidence, remaining gaps, and the next hardware decision.

If a required check cannot run, state that explicitly; do not silently mark the
work complete.

## Detailed Contracts

- `docs/work_rules.md`: source-of-truth, design-before-code, verification, and
  performance iteration rules.
- `docs/design/transformer/next_steps.md`: active architecture plan and
  hardware-first scope.
- `docs/design/transformer/software_runtime_compiler_attention.md`:
  Compiler/submitter/hardware ownership.
- `docs/design/ppa_methodology.md`: PPA evidence and comparison rules.
- `docs/design/verification_strategy.md`: verification layers and gates.
