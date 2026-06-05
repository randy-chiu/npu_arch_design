# Verification Strategy

[TOC]

This document describes the current verification layers and what each layer is
expected to catch.

## 1. Verification Principle

Every architecture step should preserve a runnable end-to-end loop. We prefer
small verified increments over large untested rewrites.

For most changes, the minimum gate is:

```text
make test
make ppa-l0-report
```

`make ppa-l0-report` matters because many recent changes are performance-structure
changes. Functional PASS alone is not enough if timeline/counter behavior
silently regresses.

## 2. Verification Layers

| Layer | Command/Test | Purpose |
| --- | --- | --- |
| Architecture spec | `make validate-arch` | validate NPU config structure |
| Python golden | unit tests in `test/rtl/test_phase0.py` | verify matmul/softmax expected math |
| Compiler/simulator | `compile_graph` + `MicroOpFunctionalSimulator` tests | verify graph lowering and functional execution |
| RTL fixture generation | `make rtl-fixtures` | produce RTL hex/includes from tooling |
| NPU core RTL | `make npu-core-sim` | standalone NPU core fixture test |
| Wrapper/SoC legacy path | `make soc-sim` | direct wrapper-window path |
| CPU firmware SoC quick | `make cpu-soc-quick` or `make cpu-soc-sim` | PicoRV32 firmware launches quick descriptor jobs |
| CPU firmware SoC full CNN | `make cpu-soc-cnn-full` | Explicit long real-MNIST CNN full-`fc1` regression |
| CPU firmware SoC all | `make cpu-soc-all` | Full CNN plus Transformer workload profile |
| Perf report | `make perf-report` | CPU SoC simulation plus JSON/HTML performance report for `WORKLOAD_PROFILE` |
| NPU subsystem elaboration | `make npu-subsystem-elab` | check the primary PPA RTL boundary compiles without simulation SoC memories |
| PPA contract | `test/ppa_contract/test_ppa_contract.py` | check PPA target/schema and Transformer manifest contracts |
| PPA L0 report | `make ppa-l0-report` | produce Level 0 measured-performance and normalized area/energy estimate output |
| Fast PPA L0 derivative | `make ppa-l0-from-perf` | regenerate and validate L0 output from an existing perf artifact; not a substitute for a fresh simulation gate |
| Full unit gate | `make test` | Python tests and available RTL sims, using quick firmware profile by default |

Compatibility aliases remain available:

```text
make ppa-model-report
make ppa-model-from-perf
make validate-ppa-model
```

They call the new `ppa-l0-*` targets. New documentation should prefer the
`ppa-l0-*` names because `model` can be mistaken for a software or network
agent rather than a Level 0 estimate.

Workload profiles:

| Profile | Target examples | Contents |
| --- | --- | --- |
| `quick` | `make cpu-soc-quick`, default `make test` | operator smoke, digits classifier, tiny Transformer micro workloads |
| `transformer` | `make cpu-soc-transformer` | same executable coverage as quick today, reserved for Transformer-focused growth |
| `cnn-full` | `make cpu-soc-cnn-full` | operator smoke, digits classifier, real MNIST CNN full `fc1`/`fc2`; no Transformer micro jobs |
| `all` | `make cpu-soc-all`, `make test-full` | real MNIST CNN full path plus Transformer micro jobs |

## 3. Golden And Simulator Tests

Golden tests cover:

- basic matmul correctness;
- graph-level matmul/softmax expected output;
- compiler output matching compatibility path;
- assembler encoding matching historical fixture path;
- simulator counters such as `mac_ops` and `dma_transfers`.

These tests catch software/tooling regressions before RTL simulation.

## 4. NPU Core RTL Test

`make npu-core-sim` builds:

```text
hw/npu_core/rtl/matrix/matmul_array.sv
hw/npu_core/rtl/npu_v0_top.sv
hw/npu_core/tb/npu_v0_tb.sv
```

It uses generated fixture files under `build/rtl_fixture`.

This layer checks that the core can execute generated programs and match
expected matmul/softmax outputs without the CPU wrapper path.

## 5. SoC Legacy Wrapper Test

`make soc-sim` checks the older direct-window wrapper control path. It is still
useful because legacy windows remain implemented and can expose register-window
breakage.

This path should not be treated as the main firmware flow.

## 6. CPU-Controlled SoC Test

`make cpu-soc-sim` is the main functional system test. It uses the quick
workload profile by default; use `make cpu-soc-cnn-full` or `make cpu-soc-all`
when the long full-CNN path is required.

It verifies:

- firmware boots on PicoRV32;
- generated headers and linker script are usable;
- firmware stages tensors/programs/descriptors in SRAM;
- CPU launches wrapper through MMIO;
- wrapper fetches through SRAM descriptor path;
- NPU core computes matmul and softmax;
- wrapper writes output to SRAM;
- firmware validates outputs;
- each descriptor job's first-batch wrapper perf CSR snapshot matches the
  existing testbench-sampled reference counters;
- `test_status` reports PASS.

This is the most important functional closed loop.

## 7. Perf Report Test

`make perf-report` runs the CPU SoC simulation and captures `PERF_JOB` lines
serialized from the CSR snapshot values firmware reads through MMIO.

It verifies:

- simulation still passes;
- performance JSON can be generated;
- HTML report can be generated;
- cycle baselines can be inspected;
- CSR summary/provenance remains usable after architecture changes; legacy
  phase-rich log replay remains parser-compatible.

`test/rtl/test_perf_report.py` also unit-tests parser/report generation with a
small synthetic log.

## 8. PPA Contract And Subsystem Boundary

The first PPA-framework gate does not claim synthesized area, timing, or power
values. It verifies that the primary PPA boundary and its Level 0 model input
contracts are structurally usable:

```text
make npu-subsystem-elab
PYTHONPATH=sw/tools python -m unittest test.ppa_contract.test_ppa_contract -v
```

Current checks:

- `npu_subsystem_top` elaborates from the existing wrapper/data-mover/core RTL
  while keeping CPU, boot ROM, and staging SRAM outside the top;
- `arch/configs/ppa/sky130hd_v0.jsonc` declares `npu_subsystem_top` as the
  primary public-ASIC estimation boundary;
- `ppa/schema/ppa_result.schema.json` preserves top/workload/target identity
  and optional future area/timing/power/energy sections;
- the initial Transformer micro workload manifest distinguishes prefill and
  decode requirements.

The Level 0 report combines RTL-measured cycle/traffic values with normalized
area/energy proxies. Mapped-area/timing, activity-driven power, and physical
ASIC gates will be added only after the corresponding analysis level is useful
and executable.

Current Level 0 entry point:

```text
make ppa-l0-report
```

For schema/report-only iteration after a successful perf run:

```text
make ppa-l0-from-perf
```

Generated outputs:

```text
build/ppa/ppa.json
build/ppa/ppa_overview.html
```

## 9. Current Baselines

After A1 matmul array, A2 structural data mover, 4-lane core host interface,
`WORDS_PER_CYCLE=4` NPU-side SRAM/data-mover/core-host movement, SoC DMA
staging, explicit data mover counters, K-streaming A/B ping-pong overlap,
grayscale digit fixtures, real MNIST CNN `fc2` SoC smoke, the first real `fc1`
SoC tile smoke, and full `fc1` 16-output-N-tile K-stream SoC coverage:

```text
make test        PASS, 46 tests (including manifest and strengthened PPA contract checks)
make perf-report PASS
matmul total cycles: 82
matmul core matmul cycles: 10
softmax total cycles: 31
digits_linear_classifier: 16 jobs, 1312 cycles
real_mnist_cnn_fc1_tile0: 1 job, 82 cycles
real_mnist_cnn_fc1_k_stream_smoke: 1 job, 186 cycles
real_mnist_cnn_fc1_full_k_stream_layer: 16 jobs, 627488 cycles
real_mnist_cnn_fc2: 32 jobs, 2624 cycles
perf summary: 68 jobs, 7 workloads, 631805 total cycles
```

Current Level 0 model output, based on the same RTL performance report:

```text
npu_subsystem structural area model: 6998.4 normalized_area_units
npu_subsystem local-state storage:    7968 bits
operator_smoke_matmul energy model:   1428.5 normalized_energy_units
real_mnist_cnn_fc1_full_k_stream_layer:
  measured cycles:                    627488
  measured data_mover.words:          2360576
  derived int8 MAC operations:        9437184
  event-energy model:                 19037384.0 normalized_energy_units
```

For the ping-pong comparison, RTL counters show `313056` saved cycles with
MAC work and moved words unchanged. The Level 0 model attributes only
`78264.0 normalized_energy_units` of modeled reduction to the shorter active
duration; it does not claim measured power or external-memory energy savings.

The current named baseline comparison in the full-CNN Level 0 report is:

```text
baseline:  npu_v0_a2_serial_k_stream
candidate: npu_v0_a2_ping_pong
workload:  real_mnist_cnn_fc1_full_k_stream_layer

measured cycles:       940544 -> 627488, -313056 (-33.285%), improvement
measured mover words:  2360576 -> 2360576, invariant
derived MAC work:      9437184 -> 9437184, invariant
energy model:          19115648.0 -> 19037384.0, -78264.0 (-0.409%), improvement
area model:            6947.2 -> 6998.4, +51.2 (+0.737%), cost
```

This is the intended report shape for future NPU iterations: improvements and
resource/energy costs remain visible together.

The wrapper-visible perf snapshot CSR is now the report/PPA performance
provenance. The descriptor-carried generated `job_id` adds one descriptor-read
cycle per job, yielding `68 jobs / 7 workloads / 631805 total cycles`;
`soc_cpu_tb` additionally checks each CSR snapshot against its validation-only
event reference.

Current explicit data mover counters for one full `fc1` K-stream output N tile:

```text
data_mover.transfer_cycles: 36884
data_mover.words: 147536
data_mover.read_words: 147472
data_mover.write_words: 64
core.matmul cycles: 11520
```

The full FC1 single-N-tile total cycle count dropped from 58784 to 39218 after
K-streaming A/B ping-pong overlap. The data mover words and core matmul cycles
stay stable, which is the intended proof that the change overlaps movement with
compute rather than skipping work.

The full `fc1` K-stream smoke verifies one complete output N tile:

```text
A[8,9216] * B[9216,8] -> C[8,8]
k_chunks = 1152
```

完整 `fc1` K-stream smoke 验证的是一个完整 output N tile，而不是完整 128 输出通道的
`fc1` layer。当前 CPU-controlled SoC smoke 已经进一步运行 16 个 output N-tile
K-stream jobs，覆盖完整 quantized `fc1` matmul layer；bias/ReLU 仍是下一步。

When a change intentionally modifies timing, update docs and explain whether
the change is functional RTL behavior or report/model accounting.

## 10. What To Add Next

Near-term verification gaps:

- add comparable baselines for earlier array and narrower-mover architecture
  changes, using explicit recorded evidence or rerunnable variants;
- later run lightweight mapped-area/timing extraction for `npu_core` and
  `npu_subsystem`;
- add activity capture and power/energy result validation after the power flow
  is executable;
- ping-pong overlap regression assertion is covered in
  `test/rtl/test_perf_report.py`: the full `fc1` synthetic PERF_JOB remains
  below the old 58784-cycle serial baseline while `data_mover.words` and
  `core.matmul` stay stable;
- unit test for `npu_v0_data_mover` edge cases: zero words, one word, multiple
  words, store direction;
- perf regression assertions for expected lane names and key counters;
- timeout/error tests for wrapper stuck-core behavior once timeout exists;
- descriptor validation/error tests once wrapper exposes errors;
- larger transfer tests after counter widths are expanded.

First-batch perf CSR coverage now present:

- `soc_tb` reads `PERF_STATUS`/`PERF_TOTAL_CYCLES` through MMIO after a
  completed legacy job and checks idle clear behavior;
- `soc_cpu_tb` correlates each descriptor job's completed snapshot for total,
  core, data-mover and SRAM-boundary counters against validation-only TB
  reference samples;
- firmware reads snapshot identity and summary fields through MMIO, and
  `PERF_JOB`/Level 0 PPA output consume those read values with
  `measured_architectural_perf_csr_snapshot` provenance.

## 11. Test Update Rule

When changing an interface or timing model:

1. Update source-of-truth config if applicable.
2. Update generated expectations or fixtures.
3. Update RTL.
4. Update software/firmware if ABI changed.
5. Update docs.
6. Run `make test`.
7. Run `make perf-report` if timing or report output changed.
