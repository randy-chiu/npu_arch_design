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
make perf-report
```

`make perf-report` matters because many recent changes are performance-structure
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
| CPU firmware SoC | `make cpu-soc-sim` | PicoRV32 firmware launches descriptor jobs |
| Perf report | `make perf-report` | CPU SoC simulation plus JSON/HTML performance report |
| NPU subsystem elaboration | `make npu-subsystem-elab` | check the primary PPA RTL boundary compiles without simulation SoC memories |
| PPA contract | `test/ppa_contract/test_ppa_contract.py` | check PPA target/schema and Transformer manifest contracts |
| PPA proxy report | `make ppa-proxy-report` | produce Level 0 measured-performance and normalized area/energy proxy output |
| Fast PPA derivative | `make ppa-proxy-from-perf` | regenerate and validate proxy output from an existing perf artifact; not a substitute for the full gate |
| Full unit gate | `make test` | Python tests and available RTL sims |

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
hw/npu_core/rtl/matmul_array.sv
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

`make cpu-soc-sim` is the main functional system test.

It verifies:

- firmware boots on PicoRV32;
- generated headers and linker script are usable;
- firmware stages tensors/programs/descriptors in SRAM;
- CPU launches wrapper through MMIO;
- wrapper fetches through SRAM descriptor path;
- NPU core computes matmul and softmax;
- wrapper writes output to SRAM;
- firmware validates outputs;
- `test_status` reports PASS.

This is the most important functional closed loop.

## 7. Perf Report Test

`make perf-report` runs the CPU SoC simulation and captures `PERF_JOB` lines.

It verifies:

- simulation still passes;
- performance JSON can be generated;
- HTML report can be generated;
- cycle baselines can be inspected;
- UI lanes remain useful after architecture changes.

`test/rtl/test_perf_report.py` also unit-tests parser/report generation with a
small synthetic log.

## 8. PPA Contract And Subsystem Boundary

The first PPA-framework gate does not claim synthesized area, timing, or power
values. It verifies that the primary PPA boundary and its Level 0 proxy input
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
make ppa-proxy-report
```

For schema/report-only iteration after a successful perf run:

```text
make ppa-proxy-from-perf
```

Generated outputs:

```text
build/ppa/proxy/ppa_proxy.json
build/ppa/proxy/ppa_proxy_report.html
```

## 9. Current Baselines

After A1 matmul array, A2 structural data mover, 4-lane core host interface,
`WORDS_PER_CYCLE=4` NPU-side SRAM/data-mover/core-host movement, SoC DMA
staging, explicit data mover counters, K-streaming A/B ping-pong overlap,
grayscale digit fixtures, real MNIST CNN `fc2` SoC smoke, the first real `fc1`
SoC tile smoke, and full `fc1` 16-output-N-tile K-stream SoC coverage:

```text
make test        PASS, 43 tests (including manifest and strengthened PPA contract checks)
make perf-report PASS
matmul total cycles: 81
matmul core matmul cycles: 10
softmax total cycles: 30
digits_linear_classifier: 16 jobs, 1296 cycles
real_mnist_cnn_fc1_tile0: 1 job, 81 cycles
real_mnist_cnn_fc1_k_stream_smoke: 1 job, 185 cycles
real_mnist_cnn_fc1_full_k_stream_layer: 16 jobs, 627472 cycles
real_mnist_cnn_fc2: 32 jobs, 2592 cycles
perf summary: 68 jobs, 7 workloads, 631737 total cycles
```

Current Level 0 proxy output, based on the same RTL performance report:

```text
npu_subsystem structural area proxy: 6998.4 normalized_area_units
npu_subsystem local-state storage:    7968 bits
operator_smoke_matmul energy proxy:   1428.25 normalized_energy_units
real_mnist_cnn_fc1_full_k_stream_layer:
  measured cycles:                    627472
  measured data_mover.words:          2360576
  derived int8 MAC operations:        9437184
  event-energy proxy:                 19037380.0 normalized_energy_units
```

For the ping-pong comparison, RTL counters show `313072` saved cycles with
MAC work and moved words unchanged. The Level 0 model attributes only
`78268.0 normalized_energy_units` of modeled reduction to the shorter active
duration; it does not claim measured power or external-memory energy savings.

The current named baseline comparison in `make ppa-proxy-report` is:

```text
baseline:  npu_v0_a2_serial_k_stream
candidate: npu_v0_a2_ping_pong
workload:  real_mnist_cnn_fc1_full_k_stream_layer

measured cycles:       940544 -> 627472, -313072 (-33.286%), improvement
measured mover words:  2360576 -> 2360576, invariant
derived MAC work:      9437184 -> 9437184, invariant
energy proxy:          19115648.0 -> 19037380.0, -78268.0 (-0.409%), improvement
area proxy:            6947.2 -> 6998.4, +51.2 (+0.737%), cost
```

This is the intended report shape for future NPU iterations: improvements and
resource/energy costs remain visible together.

Current explicit data mover counters for one full `fc1` K-stream output N tile:

```text
data_mover.transfer_cycles: 36884
data_mover.words: 147536
data_mover.read_words: 147472
data_mover.write_words: 64
core.matmul cycles: 11520
```

The full FC1 single-N-tile total cycle count dropped from 58784 to 39217 after
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

## 11. Test Update Rule

When changing an interface or timing model:

1. Update source-of-truth config if applicable.
2. Update generated expectations or fixtures.
3. Update RTL.
4. Update software/firmware if ABI changed.
5. Update docs.
6. Run `make test`.
7. Run `make perf-report` if timing or report output changed.
