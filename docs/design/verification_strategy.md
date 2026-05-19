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

## 8. Current Baselines

After A1 matmul array, A2.1 structural data mover, grayscale digit fixtures,
and real MNIST CNN `fc2` SoC smoke:

```text
make test        PASS, 27 tests
make perf-report PASS
matmul total cycles: 236
matmul core matmul cycles: 10
softmax total cycles: 53
digits_linear_classifier: 16 jobs, 3776 cycles
real_mnist_cnn_fc2: 32 jobs, 7552 cycles
```

When a change intentionally modifies timing, update docs and explain whether
the change is functional RTL behavior or report/model accounting.

## 9. What To Add Next

Near-term verification gaps:

- unit test for `npu_v0_data_mover` edge cases: zero words, one word, multiple
  words, store direction;
- perf regression assertions for expected lane names and key counters;
- timeout/error tests for wrapper stuck-core behavior once timeout exists;
- descriptor validation/error tests once wrapper exposes errors;
- larger transfer tests after counter widths are expanded.

## 10. Test Update Rule

When changing an interface or timing model:

1. Update source-of-truth config if applicable.
2. Update generated expectations or fixtures.
3. Update RTL.
4. Update software/firmware if ABI changed.
5. Update docs.
6. Run `make test`.
7. Run `make perf-report` if timing or report output changed.
