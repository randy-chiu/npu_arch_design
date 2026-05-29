# Directory Structure And Migration

## Current Implementation

The verified implementation remains in its current paths during baseline
stabilization:

| Path | Status | Ownership |
| --- | --- | --- |
| `hw/npu_core/rtl/` | `current_impl` | `npu_v0_top.sv`, current local memories, uop execution, and staged Transformer NPU v1 module directories |
| `hw/npu_core/rtl/matrix/` | `current_impl` | current matmul array plus standalone accumulator-file module |
| `hw/npu_core/rtl/{vector,reduction,sfu,memory,scheduler,kv_cache}/` | `planned_v1` | staged module ownership directories with README contracts |
| `hw/npu_wrapper/` | `current_impl` | CPU-visible wrapper, descriptor scheduler and data mover |
| `hw/soc/` | `current_impl` | CPU-controlled SoC verification path |
| `sw/tools/`, `sw/soc_cpu/` | `current_impl` | generation tools, firmware/runtime and reports |
| `workloads/`, `ppa/`, `arch/` | active contract paths | workload identity, PPA schema/baselines and source-of-truth configurations |

This round does not move `hw/` into a top-level `rtl/` tree because the current
Makefile, firmware smoke, and PPA boundary depend on those verified paths.
Within `hw/npu_core/rtl/`, module-level directories now mirror the intended v1
architecture so new blocks have clear ownership before a larger top-level
migration.

## Target Layout

```text
rtl/
  npu/
    common/
    wrapper/
    core/
    matrix/
    memory/
    vector/
    reduction/
    sfu/
  soc/
arch/
  specs/
  configs/
sw/
  compiler/
  runtime/
  tools/
    perf/
    ppa/
    golden/
    quant/
    configgen/
workloads/
  smoke/
  cnn/
  transformer/
    micro/
    block/
    models/
ppa/
  schema/
  baselines/
```

`rtl/` is a staged destination, not a second live core today. New substantial
RTL modules should prefer the target ownership boundaries once they are
integrated through compatible build targets; small corrections to the running
implementation remain in `hw/`.

## Unified NPU Direction

`hw/npu_core/rtl/npu_v0_top.sv` is the current unified tensor NPU
implementation. MNIST/CNN remains its regression workload, while future
Transformer workloads drive evolution of the same hardware. It is not a plan
to fork CNN and Transformer cores.

As functionality becomes real architectural blocks, the current top is
expected to separate into:

```text
matrix_engine
accumulator_file
vector_engine
reduction_engine
sfu_engine
memory/data_mover
wrapper/control
```

Migration requires keeping existing `make test`, `make firmware-smoke`,
`make perf-report`, and `make ppa-proxy-report` behavior operational and
keeping generated interfaces sourced from `arch/configs/` or versioned specs.
