# Performance Report Tools

This directory contains host-side tools for turning RTL simulation performance
records into machine-readable data and a browser UI.

## Current Flow

```text
make perf-report
```

Outputs:

```text
build/ppa/data/cpu_soc_perf.log
build/ppa/data/workload_manifest.json
build/ppa/data/perf.json
build/ppa/data/pipeline_report.html
```

The current report consumes `PERF_JOB` JSON lines printed by `soc_cpu_tb` from
the values firmware actually reads through the wrapper perf CSR MMIO window.
Each record is labeled `architectural_perf_csr_snapshot` and contains the
stable completed-job summary used by PPA reporting.

The report reads `workload_manifest.json` and emits a `workloads` section for
the current firmware smoke sequence. Each `PERF_JOB` carries the canonical
`job_id` consumed from its descriptor; generated firmware `JOB_ID_*` constants
and manifest entries share one emitter, so semantics do not depend on line
order. Invocation without a manifest is retained for legacy logs only and
prints a warning before order-based inference.

- `operator_smoke_matmul`: the original single matmul job;
- `operator_smoke_softmax`: the original single softmax job;
- `digits_linear_classifier`: 16 tiled `8x8x8` matmul jobs that implement the
  linear digit classifier, with CPU firmware accumulating partial sums and
  checking the predicted label;
- `real_mnist_cnn_fc1_tile0`: 1 tiled `8x8x8` matmul job for the first nonzero
  quantized `fc1` hardware-facing tile from MNIST sample 0;
- `real_mnist_cnn_fc1_k_stream_smoke`: 1 multi-chunk K-streaming matmul job for
  selected real `fc1` chunks, validating accumulator residency within one NPU
  descriptor;
- `real_mnist_cnn_fc1_full_k_stream_tile0`: 1 full single-N-tile K-streaming
  matmul job for real `fc1`, with `k_chunks=1152`;
- `real_mnist_cnn_fc2`: 32 tiled `8x8x8` matmul jobs for the original
  pretrained MNIST CNN's quantized `fc2` hardware-facing view.

## Counter Placement

The wrapper accumulates completed-job snapshot CSRs. Firmware reads `status`,
`job_id`, `op_type`, cycle counters, mover read/write words, and SRAM boundary
words after each descriptor. `soc_cpu_tb` observes those CPU read responses and
serializes them as `PERF_JOB`; the report and PPA tools therefore consume
architectural CSR values rather than TB-constructed metric values.

The TB also accumulates a minimal event reference from core/mover/SRAM signals
and asserts equivalence with snapshot storage. That path is verification-only:
it neither prints production records nor establishes report provenance.

## How One Job Is Timed

The current measurement is a cycle counter around one firmware-launched NPU
descriptor job:

```text
CPU MMIO write CTRL.start
-> wrapper descriptor/program/input/output FSM
-> wrapper publishes completed snapshot
-> firmware reads PERF_* CSRs
-> testbench records those bus-read CSR values as PERF_JOB
```

The architectural CSR bank starts counting when it accepts
`NPU_OPSCHED_CTRL.start`. Stable report fields are:

| Counter | CSR source |
| --- | --- |
| `total_cycles` | `PERF_TOTAL_CYCLES` |
| `core.total`, `core.matmul` | `PERF_CORE_ACTIVE_CYCLES`, `PERF_CORE_MATMUL_CYCLES` |
| `data_mover.*cycles` | `PERF_DATA_MOVER_*_CYCLES` |
| `data_mover.words/read_words/write_words` | `PERF_DATA_MOVER_*WORDS` |
| `sram.read_words/write_words` | `PERF_SRAM_*_WORDS` |

There are no architectural timestamp/phase packets today. Formal production
reports show stable summary values; detailed wrapper/core phase timelines from
hierarchical TB sampling are intentionally no longer treated as measurements.

This means the report measures launch-to-wrapper-completion time for NPU jobs. It
does not yet measure CPU staging before `CTRL.start`, CPU accumulation between
tile jobs, or CPU result checking after wrapper completion.

## Current Status

The first report is tied to the CPU-controlled SoC smoke simulation. It measures
the interval from CPU firmware writing `CTRL.start` to the NPU wrapper finishing
the job. The production HTML report currently shows:

- CSR-sourced completed-job summary values and workload aggregation;
- a workload summary table for grouped operator/model runs;
- per-job total cycles;
- data-movement summary counters from the mover snapshot fields;
- raw JSON for debugging and future tooling.

Current wrapper phases:

| Phase | Meaning |
| --- | --- |
| `desc_read` | wrapper reads `soc_npu_job_desc_t` from SRAM |
| `fetch_program` | wrapper fetches NPU program words from SRAM into the core host window |
| `fetch_input0` | wrapper fetches A/X input data |
| `fetch_input1` | wrapper fetches B input data for matmul |
| `start_core` | wrapper pulses core start |
| `wait_core` | wrapper waits while the core is executing |
| `write_output` | wrapper writes C/Y output data back to SRAM |
| `done` | wrapper done-latch phase |

Current core phases:

| Phase | Meaning |
| --- | --- |
| `fetch` | core-local uop fetch/decode/execute cycles, including current single-cycle LOAD/STORE/vector tasks |
| `matmul` | matmul engine cycles |
| `done` | core done assertion phase |

Historical scalar baseline before A1:

| Job | Total cycles | Core cycles | Dominant cost |
| --- | ---: | ---: | --- |
| `matmul` | 738 | 520 | 512-cycle iterative matmul |
| `softmax` | 53 | 11 | wrapper movement and single-cycle vector tasks |

A1 measured baseline after adding `matmul_array`:

| Job | Total cycles | Core cycles | Dominant cost |
| --- | ---: | ---: | --- |
| `matmul` | 236 | 18 | wrapper input/output movement |
| `softmax` | 53 | 11 | wrapper movement and single-cycle vector tasks |

The current report also includes a `Matmul model` panel for matmul jobs:

| Model | Meaning |
| --- | --- |
| measured compute | RTL-measured `core.matmul` cycles |
| scalar baseline | old 8x8x8 single-lane MAC estimate, 512 cycles |
| ideal 8x8 array | one K slice per cycle, 8 cycles |
| conservative array | ideal plus small control allowance, 12 cycles |
| projected total | current non-matmul cycles plus conservative array estimate |

The A1 result confirms the expected bottleneck shift: compute drops sharply and
wrapper data movement now dominates. The target architecture direction is
described in `docs/target_architecture.md`.

A2.0 movement profile:

| Job | SRAM read cycles | SRAM write cycles | Core host write cycles | Core host read cycles |
| --- | ---: | ---: | ---: | ---: |
| `matmul` | 153 | 64 | 144 | 64 |
| `softmax` | 33 | 8 | 24 | 8 |

For matmul, data movement still dominates the 82-cycle job. The core matmul
phase is only 10 cycles, while input/program movement and output writeback now
use the `WORDS_PER_CYCLE=4` NPU-side path. This remains the main evidence for
later A2 work on data mover counters, scratchpad banking, and overlap.

Current model profiles:

| Workload | Jobs | Total cycles | Notes |
| --- | ---: | ---: | --- |
| `digits_linear_classifier` | 16 matmul tiles | 1312 | `8x64 * 64x16` lowered into 16 current-RTL-compatible `8x8x8` jobs |
| `real_mnist_cnn_fc1_tile0` | 1 matmul tile | 82 | First nonzero original CNN `fc1` quantized tile, `K=56`, `N=0`; SoC tile checkpoint, not full layer |
| `real_mnist_cnn_fc1_k_stream_smoke` | 1 K-stream job | 186 | Four selected real `fc1` K chunks accumulated inside one descriptor; validates K streaming, not full layer |
| `real_mnist_cnn_fc1_full_k_stream_layer` | 16 K-stream jobs | 627488 | Full `fc1` layer; each output-N descriptor has `k_chunks=1152`, compared against named serial baseline |
| `real_mnist_cnn_fc2` | 32 matmul tiles | 2624 | Original CNN `fc2: 128 -> 10` quantized view lowered into 32 current-RTL-compatible `8x8x8` jobs |

The model profile currently counts only NPU job intervals. CPU-side work between
jobs, including copying tile tensors, accumulating partial sums, and argmax, is
validated by firmware but not yet measured as a separate span.

The UI intentionally does not render separate `SRAM NPU port` and
`Core host window` timelines because those phases currently overlap almost
one-to-one with wrapper phases. Instead, the `Wrapper phases` rows include the
data path detail:

| Wrapper phase | Data path detail |
| --- | --- |
| `Descriptor read` | wrapper reads job descriptor words from SRAM |
| `Program fetch` | wrapper reads program words from SRAM and writes core `instr_mem` through the host window |
| `Input0 fetch` | wrapper reads input0 tensor from SRAM and writes core A/X window |
| `Input1 fetch` | wrapper reads input1 tensor from SRAM and writes core B window |
| `Output writeback` | wrapper reads core C/Y output window and writes result words to SRAM |

Here `core host write/read` means the wrapper is using the NPU core's host
window interface. It does not mean the core is independently writing SRAM.

The core `Uop fetch/execute` phase is different from wrapper `fetch_*` phases.
Wrapper fetch phases move data/program words from SRAM into the core before
launch. Core uop fetch/execute happens after launch, inside `npu_v0_top`, where
the core reads its already-loaded `instr_mem` and executes micro-ops such as
`LOAD`, `STORE`, vector operations, and `MATMUL`.

The current host-window preload/readback path is temporary. It keeps functional
bring-up simple, but it requires program words to fit in fixed-size `instr_mem`
before launch and serializes tensor movement around compute. A2 should replace
this with explicit data mover, burst, scratchpad banking, and later instruction
buffer/prefetch behavior. The working plan is in `docs/data_mover_a2.md`.

## Timeline Semantics

Legacy phase-rich records can still be rendered by the report parser for
historical log replay. Production CSR-sourced records intentionally do not
claim detailed wrapper/core phase measurement. Historical reconstruction used:

```text
CPU firmware:  MMIO start -> poll/wait for done
NPU wrapper:   desc/program/input -> start core -> wait core -> output
Data mover:         program/input load                 output store
NPU core:                              fetch/execute/done
```

The key use is spotting pipeline structure and wasted time. For example,
`wait_core` overlapping with core `matmul` is expected. A long CPU wait with no
matching wrapper/core work would indicate missing accounting or a real stall.

The `Data mover` lane is currently reconstructed from wrapper movement phases.
After A2 RTL grows burst timing or overlap, this lane should come from data
mover state directly and may no longer match wrapper phase boundaries.

## Current Limitations

- CPU time before `CTRL.start` is not measured yet. Descriptor construction,
  SRAM buffer writes, and result checking are outside the current timeline.
- CPU time between tiled model jobs is not measured yet. For
  `digits_linear_classifier`, partial-sum accumulation and argmax are validated
  by firmware but not included in workload cycle totals.
- The CPU lane currently models firmware as `MMIO start` followed by polling
  wait; it does not yet break down individual firmware instructions or bus
  transactions.
- The wrapper is still mostly sequential, so there is little true overlap except
  wrapper wait overlapping core execution.
- Core vector operations are still single-cycle RTL tasks, so softmax timing is
  not a realistic vector/SFU pipeline model yet.
- Report counters are read from synthesizable wrapper snapshot CSRs by
  firmware; the testbench hierarchy observer is retained only as an
  implementation-correlation check.

## Extension Points

Near-term extensions:

- Add CPU-side spans before and after NPU launch: input staging, descriptor
  write, program copy, result check, and test-status write.
- Keep workload metadata generation in
  `sw/tools/firmware/emit_soc_cpu_smoke_data.py` synchronized with firmware
  regression jobs as they evolve.
- Split CPU polling into MMIO read transactions and idle cycles.
- Convert SRAM NPU-port read/write spans from simple one-word-per-cycle movement
  into burst/bandwidth-aware data mover counters.
- Add bus-level spans for CPU accesses to SRAM, wrapper registers, and test
  status.
- Emit per-job workload metadata such as tensor shape, program words, input
  words, and output words.

NPU wrapper extensions:

- Separate SRAM request, SRAM ready, host-window write, and output readback.
- Track descriptor validation, error handling, timeout, IRQ, and command queue
  phases once those features exist.
- Show overlap when future DMA/data movement can run concurrently with core
  compute.

NPU core extensions:

- Split fetch, decode, issue, execute, writeback when the core grows a real
  pipeline.
- Track MAC active cycles, accumulator write cycles, vector/SFU active cycles,
  and stalls.
- Add utilization counters: MAC lane utilization, vector lane utilization,
  scratchpad bank conflicts, and instruction bubbles.

Report/UI extensions:

- Add filtering by job/operator/module.
- Add stacked global timeline across multiple jobs instead of one independent
  timeline per job.
- Add critical-path highlighting and idle-gap warnings.
- Add compare mode between two `perf.json` files for architecture experiments.
