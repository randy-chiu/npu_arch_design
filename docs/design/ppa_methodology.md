# ASIC-Oriented PPA Methodology / 面向 ASIC 的 PPA 方法

[TOC]

## 1. Goal / 目标

This project uses one PPA report family to guide NPU architecture decisions:

```text
functional correctness
  -> measured performance and traffic
  -> structural area and event-energy models
  -> technology-mapped area/timing estimate
  -> physical/activity-driven power estimate
  -> energy and efficiency comparison
```

本项目的近期目标不是得到流片 signoff 结论，而是在没有私有工艺库和板卡的条件
下，建立可重复、可比较、面向 ASIC 的架构评估框架。公开 PDK/开源 flow 的结果
用于比较架构 variant，不应表述为最终芯片指标。

The methodology is layered so that early architecture analysis does not block
on a heavy physical-design tool flow before the evaluation boundary and
representative workloads are stable.

本方法采用分层推进：当前先建立真实性能计数加结构/事件模型，不要求一开始就
安装完整物理实现工具链；只有通过早期筛选的架构 variant 才进入更重的 ASIC
实现评估。

## 2. Measurement Boundaries / 测量边界

PPA results must name one of these tops:

| Top | Includes | Use |
| --- | --- | --- |
| `npu_core` | matrix/vector/reduction datapath, core control, local core state | microarchitecture breakdown |
| `npu_subsystem` | Host wrapper plus NPU core command processor/data mover/compute cluster | primary NPU architecture decision boundary |
| `soc_reference` | CPU, bus, simulation memories/peripherals, NPU subsystem | system integration reference only |

`npu_subsystem` is the primary comparison boundary because core performance
cannot be evaluated independently from data delivery and launch/control cost.
`npu_core` is still reported for attribution.

当前 CPU SoC smoke 使用的 oversized boot ROM/SRAM 是功能仿真 staging 机制。
除非单独声明为 `soc_reference`，这些仿真容量不得计入 NPU 主 PPA 结论。

## 3. Report Information Architecture / 报告信息架构

PPA means performance, power, and area. The generated HTML report is a report
family with one overview and three detailed pages:

```text
ppa_overview.html
  -> perf.html
  -> power.html
  -> area.html
```

The overview answers:

- what workload profile, graphs, layers, operators, and shapes were tested;
- whether the run is comparable with the selected previous iteration;
- overall performance, power/energy, and area changes;
- which evidence level and units apply to every dimension.

The detailed pages answer:

| Page | Required content |
| --- | --- |
| `perf.html` | graph/layer/operator structure, shapes, theoretical versus measured cycles, per-job module timeline, bottleneck explanation |
| `power.html` | workload event/energy breakdown now; activity-based power when available |
| `area.html` | module/resource area breakdown now; mapped/physical area when available |

All generated PPA artifacts belong under one directory:

```text
build/ppa/
  ppa_overview.html
  perf.html
  power.html
  area.html
  cases/<workload_profile>.html
  data/
```

There is no separate generated `build/perf/` report directory. Performance is
one dimension of PPA, and its logs/JSON are internal data under
`build/ppa/data/`.

The overview must name the selected `WORKLOAD_PROFILE` as the tested case and
link to its case page. For example, a run with
`WORKLOAD_PROFILE=transformer` creates a Transformer PPA entry. Its case page
contains:

- the computation graph executed by that case;
- each layer/operator shape and dataflow edges;
- theoretical and measured performance, power/energy, and area attribution;
- the aligned per-operator pipeline timeline.

Test-case pages are user-facing architecture views:

- computation graphs use model-viewer-style operator cards with readable
  stage names, tensor inputs/outputs, shapes, and directed edges;
- readable labels such as `Decode Projection GEMM` are primary; internal
  workload IDs remain secondary correlation details;
- pipeline lanes use stable module-specific colors consistently across pages;
- pipeline lanes show the CPU-visible wrapper separately from the NPU core;
  command processor, data mover, and compute cluster are nested under NPU
  core, and execution engines are nested under the compute cluster;
- measured versus state-machine-derived timing is identified by provenance,
  not by collapsing every lane to one color.

### Timeline Truthfulness Gate

The primary purpose of the performance timeline is to let an architect inspect
the real internal execution order and identify optimization opportunities.
Every architecture iteration must therefore pass a timeline truthfulness gate:

- every displayed work span names the real hardware module or explicitly names
  control logic inside a module;
- every span start/end comes from a cycle event, not from proportional
  allocation of an aggregate total;
- child execution/control spans account for every compute-cluster active cycle,
  with overlap counted explicitly rather than added twice;
- spans stay within the job measurement interval;
- inactive, bypassed, waiting, and working are distinct states;
- irrelevant empty lanes are omitted;
- lane totals reconcile with authoritative completed-job counters;
- tooltip text states the operation, source/destination or wait reason, and
  cycle interval.

If any gate fails, the report must label the timeline incomplete and the
architecture change cannot use that timeline as optimization evidence.

The word `model` must not appear in user-facing report names or page labels.
When physical power or area is unavailable, the page remains the power or area
page and explicitly labels the current evidence as a normalized model.

## 4. Evidence Views / 证据视角

The previous `L0/L1/L2/L3` names incorrectly suggested that one result replaced
another. These are parallel views of the same named architecture variant.
Every result must state its evidence view and metric provenance.

| Evidence view | Method | Current role | Claim boundary |
| --- | --- | --- | --- |
| `rtl_workload_view` | Architectural perf CSR snapshots plus structural resources and parameterized event-energy coefficients | fast workload-driven architecture iteration | cycles/traffic measured; area/energy are normalized estimates only |
| `mapped_area_timing_view` | Yosys/ABC mapping with a public Liberty library; optionally OpenSTA | check resource cost and timing feasibility | pre-layout mapped area/timing estimate |
| `activity_power_view` | mapped netlist/library plus workload activity | workload-sensitive on-chip power/energy trend | pre-layout activity-driven estimate |
| `physical_implementation_view` | OpenROAD/OpenLane placement, CTS, routing, parasitic-aware reports | selected-variant validation | public-process physical estimate, not signoff |

The views coexist for the same architecture variant; none replaces another:

```text
one RTL/config/commit variant
  -> rtl_workload_view: simulate workloads to measure cycles, events, traffic
  -> mapped_area_timing_view: map the same RTL to estimate area and timing
  -> activity_power_view: apply workload activity to estimate power/energy
```

The RTL workload view remains the fast workload-sensitive loop. The mapped
area/timing view answers whether the resources and one-cycle assumptions are
credible under a declared library and clock constraint. Reports may correlate
views only when variant ID, RTL/config revision, synthesis top, and relevant
parameterization match.

这些证据视角应长期并存。RTL workload view回答workload执行了多少cycle、
发生了多少搬运和等待；mapped area/timing view回答并行资源、宽端口和
单cycle假设需要多少面积、能否达到目标时钟。只有variant、RTL/config版本、
综合top和参数一致时，才能将不同视角组合成同一个架构结论。

Current machine-readable schemas and make targets still contain legacy
`L0_model`/`ppa-l0-*` identifiers. P8a must migrate those identifiers to view
names without changing the underlying measured results or baseline identity.

Current executable view:

```text
rtl_workload_view:
  measured performance and movement counters
  structural area model
  event-based energy model
```

The existing `flows/asic/openroad/` directory is retained as the future
`physical_implementation_view` integration point.

当前第一步输出必须明确注明：cycle 与 movement 来自 RTL 实测；area 是结构
模型；energy 是基于事件和系数的模型，不是综合后功耗。

## 4. Technology Target And Interpretation / 工艺目标与解释

Initial target policy:

| Target | Role |
| --- | --- |
| `sky130hd` | future first public physical implementation baseline target |
| `nangate45` | optional fast/reference comparison target |
| `asap7` | optional advanced-node trend comparison after flow stability |

The flow location is:

```text
flows/asic/
```

Machine-readable target assumptions live under:

```text
arch/configs/ppa/
```

For `L1` through `L3`, every report must carry the target, tool revision,
constraints, RTL/config identity, and implementation stage. For `L0_model`,
the report carries its model coefficient configuration and states that no
technology result is used.

## 5. Metrics / 指标

### Performance

Required metrics:

```text
cycles/job
latency/job
compute cycles
movement cycles
stall/wait cycles when available
bytes or words moved
MAC operations
utilization where meaningful
theoretical compute cycles versus measured compute cycles
compute overhead and compute efficiency
non-compute overhead and end-to-end efficiency
```

Theoretical cycle values are analysis models, not measured counters. Reports
must include the formula/basis and keep them separate from measured cycle
provenance. A low compute efficiency points to engine/control inefficiency; a
high compute efficiency with low end-to-end efficiency points to movement,
descriptor, or runtime overhead.

Transformer extensions:

```text
cycles/token
latency/token
tokens/s
weights bytes/token
KV-cache read/write bytes/token
```

### Structural Area Model, Area And Timing

At `L0_model`, area reports structure rather than physical units:

```text
MAC lane count
stored bits by named local buffer
data mover lane count
normalized_area_units from declared coefficients
unmodeled structures and memory-boundary assumptions
```

`normalized_area_units` is only a ranking aid. It must not be presented as
`um^2`, gate equivalents, cell area, or synthesized utilization.

At `L1` through `L3`, required metrics become:

Required metrics:

```text
total cell area
sequential/combinational area breakdown when available
macro area separately reported
critical path / worst slack
target clock period
achieved Fmax estimate
```

Local buffers must be labeled as one of:

```text
flop/register implementation
inferred memory
modeled SRAM macro
integrated SRAM macro
```

Comparing variants with different memory-accounting modes is invalid unless the
difference is explicitly highlighted.

### Event Energy Model, Power And Energy

At `L0_model`, energy is computed from measured or inferred events and declared
replaceable coefficients:

```text
normalized_energy =
    matmul_mac_ops     * E_int8_mac_accumulate
  + mover_read_words   * E_onchip_read_word
  + mover_write_words  * E_onchip_write_word
  + external_bytes     * E_external_byte      // when provided by a manifest
```

This model intentionally does not claim leakage power, clock-tree power, wire
power, or real joules. It shows whether a proposed architecture exchanges
latency for more stored bits or traffic/compute event cost.

Required metrics once the tool flow exists:

```text
leakage power
dynamic power
total power
activity capture interval
energy/job
pJ/MAC for matrix workloads
```

Transformer extensions:

```text
energy/token
on-chip energy/token
estimated external-memory energy/token
```

For LLM decode, reporting on-chip energy without weight/KV-cache traffic is
insufficient. The first implementation may model external-memory energy, but it
must report modeled energy separately from synthesized on-chip power.

## 6. Activity Windows / 活动采集窗口

Two power scopes are permitted:

| Scope | Definition |
| --- | --- |
| `kernel_only` | activity while the NPU job or grouped NPU workload runs |
| `staging_inclusive` | includes defined CPU/DMA/input/output staging phases |

The current `PERF_JOB` boundary is the starting point for `kernel_only`
capture. Initial results should prefer `kernel_only` for NPU microarchitecture
comparison and add `staging_inclusive` when system movement is being judged.

## 7. Workloads / 工作负载

Required baseline set:

| Group | Workloads |
| --- | --- |
| Smoke | matmul, softmax |
| CNN compatibility | real MNIST CNN `fc1` K-stream and later `fc1 -> fc2` |
| Transformer micro | GEMM, GEMV/skinny GEMM, RMSNorm, attention kernels, KV-cache traffic |
| Transformer block | tiny decoder-only block after kernel support exists |

MNIST validates the existing integration path. Transformer workloads determine
future compute, vector/reduction, precision, and memory-system priorities.

## 8. Result Contract / 结果合同

Checked-in schema and baseline summaries live under:

```text
ppa/schema/
ppa/baselines/
```

Generated run artifacts live only under:

```text
build/ppa/
```

Every meaningful architecture iteration should record:

```text
design/config/RTL identity
technology target and tool flow
workload and activity scope
functional status
performance metrics
area/timing metrics
power/energy metrics when available
external-memory estimate when applicable
delta versus named baseline
interpretation and remaining bottleneck
```

### Baseline Delta Rule / 基线差异规则

After a named NPU baseline exists, every architecture iteration that changes
RTL behavior or hardware structure must emit a candidate-versus-baseline
comparison for the relevant workload set.

建立命名基线后，每次改变 RTL 行为或硬件结构的 NPU 迭代都必须针对相关 workload
输出 candidate 与 baseline 的差异对比。

The comparison must contain:

```text
baseline variant and candidate variant
evidence level and measurement/model provenance
workloads common to both variants
latency/cycle delta
movement/operation invariant or delta
area model or measured-area delta when available
energy model or measured-energy delta when available
improvements
costs/regressions
metrics unavailable at the current evidence level
```

A report is not allowed to present only favorable metrics. A design can be
preferred only after its benefits and costs are visible together. At
`L0_model`, this means that a performance improvement may be highlighted, but
any added buffer bits, lane resources, modeled energy increase, or unknown
external-memory cost must remain visible.

报告不得只呈现优势指标。只有收益与代价同时可见时，才能判断新设计是否更好。在
`L0_model` 阶段，即使性能提升，也必须显示新增 buffer bit、lane 资源、模型能耗
变化以及尚未覆盖的外部 memory 代价。

Level 0 generated output is written under:

```text
build/ppa/ppa.json
build/ppa/ppa_overview.html
```

The executable `L0_model` report contract is documented in
`ppa/schema/ppa_schema_v0.md` and validated by
`sw/tools/ppa/schema_check.py`. `normalized_area_units` and
`normalized_energy_units` are not physical area or joules. A report is
directly comparable to a baseline only when schema, evidence level, model
coefficient model/units, common workload names, and declared
`workload_manifest_id` are compatible.

`make ppa-l0-report` remains the complete validation gate because it
regenerates RTL-measured performance. When an existing `build/ppa/data/perf.json`
is already valid and only schema/report presentation is changing,
`make ppa-l0-from-perf` performs the derived Level 0 generation and
validation without repeating the long SoC run.

The older `ppa-model-*` targets remain Makefile aliases for compatibility. New
documentation should prefer `ppa-l0-*` because `model` can be mistaken for a
software or network agent rather than a Level 0 estimate.

Named baseline summaries are checked in under:

```text
ppa/baselines/
```

## 9. Development Order / 开发顺序

1. `rtl_workload_view`: integrate current RTL counters, structural area config, and
   event-energy config into one comparable report.
2. Use existing matmul/data-mover/ping-pong measurements and Transformer
   workload manifests to check whether the model exposes useful tradeoffs.
3. `mapped_area_timing_view`: add lightweight Yosys/ABC mapping and optional OpenSTA timing
   for the retained baseline before accepting substantial datapath widening,
   additional storage ports, larger arrays, or fusion structures. Continue
   L1 ranking for representative candidates in parallel with workload growth.
4. `activity_power_view`: add job-scoped activity-based on-chip power extraction.
5. `physical_implementation_view`: use SKY130HD/OpenROAD/OpenLane for selected variants after
   memory accounting and Transformer evaluation shapes are stable.

Transformer counter expansion order:

1. define and test one job measurement interval so no trace event lies outside
   `total_cycles`;
2. define stable semantic command, compute-control, engine-active, wait-reason,
   and movement events;
3. add automatic timeline conservation checks for every measured job;
4. keep current wrapper CSR snapshot as the production aggregate source and
   cross-check it against cycle events;
5. aggregate stable events through wrapper-visible completed-job snapshots;
6. extend `PERF_JOB` and PPA schema with provenance labels for each new field;
7. update energy coefficients only after event names and units are stable.

Reports must not mix measured primitive counters with derived/model-only
attention fields without separate provenance fields.
