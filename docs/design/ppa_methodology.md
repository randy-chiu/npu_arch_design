# ASIC-Oriented PPA Methodology / 面向 ASIC 的 PPA 方法

[TOC]

## 1. Goal / 目标

This project uses PPA evidence to guide NPU architecture decisions:

```text
functional correctness
  -> measured cycles and traffic
  -> structural area and event-energy proxy
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

本方法采用分层推进：当前先建立真实性能计数加结构/事件 proxy，不要求一开始就
安装完整物理实现工具链；只有通过早期筛选的架构 variant 才进入更重的 ASIC
实现评估。

## 2. Measurement Boundaries / 测量边界

PPA results must name one of these tops:

| Top | Includes | Use |
| --- | --- | --- |
| `npu_core` | matrix/vector/reduction datapath, core control, local core state | microarchitecture breakdown |
| `npu_subsystem` | wrapper/scheduler, descriptor/data mover boundary, core | primary NPU architecture decision boundary |
| `soc_reference` | CPU, bus, simulation memories/peripherals, NPU subsystem | system integration reference only |

`npu_subsystem` is the primary comparison boundary because core performance
cannot be evaluated independently from data delivery and launch/control cost.
`npu_core` is still reported for attribution.

当前 CPU SoC smoke 使用的 oversized boot ROM/SRAM 是功能仿真 staging 机制。
除非单独声明为 `soc_reference`，这些仿真容量不得计入 NPU 主 PPA 结论。

## 3. Evidence Levels / 结果可信度层级

Every result must state its evidence level. Metrics from different levels may
appear in one report, but must not be confused.

| Level | Method | Current role | Claim boundary |
| --- | --- | --- | --- |
| `L0_proxy` | RTL `PERF_JOB` counters plus structural resources and parameterized event-energy coefficients | immediate architecture comparison | cycles/traffic measured; area/energy normalized proxies only |
| `L1_mapped` | Yosys/ABC mapping with a public Liberty library; optionally OpenSTA | lightweight ASIC area/timing trend | pre-layout mapped estimate |
| `L2_power` | mapped netlist/library plus workload activity | workload-sensitive on-chip energy trend | pre-layout activity-driven estimate |
| `L3_physical` | OpenROAD/OpenLane placement, CTS, routing, parasitic-aware reports | selected-variant validation | public-process physical estimate, not signoff |

Immediate implementation target:

```text
L0_proxy:
  measured performance and movement counters
  structural area proxy
  event-based energy proxy
```

The existing `flows/asic/openroad/` directory is retained as the future
`L3_physical` integration point. It is not required to begin Level 0
comparison.

当前第一步输出必须明确注明：cycle 与 movement 来自 RTL 实测；area 只是结构
proxy；energy 只是基于事件和系数的模型，不是综合后功耗。

## 4. Technology Target And Interpretation / 工艺目标与解释

Initial target policy:

| Target | Role |
| --- | --- |
| `sky130hd` | future first public physical ASIC baseline target (`L3_physical`) |
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
constraints, RTL/config identity, and implementation stage. For `L0_proxy`,
the report carries its proxy coefficient configuration and states that no
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
```

Transformer extensions:

```text
cycles/token
latency/token
tokens/s
weights bytes/token
KV-cache read/write bytes/token
```

### Structural Area Proxy, Area And Timing

At `L0_proxy`, area reports structure rather than physical units:

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

### Event Energy Proxy, Power And Energy

At `L0_proxy`, energy is computed from measured or inferred events and declared
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
area proxy or measured-area delta when available
energy proxy or measured-energy delta when available
improvements
costs/regressions
metrics unavailable at the current evidence level
```

A report is not allowed to present only favorable metrics. A design can be
preferred only after its benefits and costs are visible together. At
`L0_proxy`, this means that a performance improvement may be highlighted, but
any added buffer bits, lane resources, modeled energy increase, or unknown
external-memory cost must remain visible.

报告不得只呈现优势指标。只有收益与代价同时可见时，才能判断新设计是否更好。在
`L0_proxy` 阶段，即使性能提升，也必须显示新增 buffer bit、lane 资源、模型能耗
变化以及尚未覆盖的外部 memory 代价。

Level 0 generated output is written under:

```text
build/ppa/proxy/ppa_proxy.json
build/ppa/proxy/ppa_proxy_report.html
```

Named baseline summaries are checked in under:

```text
ppa/baselines/
```

## 9. Development Order / 开发顺序

1. `L0_proxy`: integrate current RTL counters, structural area config, and
   event-energy config into one comparable report.
2. Use existing matmul/data-mover/ping-pong measurements and Transformer
   workload manifests to check whether the proxy exposes useful tradeoffs.
3. `L1_mapped`: add lightweight Yosys/ABC mapping and optional OpenSTA timing
   once representative variants need ASIC area/timing ranking.
4. `L2_power`: add job-scoped activity-based on-chip power extraction.
5. `L3_physical`: use SKY130HD/OpenROAD/OpenLane for selected variants after
   memory accounting and Transformer evaluation shapes are stable.
