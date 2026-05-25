# PPA Contracts And Baselines

This directory stores small, reviewable PPA inputs and result summaries:

```text
schema/       machine-readable result contracts
baselines/    selected comparable result summaries
constraints/  measurement policy and future shared constraints
```

Large run artifacts, activity dumps, netlists, logs, and generated reports must
stay under `build/ppa/` and must not be checked in here.

The active methodology is documented in:

```text
docs/design/ppa_methodology.md
```

Current executable level:

```text
make ppa-proxy-report
```

It generates Level 0 output under `build/ppa/proxy/`. Performance/traffic
inputs are RTL-measured counters; area and energy outputs are normalized proxy
values from `arch/configs/ppa/area_proxy_v0.jsonc` and
`arch/configs/ppa/energy_proxy_v0.jsonc`.

Current named baseline:

```text
baselines/l0/npu_v0_a2_serial_k_stream.json
```

It records the verified serial K-stream `fc1` measurement prior to A/B
ping-pong. The current report compares `npu_v0_a2_ping_pong` against that
baseline and must show both the cycle/energy-proxy benefit and the additional
buffer/area-proxy cost.

Baseline policy:

- a baseline identifies a concrete NPU variant and the source of its evidence;
- a candidate must compare only common workloads with compatible metric
  provenance;
- a preferred variant may highlight advantages but must also publish costs,
  regressions, and currently unavailable metrics.
