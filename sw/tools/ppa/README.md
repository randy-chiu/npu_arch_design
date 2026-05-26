# PPA Host Tools

This directory owns host-side PPA result normalization, schema validation,
baseline comparison, and report generation.

EDA flow configuration belongs under `flows/asic/`; large generated artifacts
belong under `build/ppa/`; selected small baseline summaries belong under
`ppa/baselines/`.

Current entry point:

```text
PYTHONPATH=sw/tools python -m ppa.proxy_report \
  --perf-json build/perf/perf.json \
  --area-config arch/configs/ppa/area_proxy_v0.jsonc \
  --energy-config arch/configs/ppa/energy_proxy_v0.jsonc \
  --baseline-json ppa/baselines/l0/npu_v0_a2_serial_k_stream_proxy.json \
  --json-out build/ppa/proxy/ppa_proxy.json \
  --html-out build/ppa/proxy/ppa_proxy_report.html
```

This is a Level 0 report: performance is measured from RTL counters; area and
energy are explicitly normalized proxies. The report includes
candidate-versus-baseline deltas and exposes costs as well as improvements.

Build targets:

```text
make ppa-proxy-report       # full perf simulation plus proxy generation/validation
make ppa-proxy-from-perf    # reuse build/perf/perf.json for fast report iteration
```
