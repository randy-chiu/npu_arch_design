# Workloads

Workloads are first-class architecture evaluation inputs rather than only test
fixtures.

```text
manifests/     versioned shape, precision, scenario, and required-metric definitions
smoke/         minimal operator checks
cnn/           compatibility and existing system-regression workloads
transformer/   long-term LLM inference architecture drivers
```

Existing MNIST implementation and fixtures remain in their current tested
locations while manifests and new Transformer work enter here. Migration of
working fixture code should happen only when it reduces duplication or enables
a required comparison.

See `docs/design/transformer/workloads.md`.

The current SoC regression's explicit job-to-workload contract is generated
beside firmware fixture artifacts by `sw/tools/firmware/emit_soc_cpu_smoke_data.py`
into `build/perf/workload_manifest.json`; see
`docs/design/workload_manifest.md`. Versioned Transformer input manifests
remain checked in under `manifests/`.
