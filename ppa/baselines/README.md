# PPA Baselines

New baselines are frozen copies of a validated `ppa_proxy.json`, stored by
evidence level and named architecture variant:

```text
l0/<variant>.json
```

Each frozen baseline must retain the report schema/evidence level, git commit
or revision reference, architecture/config identity, `workload_manifest_id`,
and area/energy coefficient version and values. A candidate must use the same
schema, evidence level, coefficient model, units, and workload manifest before
numerical deltas can be interpreted directly.

`l0/npu_v0_a2_serial_k_stream_proxy.json` is the active frozen Level 0
baseline for the serial-to-ping-pong comparison. Its original recorded
RTL-counter input remains in `l0/npu_v0_a2_serial_k_stream.json` only as source
evidence; new comparison commands consume the validated frozen report.

Without `--baseline-json`, a generated report must set `comparison` to `null`
and its HTML report states that no baseline was provided.
