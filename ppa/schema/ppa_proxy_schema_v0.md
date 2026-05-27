# PPA Proxy Report Schema v0

## Contract

`npu_ppa_proxy_report_v0` is the machine-readable contract for the current
`L0_proxy` architecture comparison report. It is checked by
`sw/tools/ppa/schema_check.py`; the related JSON Schema is
`ppa/schema/ppa_proxy_report.schema.json`.

This evidence level is deliberately limited:

- `normalized_area_units` is a structural ranking proxy, not `mm^2`, cell
  area, or utilization.
- `normalized_energy_units` is event count multiplied by declared normalized
  coefficients, not joules or measured power.
- external-memory traffic/energy is included only when explicit workload
  metadata supplies it; the current CNN regression generally omits it.
- results support relative architecture comparisons, not implementation
  signoff.

## Required Top-Level Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Must be `npu_ppa_proxy_report_v0`. |
| `evidence_level` | string | Must be `L0_proxy`. |
| `design` | object | Design `top` and `variant` identity. |
| `proxy_config` | object | Area/energy coefficient versions, units, and coefficient sets used for comparability checks. |
| `metric_provenance` | object | Origin and interpretation of performance, area, energy, timing, and power metrics. |
| `area_proxy` | object | Structural resource model and normalized total. |
| `workloads` | array | Per-workload measured performance and modeled event energy. |
| `limitations` | array | Unmodeled quantities and claim boundaries. |
| `comparison` | object or null | Baseline comparison, or `null` when no baseline is provided. |

Optional top-level fields are `source_perf_report`, `workload_manifest_id`,
`run_metadata`, and `highlights`. A checked-in frozen baseline must populate
`run_metadata` with its revision/evidence reference, functional status,
activity scope, and generation command.

## Area Proxy

Required fields are `units`, `resources`, `coefficients`, and
`normalized_area_units`. `units` must be `normalized_area_units`.
`resources` describes structural counts such as INT8 MAC lanes, stored bits,
data mover lanes, and wrapper control units. `coefficients` declares the
normalized weighting model. `contributions` and `excludes` are explanatory
optional fields.

## Workload Performance And Energy

Every `workloads[]` item requires `name`, `performance`, and `energy_proxy`.

| Field | Units / meaning |
| --- | --- |
| `performance.cycles` | RTL-simulation cycles over the declared job boundary. |
| `performance.core_matmul_cycles` | RTL-sampled core matmul cycles. |
| `performance.data_mover_words` | RTL-sampled on-chip mover words. |
| `performance.provenance` | Production value is `measured_architectural_perf_csr_snapshot`; legacy replay may retain `measured_rtl_perf_job_counters`. |
| `energy_proxy.events` | Event counts used by the model, including MAC work and mover traffic. |
| `energy_proxy.coefficients` | Normalized energy/event coefficients. |
| `energy_proxy.normalized_energy_units` | Normalized event-energy total; not joules. |

## Comparison Compatibility

`comparison` is `null` if no baseline is supplied. If supplied, it contains
`comparable`, `compatibility`, candidate/baseline identity, area delta,
per-common-workload cycle/energy/movement/MAC deltas, and improvements/costs.

A direct comparison is valid only when:

- `schema` and `evidence_level` are equal;
- area and energy coefficient versions, coefficient values, and units agree;
- at least one workload name is common;
- `workload_manifest_id` agrees when either report declares one.

When these conditions fail, the report preserves the baseline reference but
sets `comparable` to `false`, lists `compatibility.issues`, and does not
publish numerical deltas as comparable evidence.

The lightweight validator additionally rejects negative structural/event/
performance values, duplicate workload names, incorrect area or energy
contribution totals, and inconsistent comparison deltas.
