# Reduction Engine RTL

Planned Transformer NPU v1 location for row/vector reductions:

- `REDUCE_MAX`
- `REDUCE_SUM`
- `REDUCE_SUMSQ`

Initial maximum row length is 128.

Current module:

| File | Scope |
| --- | --- |
| `reduction_engine.sv` | Standalone v1 primitive reduction engine for MAX/SUM/SUMSQ |
