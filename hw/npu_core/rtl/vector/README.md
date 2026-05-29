# Vector Engine RTL

Planned Transformer NPU v1 location for primitive vector operations:

- `VEC_ADD`
- `VEC_SUB`
- `VEC_MUL`
- `VEC_SCALE`
- `VEC_REQUANT`
- `VEC_CLAMP`

Current module:

| File | Scope |
| --- | --- |
| `vector_engine.sv` | Standalone v1 primitive vector engine for ADD/SUB/MUL/SCALE/REQUANT/CLAMP |
