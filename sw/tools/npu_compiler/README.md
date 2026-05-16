# NPU Compiler

Host-side graph-to-operator-stream and graph-to-uop lowering code lives here.

Current Phase 0 entry:

```text
phase0.py
```

It reads operator templates from:

```text
sw/npu_core/operators/phase0_intrinsics.json
```

and lowers graph ops such as matmul and softmax into the JSON uop stream
validated by the current Phase 0 ISA model.

`sw/tools/npu_phase0/compiler.py` remains as a compatibility wrapper so older
tests and CLI paths can continue importing `npu_phase0.compiler.compile_graph`.
