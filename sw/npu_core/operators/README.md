# NPU Core Operators

Operator definitions consumed by the NPU compiler live here.

This directory is not for CPU driver/runtime code. It describes what the NPU
core should execute for stable operator implementations, using the current NPU
core ISA/uop vocabulary.

Current Phase 0 file:

```text
phase0_intrinsics.json
```

That JSON file describes matmul and softmax lowering templates. The host-side
compiler in `sw/tools/npu_compiler` reads those templates and instantiates them
for graph tensors/shapes. The assembler in `sw/tools/npu_assembler` then encodes
the resulting uop stream into 32-bit program words.
