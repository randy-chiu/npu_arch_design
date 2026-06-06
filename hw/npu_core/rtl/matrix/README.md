# Matrix Engine RTL

Current modules:

| File | Scope |
| --- | --- |
| `matmul_array.sv` | Current 8x8 output-parallel int8 matmul array |
| `accumulator_file.sv` | Transformer NPU v1 standalone int32 accumulator file |

`accumulator_file.sv` is now integrated by `npu_v0_compute_cluster` for matmul and
K-stream partial-sum residency. The old internal `acc_buf` storage has been
removed.
