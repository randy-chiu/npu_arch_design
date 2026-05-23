# FC1 K-Streaming Matmul Design / FC1 K 轴流式矩阵乘设计

[TOC]

This document defines the hardware/software contract for mapping the real MNIST
CNN `fc1: 9216 -> 128` layer onto the current Phase 0 NPU.

本文档定义真实 MNIST CNN `fc1: 9216 -> 128` 映射到当前 Phase 0 NPU 的软硬件
合同。

## 1. Problem / 问题

The real `fc1` layer is:

真实 `fc1` 层是：

```text
flat[9216] * fc1.weight[128, 9216]^T + fc1.bias -> fc1[128]
```

The current NPU core executes one physical `8x8x8` matmul tile:

当前 NPU core 执行的物理矩阵乘 tile 是：

```text
A[8x8] * B[8x8] -> C[8x8]
```

Naively decomposing full `fc1` into current descriptor jobs gives:

如果把完整 `fc1` 直接拆成当前 descriptor job：

```text
K chunks = 9216 / 8 = 1152
N chunks = 128 / 8 = 16
jobs     = 1152 * 16 = 18432
```

That path is not acceptable as the architecture direction. It moves partial
sums through CPU firmware and SRAM and mostly measures descriptor/program/input/
output overhead.

这不是合理的架构方向。它会让 partial sum 经过 CPU firmware 和 SRAM，主要测到的
是 descriptor、program、input、output 的重复搬运开销，而不是 NPU 的真实算力。

## 2. Current Compute Parallelism / 当前计算并行度

The physical `matmul_array.sv` remains parameterized as:

当前 `matmul_array.sv` 的物理参数保持为：

```text
M = 8
N = 8
K = 8
```

Per active cycle, the array iterates over all output coordinates `(i, j)` in
one clocked RTL block:

每个 active cycle 内，阵列在同一个时钟过程里对所有 output 坐标 `(i, j)` 并行
更新：

```text
for i in 0..7:
  for j in 0..7:
    result[i,j] += A[i,k_idx] * B[k_idx,j]
```

Important timing interpretation:

关键时序解释：

- the nested RTL `for i/j` loops describe parallel register updates, not
  software-serial loop execution;
- `k_idx` is the only loop dimension that advances across cycles;
- one active cycle performs 64 signed int8-by-int8 MACs into int32
  accumulators;
- one `8x8x8` tile therefore needs 8 active MAC cycles;
- the observed `core.matmul` phase is about 10 cycles because it also includes
  start/done/commit state-machine overhead.

- RTL 里的嵌套 `for i/j` 表示同拍并行寄存器更新，不是软件顺序循环；
- 只有 `k_idx` 维度跨 cycle 前进；
- 每个 active cycle 完成 64 个 signed int8-by-int8 MAC，累加到 int32；
- 一个 `8x8x8` tile 需要 8 个 active MAC cycle；
- 当前 perf 中观测到的 `core.matmul` 约为 10 cycles，因为还包含 start/done/commit
  状态机开销。

The physical compute width is therefore:

所以当前物理计算宽度是：

```text
64 MAC/cycle
8 K cycles per 8x8x8 tile
```

### 2.1 Cycle-By-Cycle Example / 逐拍计算例子

For one physical tile:

对一个物理 tile：

```text
A shape: 8x8
B shape: 8x8
C shape: 8x8
```

Mathematically:

数学上：

```text
C[i,j] = A[i,0]*B[0,j]
       + A[i,1]*B[1,j]
       + ...
       + A[i,7]*B[7,j]
```

The current RTL does **not** compute one `C[i,j]` at a time. Instead, each cycle
selects one K slice and updates all 64 outputs in parallel.

当前 RTL **不是**一次只算一个 `C[i,j]`。它每一拍选择一个 K slice，然后同拍并行
更新 64 个输出元素。

Cycle 0, `k_idx = 0`:

第 0 拍，`k_idx = 0`：

```text
Uses A column 0 and B row 0:

              B[0,0] B[0,1] B[0,2] ... B[0,7]
                |      |      |           |
A[0,0]  ->   C[0,0] C[0,1] C[0,2] ... C[0,7]
A[1,0]  ->   C[1,0] C[1,1] C[1,2] ... C[1,7]
A[2,0]  ->   C[2,0] C[2,1] C[2,2] ... C[2,7]
  ...          ...    ...    ...       ...
A[7,0]  ->   C[7,0] C[7,1] C[7,2] ... C[7,7]

Operation in this cycle:
C[i,j] += A[i,0] * B[0,j]   for all i=0..7, j=0..7
```

This is 64 parallel MACs:

这一拍是 64 个并行 MAC：

```text
8 A values from A[:,0]
8 B values from B[0,:]
outer product -> 8x8 partial C update
```

Cycle 1, `k_idx = 1`:

第 1 拍，`k_idx = 1`：

```text
Uses A column 1 and B row 1:

              B[1,0] B[1,1] B[1,2] ... B[1,7]
                |      |      |           |
A[0,1]  ->   C[0,0] C[0,1] C[0,2] ... C[0,7]
A[1,1]  ->   C[1,0] C[1,1] C[1,2] ... C[1,7]
A[2,1]  ->   C[2,0] C[2,1] C[2,2] ... C[2,7]
  ...          ...    ...    ...       ...
A[7,1]  ->   C[7,0] C[7,1] C[7,2] ... C[7,7]

Operation in this cycle:
C[i,j] += A[i,1] * B[1,j]   for all i=0..7, j=0..7
```

After cycle 1, every `C[i,j]` has accumulated two terms:

第 1 拍结束后，每个 `C[i,j]` 已经累加了两项：

```text
C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
```

The full 8-cycle tile looks like:

完整 8 拍 tile 如下：

| Cycle | K slice | Parallel operation / 并行操作 | Terms accumulated in every C[i,j] / 每个 C[i,j] 已累加项 |
| ---: | ---: | --- | --- |
| 0 | 0 | `C[i,j] += A[i,0] * B[0,j]` for all 64 `(i,j)` | k=0 |
| 1 | 1 | `C[i,j] += A[i,1] * B[1,j]` for all 64 `(i,j)` | k=0..1 |
| 2 | 2 | `C[i,j] += A[i,2] * B[2,j]` for all 64 `(i,j)` | k=0..2 |
| 3 | 3 | `C[i,j] += A[i,3] * B[3,j]` for all 64 `(i,j)` | k=0..3 |
| 4 | 4 | `C[i,j] += A[i,4] * B[4,j]` for all 64 `(i,j)` | k=0..4 |
| 5 | 5 | `C[i,j] += A[i,5] * B[5,j]` for all 64 `(i,j)` | k=0..5 |
| 6 | 6 | `C[i,j] += A[i,6] * B[6,j]` for all 64 `(i,j)` | k=0..6 |
| 7 | 7 | `C[i,j] += A[i,7] * B[7,j]` for all 64 `(i,j)` | k=0..7, complete |

Another way to view it is as 8 outer products:

也可以把它理解成 8 次 outer product：

```text
C = 0
C += A[:,0] outer B[0,:]   // cycle 0
C += A[:,1] outer B[1,:]   // cycle 1
C += A[:,2] outer B[2,:]   // cycle 2
C += A[:,3] outer B[3,:]   // cycle 3
C += A[:,4] outer B[4,:]   // cycle 4
C += A[:,5] outer B[5,:]   // cycle 5
C += A[:,6] outer B[6,:]   // cycle 6
C += A[:,7] outer B[7,:]   // cycle 7
```

The K-streaming job simply repeats this 8-cycle physical tile for multiple
larger-K chunks, while keeping `acc_buf` alive between chunks.

K-streaming job 只是对多个更大 K 的 chunk 重复这个 8 拍物理 tile，同时让
`acc_buf` 在 chunk 之间保持不清零。

## 3. Current Core Storage Map / 当前 Core 内部存储划分

The current NPU core has small internal arrays. Names such as `dram_a` are
historical; they are not external DRAM.

当前 NPU core 内部只有小型数组。`dram_a` 这类名字是历史命名，不代表外部 DRAM。

| Storage | Shape / Entries | Type | Purpose |
| --- | ---: | --- | --- |
| `dram_a` | 64 entries = `8x8` | int8 | Host preload window for A tile / A tile host 写入窗口 |
| `dram_b` | 64 entries = `8x8` | int8 | Host preload window for B tile / B tile host 写入窗口 |
| `spad_a` | 64 entries = `8x8` | int8 | Scratchpad loaded by `LOAD A` / `LOAD A` 后的 scratchpad |
| `spad_b` | 64 entries = `8x8` | int8 | Scratchpad loaded by `LOAD B` / `LOAD B` 后的 scratchpad |
| `acc_buf` | 64 entries = `8x8` | int32 | Resident accumulator and output staging / 常驻累加器和输出暂存 |
| `dram_c` | 64 entries = `8x8` | int32 | Host-readable C output window / host 可读 C 输出窗口 |
| `instr_mem` | 16 entries | uint32 | Micro-op program memory / micro-op 程序存储 |
| `dram_x` | 8 entries | int8 | Softmax input window / softmax 输入窗口 |
| `vec_buf` | 8 entries | int16 | Softmax vector staging / softmax vector 暂存 |
| `dram_y` | 8 entries | uint8 | Softmax output window / softmax 输出窗口 |

For K-streaming, the key point is that `acc_buf[8x8]` stays resident while
`dram_a/dram_b/spad_a/spad_b` are overwritten with the next K chunk.

对 K-streaming 来说，关键是 `acc_buf[8x8]` 在多个 K chunk 之间保持常驻，而
`dram_a/dram_b/spad_a/spad_b` 可以被下一个 K chunk 覆盖。

The design does **not** add an `8x9216` A buffer or `9216x8` B buffer inside the
core.

本设计**不会**在 core 内部增加 `8x9216` 的 A buffer 或 `9216x8` 的 B buffer。

## 4. Design Goal / 设计目标

Keep the physical MAC tile small:

保持小的物理 MAC tile：

```text
M = 8
N = 8
K_STEP = 8
```

Expose a larger logical job to firmware:

向 firmware 暴露更大的逻辑 job：

```text
C[8x8] = A[8xK_TOTAL] * B[K_TOTALx8]
```

Inside one descriptor job, the wrapper streams K chunks and the NPU core keeps
the partial sum resident:

在一个 descriptor job 内，wrapper 沿 K 轴流式搬运 chunk，NPU core 保持 partial
sum 常驻：

```text
acc[8x8] = 0

for k0 in 0..K_TOTAL step 8:
    fetch A_tile[8x8]
    fetch B_tile[8x8]
    acc += A_tile * B_tile

store acc[8x8] once
```

For full `fc1`, the SoC-visible layer should become 16 output-tile jobs instead
of 18432 micro-tile jobs.

对于完整 `fc1`，SoC 可见的 layer 应该变成 16 个 output-tile job，而不是 18432
个 micro-tile job。

## 5. First Implementation Contract / 第一版实现合同

Add a new descriptor op:

新增 descriptor op：

```text
SOC_NPU_JOB_OP_MATMUL_K_STREAM = 3
```

Descriptor fields:

Descriptor 字段：

| Field | Meaning / 含义 |
| --- | --- |
| `op_type` | `SOC_NPU_JOB_OP_MATMUL_K_STREAM` |
| `program_addr` | existing matmul micro-op program / 现有 matmul micro-op 程序 |
| `program_words` | existing program length / 现有程序长度 |
| `input0_addr` | packed A stream base / packed A stream 起始地址 |
| `input0_words` | words per A chunk; first version fixed at 64 / 每个 A chunk 的 word 数，第一版固定 64 |
| `input1_addr` | packed B stream base / packed B stream 起始地址 |
| `input1_words` | words per B chunk; first version fixed at 64 / 每个 B chunk 的 word 数，第一版固定 64 |
| `output_addr` | final C tile output base / 最终 C tile 输出地址 |
| `output_words` | output words; first version fixed at 64 / 输出 word 数，第一版固定 64 |
| `k_chunks` | number of `8x8x8` chunks to accumulate / 需要累加的 `8x8x8` chunk 数 |

The first version intentionally uses packed streams:

第一版刻意使用 packed stream：

```text
A stream layout:
  chunk0 A[8x8], chunk1 A[8x8], ...

B stream layout:
  chunk0 B[8x8], chunk1 B[8x8], ...
```

This avoids adding stride fields before the RTL data mover has a richer address
generator. Later versions can add `a_stride_words`, `b_stride_words`, and
layout metadata so firmware does not need to prepack streams.

这样可以在 data mover 有更完整地址生成器之前，避免过早加入 stride 字段。后续可
以增加 `a_stride_words`、`b_stride_words` 和 layout metadata，让 firmware 不必
预先 pack stream。

## 6. Wrapper Behavior / Wrapper 行为

For `MATMUL_K_STREAM`, the wrapper FSM performs:

对于 `MATMUL_K_STREAM`，wrapper FSM 执行：

```text
DESC_READ
FETCH_PROGRAM
WRITE_CORE_ACC_CTRL(clear=1, accumulate=1)
for chunk in 0..k_chunks-1:
    FETCH_INPUT0 from input0_addr + chunk * input0_words * 4
    FETCH_INPUT1 from input1_addr + chunk * input1_words * 4
    START_CORE
    WAIT_CORE
WRITE_OUTPUT
WRITE_CORE_ACC_CTRL(clear=0, accumulate=0)
DONE
```

The core still executes the existing matmul program. The difference is that
core control tells `MATMUL` to accumulate into `acc_buf` instead of overwriting
it.

core 仍然执行现有 matmul 程序。区别是 core control 会告诉 `MATMUL` 累加到
`acc_buf`，而不是覆盖 `acc_buf`。

## 7. Core Behavior / Core 行为

Add a small core control register in the host window:

在 host window 中增加一个小的 core control register：

```text
host_addr 0x500:
  bit 0: matmul_accumulate_enable
  bit 1: clear_accumulator pulse
```

Current normal matmul:

当前普通 matmul：

```text
acc_buf = A_tile * B_tile
```

K-streaming matmul:

K-streaming matmul：

```text
acc_buf += A_tile * B_tile
```

The physical MAC array remains `8x8x8`.

物理 MAC array 仍然保持 `8x8x8`。

## 8. First Verification Step / 第一阶段验证

The first RTL smoke should not try to stage all 1152 chunks of a full FC1
output tile through the current boot ROM. Current memory limits are:

第一版 RTL smoke 不应该尝试把完整 FC1 output tile 的 1152 个 chunks 都塞进当前
boot ROM。当前内存限制是：

```text
boot ROM: 32 KiB
SRAM:     128 KiB
```

A full packed stream for one `fc1` N tile would need:

一个 `fc1` N tile 的完整 packed stream 需要：

```text
A stream: 1152 * 64 words
B stream: 1152 * 64 words
total:    147456 words before output/program/firmware
```

That exceeds the current firmware image and SRAM budget.

这超过了当前 firmware image 和 SRAM 预算。

Therefore the first implementation verifies the new contract with a real FC1
derived multi-chunk stream:

因此第一版实现使用真实 FC1 派生的 multi-chunk stream 验证新合同：

```text
real_mnist_cnn_fc1_k_stream_smoke
  sample: MNIST test sample 0
  chunks: 4 selected nonzero FC1 K chunks
  output: one 8x8 N tile
```

This proves:

这证明：

- descriptor ABI can express a K-streaming job;
- wrapper loops over K chunks within one descriptor;
- core accumulator persists across chunk launches;
- output is written once and checked by firmware;
- perf report can classify the new job.

- descriptor ABI 能表达 K-streaming job；
- wrapper 能在一个 descriptor 内循环多个 K chunk；
- core accumulator 能跨多次 chunk launch 保持；
- output 只写回一次并由 firmware 校验；
- perf report 能识别这个新 job。

Full `fc1` layer execution is a follow-up after compact staging or an external
load path exists.

完整 `fc1` layer 执行需要等 compact staging 或外部加载路径具备后再做。

## 9. Planner And Full N-Tile Artifact / Planner 与完整 N-tile Artifact

The K-streaming job generator is now a compiler-side planning step, not private
logic hidden inside the firmware smoke data emitter.

K-streaming job 的生成现在是 compiler 侧的 planner 步骤，而不是藏在 firmware
smoke data emitter 里的私有逻辑。

Planner input:

Planner 输入：

```text
A:        [8, K] int8 activation tile rows
B:        [K, N_total] int8 weight matrix in hardware-facing layout
n_offset: output-column tile base, multiple of 8
K_STEP:   8 for the current physical core
```

Planner output:

Planner 输出：

```text
k_chunks:    number of physical 8x8x8 chunks
k_offsets:   K base offset per chunk
a_stream:    k_chunks * [8, 8] int8 packed A chunks
b_stream:    k_chunks * [8, 8] int8 packed B chunks
expected_c:  [8, 8] int32 accumulated output tile
metadata:    M/N/K_STEP, per-chunk word counts, output word count
```

For the current real MNIST CNN `fc1` first output N tile:

对于当前真实 MNIST CNN `fc1` 的第一个 output N tile：

```text
logical shape seen by software: A[8, 9216] * B[9216, 8] -> C[8, 8]
physical core chunk:           A[8, 8]    * B[8, 8]    -> partial C[8, 8]
k_chunks:                      9216 / 8 = 1152
```

The full single-N-tile artifact is therefore a list of 1152 physical chunks
with one logical accumulated `expected_c`. For the current bring-up checkpoint,
the generated C firmware data also includes this full packed stream, and the CPU
smoke copies it into enlarged simulation SRAM before launching one
`MATMUL_K_STREAM` descriptor.

因此，完整 single-N-tile artifact 是 1152 个 physical chunks 加一个逻辑累加后的
`expected_c`。当前 bring-up checkpoint 中，生成的 C firmware data 也包含这份完整
packed stream，CPU smoke 会先把它拷贝到放大后的仿真 SRAM，再发起一个
`MATMUL_K_STREAM` descriptor。

This intentionally enlarges the simulation boot ROM and SRAM. It is not the
final deployment model; it is a direct architecture checkpoint for verifying
full K-axis accumulation in the NPU.

这会有意放大仿真 boot ROM 和 SRAM。它不是最终部署模型，而是用于验证 NPU 内部完整
K 轴累加能力的直接架构 checkpoint。

Current coding boundary:

当前编码边界：

- firmware smoke uses the shared planner to emit the existing 4-chunk
  `real_mnist_cnn_fc1_k_stream_smoke`;
- the test suite also builds a full 1152-chunk FC1 single-N-tile plan in memory
  and compares it with direct logical matmul;
- the CPU-controlled SoC smoke now stages the full 1152-chunk packed stream in
  SRAM and verifies one full FC1 output N tile;
- full `fc1` layer execution still needs 16 output N-tile jobs and post-matmul
  bias/ReLU handling.

- firmware smoke 使用共享 planner 生成现有 4-chunk
  `real_mnist_cnn_fc1_k_stream_smoke`；
- 测试套件会在内存中生成完整 1152-chunk FC1 single-N-tile plan，并和直接逻辑
  matmul 对比；
- CPU-controlled SoC smoke 现在会把完整 1152-chunk packed stream 展开到 SRAM，
  并验证一个完整 FC1 output N tile；
- 完整 `fc1` layer 执行仍然需要 16 个 output N-tile jobs，以及 matmul 后的
  bias/ReLU 处理。

## 10. Follow-Up Work / 后续工作

1. Keep the `8x8x8` MAC tile unchanged and improve the data path around it.
2. Replace the wrapper-to-core one-word host-window preload path with a real
   movement path.
3. Parameterize and widen movement bandwidth with `WORDS_PER_CYCLE` and
   `SETUP_CYCLES`, backed by a wider core preload interface.
4. Add double buffering so chunk fetch can overlap chunk compute.
5. Add stride/layout fields so streams can be read from natural tensor layout.
6. Replace oversized C/boot-ROM staging with host preload or a loader path for
   large real weight/activation streams.
7. Run full `fc1` as 16 K-streaming N-tile jobs and apply bias/ReLU.
8. Add per-job metadata for `k_chunks`, logical shape, and stream words in
   `PERF_JOB`.
9. Use perf data to decide whether `K_STEP` should grow from 8 to 16/32/64.

1. 保持 `8x8x8` MAC tile 不变，优先改进其周围的数据路径。
2. 把 wrapper-to-core 的逐 word host-window preload 路径替换成真正的数据搬运路径。
3. 用 `WORDS_PER_CYCLE` 和 `SETUP_CYCLES` 参数化并加宽搬运带宽，同时用更宽的
   core preload interface 支撑。
4. 增加双缓冲，让 chunk fetch 和 chunk compute 可以重叠。
5. 增加 stride/layout 字段，让 stream 可以从自然 tensor layout 读取。
6. 用 host preload 或 loader path 替代当前放大的 C/boot-ROM staging，把大规模真实
   weight/activation stream 从 boot ROM 中移出去。
7. 用 16 个 K-streaming N-tile job 跑完整 `fc1`，并处理 bias/ReLU。
8. 在 `PERF_JOB` 中加入 `k_chunks`、logical shape、stream words 等 per-job
   metadata。
9. 用 perf 数据判断 `K_STEP` 是否应该从 8 扩展到 16/32/64。
