# K-Streaming Ping-Pong Buffer Design

[TOC]

This document describes the next NPU-side optimization after explicit data
mover counters: overlapping K-chunk movement with core compute by adding
ping-pong A/B buffers.

本文档描述 explicit data mover counters 之后的下一步 NPU 侧优化：通过 A/B
ping-pong buffer，让 K chunk 的搬运和 core compute 重叠。

## 1. Current Problem / 当前问题

The current full FC1 single-N-tile job is functionally correct:

```text
A[8,9216] * B[9216,8] -> C[8,8]
k_chunks = 1152
```

But the wrapper still executes each K chunk mostly serially:

```text
fetch A chunk i
fetch B chunk i
start core compute chunk i
wait core done
fetch A chunk i+1
fetch B chunk i+1
start core compute chunk i+1
wait core done
...
```

当前 full FC1 single-N-tile 功能已经正确，但每个 K chunk 的搬运和计算仍基本串行。
也就是说，core 计算 chunk `i` 时，data mover 没有同时预取 chunk `i+1`。

Measured full FC1 single-N-tile counters:

```text
total_cycles: 58784
k_chunks: 1152
fetch_input0 cycles: 18432
fetch_input1 cycles: 18432
core matmul cycles: 11520
data_mover transfer cycles: 36884
data_mover read_words: 147472
data_mover write_words: 64
```

The important observation is:

```text
movement and compute are added serially today
```

关键问题：

```text
当前搬运时间和计算时间是串行相加的
```

## 2. Current Serial Timeline / 当前串行时序

For one K chunk, the current timeline is:

```text
time ->

Data mover:  [load A(i)] [load B(i)]                         [load A(i+1)] [load B(i+1)]
Core:                              [compute chunk i]                                      [compute chunk i+1]
acc_buf:                           accumulate C += A(i)*B(i)                              accumulate C += A(i+1)*B(i+1)
```

Per chunk, the cost is close to:

```text
chunk_cycles ~= load_A + load_B + compute
```

当前每个 chunk 的耗时接近：

```text
chunk_cycles ~= load_A + load_B + compute
```

This wastes cycles because the data mover and the MAC array use different
resources. While the core consumes one A/B tile, the data mover could be
preparing the next A/B tile.

这会浪费 cycle，因为 data mover 和 MAC array 使用的是不同资源。core 消费当前 A/B
tile 时，data mover 理论上可以准备下一组 A/B tile。

## 3. Overlapped Timeline / 重叠时序

Ping-pong buffering changes the timeline to:

```text
time ->

Data mover:  [load chunk 0 -> bank0] [load chunk 1 -> bank1] [load chunk 2 -> bank0] [load chunk 3 -> bank1]
Core:                                [compute chunk 0 bank0] [compute chunk 1 bank1] [compute chunk 2 bank0]
acc_buf:                             accumulate chunk 0      accumulate chunk 1      accumulate chunk 2
```

After the initial fill, each steady-state step becomes close to:

```text
chunk_cycles ~= max(load_A + load_B, compute)
```

而不是：

```text
chunk_cycles ~= load_A + load_B + compute
```

This is the main advantage of overlap. It does not reduce the number of bytes
moved and it does not increase MAC count. It hides movement behind compute when
possible.

overlap 的优势不是减少数据量，也不是增加 MAC 数量，而是在可能的时候把搬运隐藏在
计算后面。

## 4. Buffer Structure / Buffer 结构

Current core storage has one A preload path and one B preload path:

```text
dram_a -> spad_a -> matmul array
dram_b -> spad_b -> matmul array
acc_buf stays resident across K chunks
```

Ping-pong adds two banks for A and B:

```text
             +-------------------+
Data mover ->| A bank 0 / B bank 0|--> Core compute
             +-------------------+

             +-------------------+
Data mover ->| A bank 1 / B bank 1|--> Core compute
             +-------------------+

acc_buf remains single and resident:

acc_buf += A(active_bank) * B(active_bank)
```

中文描述：

```text
load_bank   = data mover 正在写入的 bank
compute_bank = core 正在读取计算的 bank

chunk 完成后:
  load_bank 和 compute_bank 交换
```

`acc_buf` 不做 ping-pong。它必须保持单一 resident accumulator，因为 K 轴多个
chunk 都要累加到同一个 `C[8,8]` output tile。

### 4.1 Bank Principle And Ownership / Bank 原理与归属

A bank is not a different kind of data. It is an access-partitioning mechanism:
two equivalent storage slots hold the same kind of tile, and the control logic
decides which slot is owned by the producer and which slot is owned by the
consumer at a given time.

Bank 不是一种新的数据类型，而是一种访问隔离机制：两块等价的存储都能保存同一种
tile，控制逻辑在每个阶段决定哪一块属于生产者、哪一块属于消费者。

For K-streaming ping-pong:

```text
producer: data mover / wrapper, writes the next A/B chunk
consumer: NPU core, reads the current A/B chunk for compute
```

The ownership is temporal, not permanent:

```text
chunk 0:
  bank0 belongs to core      // core computes chunk 0
  bank1 belongs to data mover// data mover prefetches chunk 1

chunk 1:
  bank1 belongs to core      // core computes chunk 1
  bank0 belongs to data mover// data mover prefetches chunk 2
```

中文理解：

```text
某个 bank “属于 core” 的意思是：
  core 本轮 compute 会从这块 bank 对应的数据启动计算，data mover 不允许覆盖它。

某个 bank “属于 data mover” 的意思是：
  这块 bank 当前不被 core 本轮 compute 使用，data mover 可以把下一组 A/B chunk
  写进去。

ownership 会在 chunk 边界交换：
  当前 compute 完成 + 下一 bank preload 完成 后，load_bank 和 compute_bank 对调。
```

This is why banking helps even if the physical storage is still implemented as
register arrays in the current RTL. The important change is not the memory
primitive; it is that the producer and consumer no longer contend for one
logical A/B staging slot.

这也是为什么即使当前 RTL 仍然用 register array 实现存储，bank 依然有意义。关键
变化不是存储 primitive，而是 producer 和 consumer 不再争用同一个逻辑 A/B staging
位置。

The first implementation banks only A/B staging:

```text
A bank 0 / B bank 0
A bank 1 / B bank 1
single spad_a / spad_b
single acc_buf
```

The core latches `compute_bank` when a core launch starts. That latched bank is
used by the `LOAD A` and `LOAD B` uops for the current program execution. The
wrapper may then change the host-write bank for the data mover while the core is
already computing, because the current chunk has already been loaded into
`spad_a/spad_b`.

第一版只 bank A/B staging。core 在 launch 时锁存 `compute_bank`，本轮 program
里的 `LOAD A/B` 从这个 bank 读入 `spad_a/spad_b`。之后 wrapper 可以把 host-write
bank 切到另一块，让 data mover 在 core compute 时预取下一 chunk。

`acc_buf` deliberately remains unbanked. K-streaming requires one resident
accumulator:

```text
acc_buf += A(chunk i) * B(chunk i)
```

Banking `acc_buf` would create two partial sums and require an extra merge or
reduction step, which is not the problem this optimization is trying to solve.

`acc_buf` 刻意不 bank。K-streaming 需要单份常驻 accumulator。如果把 `acc_buf`
也拆成两份，就会产生两份 partial sum，还需要额外 merge/reduce；这不是当前要解决的
瓶颈。

## 5. First RTL Scope / 第一版 RTL 边界

First implementation should be conservative:

1. Keep physical MAC tile unchanged: `8x8x8`.
2. Keep one resident `acc_buf`.
3. Add A/B bank select in the core preload/read path.
4. Add wrapper state for preload-next while compute-current.
5. Support only K-streaming matmul at first.
6. Keep normal single-tile matmul and softmax behavior unchanged.

第一版不要引入复杂队列、乱序或更多 bank。只做 K-streaming matmul 的 ping-pong：

```text
bank 0: current compute or next preload
bank 1: next preload or current compute
```

## 6. Proposed FSM / 建议 FSM

Current K-stream loop:

```text
FETCH_A(i)
FETCH_B(i)
START_CORE(i)
WAIT_CORE(i)
i++
```

Proposed overlapped loop:

```text
PREFILL_A(0)
PREFILL_B(0)

for i in 0..k_chunks-1:
  START_CORE(i, compute_bank)
  if i+1 < k_chunks:
    PREFETCH_A(i+1, load_bank) while core busy
    PREFETCH_B(i+1, load_bank) while core busy
  WAIT_CORE(i)
  swap(compute_bank, load_bank)
```

Important scheduling detail:

```text
If prefetch finishes before compute:
  data mover waits, core continues

If compute finishes before prefetch:
  wrapper must not start next compute until prefetch is complete
```

重要调度细节：

```text
如果 prefetch 先完成:
  data mover 等待，core 继续计算

如果 core 先完成:
  wrapper 不能启动下一个 compute，必须等 prefetch 完成
```

This means the wrapper needs a small two-condition barrier:

```text
next compute can start when:
  previous compute done
  next bank preload done
```

## 7. Expected Impact / 预期收益

For full FC1 single-N-tile today:

```text
data_mover transfer cycles: 36884
core matmul cycles: 11520
```

Because movement is larger than compute, ideal overlap cannot reduce the job to
compute time alone. It should move the K-stream part toward:

```text
rough steady-state lower bound ~= max(total movement, total compute)
```

instead of:

```text
current ~= total movement + total compute + wrapper overhead
```

由于当前 movement 大于 compute，理想 overlap 不能把总时间降到纯 compute time。它
的目标是让 K-stream 主体接近：

```text
max(total movement, total compute)
```

而不是当前的：

```text
total movement + total compute + wrapper overhead
```

The explicit data mover counters added before this step are required to verify
whether the bottleneck shifts from:

```text
movement + compute
```

to:

```text
max(movement, compute) plus synchronization bubbles
```

## 8. Verification Plan / 验证计划

Functional checks:

1. Existing `make npu-core-sim` still passes.
2. Existing `make soc-sim` still passes.
3. Existing `make cpu-soc-sim` still passes.
4. Full FC1 single-N-tile output still matches expected C tile.

Perf checks:

1. `data_mover.words` remains unchanged for the same workload.
2. `core.matmul` remains unchanged for the same workload.
3. full FC1 total cycles should drop if overlap is working.
4. Any remaining bubbles should be visible as wrapper wait or future
   `dm_stall_cycles/core_input_wait_cycles`.

功能验证必须证明结果不变；性能验证必须证明减少的是串行等待，而不是少搬了数据或少
算了 MAC。

## 9. Implementation Result / 实现结果

First implementation status:

- A/B staging is double-buffered in the NPU core.
- Bank 0 keeps the historical `dram_a` / `dram_b` names so existing core tests
  and legacy direct-window debug paths remain stable.
- Bank 1 is added as `dram_a_bank1` / `dram_b_bank1`.
- `spad_a`, `spad_b`, and `acc_buf` remain single-copy.
- Core control register `0x500` now carries:
  - bit 0: `matmul_accumulate_enable`;
  - bit 1: `clear_accumulator` pulse;
  - bit 2: `host_write_bank`;
  - bit 3: `compute_bank_select`.
- The core latches `compute_bank_select` into `compute_bank_active` at launch.
  The current program's `LOAD A/B` uops read from that latched bank.
- The wrapper writes the next bank select after starting the current K chunk,
  then prefetches the next A/B chunk while the core is active.
- The wrapper waits at a two-condition barrier:
  `core_done_seen && next_prefetch_done`.

第一版实现状态：

- NPU core 里的 A/B staging 已双缓冲；
- bank 0 保留历史名字 `dram_a` / `dram_b`，避免破坏现有 core test 和 legacy
  direct-window 调试路径；
- bank 1 新增为 `dram_a_bank1` / `dram_b_bank1`；
- `spad_a`、`spad_b`、`acc_buf` 仍是单份；
- core launch 时锁存 compute bank，本轮 `LOAD A/B` 从锁存 bank 读；
- wrapper 在当前 chunk 启动后切换 host-write bank，并在 core active 期间预取下一
  chunk；
- chunk 边界使用 `core_done_seen && next_prefetch_done` barrier。

Measured result after implementation:

```text
make npu-core-sim: PASS
make soc-sim: PASS
make cpu-soc-sim: PASS, 53 PERF_JOB records
make perf-report: PASS
make test: PASS, 31 tests
```

Full FC1 single-N-tile before overlap:

```text
total_cycles: 58784
data_mover.transfer_cycles: 36884
data_mover.words: 147536
core.matmul cycles: 11520
```

Full FC1 single-N-tile after overlap, after the descriptor acquired its
generated `job_id` word:

```text
total_cycles: 39218
data_mover.transfer_cycles: 36884
data_mover.words: 147536
core.matmul cycles: 11520
```

This confirms the intended effect: transferred words and MAC work stay stable,
while wall-clock job cycles drop because A/B chunk movement is overlapped with
core execution.

这说明优化命中了预期目标：搬运数据量和 MAC 工作量保持不变，job 总 cycle 下降来自
A/B chunk 搬运与 core 执行的重叠。

## 10. Risks / 风险

- Core and data mover may access the same bank if bank ownership is wrong.
- `acc_buf` clear/accumulate timing must remain unchanged.
- Last chunk has no next prefetch and needs clean tail handling.
- Existing normal matmul path must not accidentally require ping-pong state.
- Current core preload host address map may need bank-select extension.

风险重点是 bank ownership：data mover 写的 bank 和 core 读的 bank 不能相同。
另外，`acc_buf` 的 clear/accumulate 语义不能被 ping-pong 破坏。

## 11. Open Questions / 待定问题

Resolved for first implementation:

1. Bank select is encoded in the core control register, not in host address
   bits.
2. Program load remains outside the overlap loop and is done once per
   descriptor.
3. The wrapper infers preload completion from the existing data mover segment
   completion.
4. Core input-wait counters are deferred until the basic overlap path is stable.

Remaining questions:

1. Should data mover expose `preload_done_bank` signals directly, or should the
   wrapper infer completion from existing segment complete?
2. Do we add core input-wait counters now that basic overlap works?
3. Should the perf report render prefetch and compute subspans as overlapping
   lanes instead of only aggregate counters?

第一版已经采用的最小改动：

```text
bank select through core control register
program loaded once per descriptor
wrapper uses data mover segment complete
core input-wait counters deferred until overlap works
```
