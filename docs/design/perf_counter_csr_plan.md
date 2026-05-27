# Performance Counter CSR Plan / 性能计数 CSR 计划

## Current Boundary / 当前边界

The production `PERF_JOB`/PPA path now consumes completed-job values read by
firmware from the wrapper CSR snapshot and labels them
`measured_architectural_perf_csr_snapshot`. `hw/soc/tb/soc_cpu_tb.sv` still
observes a minimal event subset only to independently check the CSR
implementation; those reference values are not emitted as report input.

当前正式 `PERF_JOB`/PPA 路径已使用 firmware 通过 MMIO 读取的 wrapper CSR
snapshot，并标记为 `measured_architectural_perf_csr_snapshot`。testbench
仅保留最小事件 reference 用于核对 CSR 实现，不再将逐周期聚合值作为 report 输入。

## First-Batch Register Contract / 第一批寄存器合同

The source of truth is `arch/configs/npu_wrapper_v0.jsonc`. All counter data
registers expose the last completed snapshot and are read-only.

| Register | Offset | Meaning |
| --- | ---: | --- |
| `PERF_CTRL` | `0x040` | write bit 0 to clear the retained snapshot/status while idle |
| `PERF_STATUS` | `0x044` | bit 0 valid, bit 1 running, bit 2 overflow |
| `PERF_TOTAL_CYCLES` | `0x048` | launch-to-completion elapsed cycles |
| `PERF_CORE_ACTIVE_CYCLES` | `0x04c` | cycles attributed to core execution/completion |
| `PERF_CORE_MATMUL_CYCLES` | `0x050` | cycles in the matrix-engine state |
| `PERF_DATA_MOVER_ACTIVE_CYCLES` | `0x054` | cycles in any mover phase |
| `PERF_DATA_MOVER_SETUP_CYCLES` | `0x058` | mover setup cycles |
| `PERF_DATA_MOVER_TRANSFER_CYCLES` | `0x05c` | mover valid-transfer cycles |
| `PERF_DATA_MOVER_STALL_CYCLES` | `0x060` | mover stall cycles |
| `PERF_DATA_MOVER_WORDS` | `0x064` | valid on-chip words moved |
| `PERF_SRAM_READ_WORDS` | `0x068` | words read at the NPU SRAM boundary, including descriptor reads |
| `PERF_SRAM_WRITE_WORDS` | `0x06c` | words written at the NPU SRAM boundary |
| `PERF_JOB_ID` | `0x070` | descriptor workload identity retained in the snapshot |
| `PERF_OP_TYPE` | `0x074` | descriptor operation identity retained in the snapshot |
| `PERF_DATA_MOVER_READ_WORDS` | `0x078` | words moved from SRAM into core host windows |
| `PERF_DATA_MOVER_WRITE_WORDS` | `0x07c` | words moved from core host windows into SRAM |

Semantics:

- an accepted `CTRL.start` clears internal running accumulators and begins a
  new window; the previous completed snapshot remains readable while running;
- completion atomically replaces the visible snapshot and sets `valid`;
- `PERF_CTRL.clear` clears `valid`, `overflow`, and visible snapshot registers
  only while idle; writes while running are ignored;
- all first-batch counters are unsigned 32-bit saturating counters; saturation
  sets the snapshot `overflow` bit;
- there is no manual mid-job snapshot in v0.

语义：

- 接收 `CTRL.start` 后清空内部运行计数并开始新窗口；运行期间仍可读取上一个已完成
  job 的快照；
- job 完成时原子更新可见快照并设置 `valid`；
- 空闲时写 `PERF_CTRL.clear` 清除快照、`valid` 和 `overflow`，运行期间写入被忽略；
- 第一批计数器均为无符号 32 位饱和计数，发生饱和时置快照 `overflow`；
- v0 不提供运行中手动 snapshot。

## Deferred Counters / 延后计数

| Counter / register | Meaning |
| --- | --- |
| `mac_ops` | Count of architecturally committed MAC operations. |
| `instr_count` | Completed NPU instructions/uops. |
| extended `error/status` | Illegal command, timeout, and other execution-error attribution. |

`mac_ops` and `instr_count` remain deferred because their committed-event
definitions are not yet carried as stable core signals. They must not be
inferred silently in an architectural CSR.

## Integration Status / 集成状态

Implemented:

1. `arch/configs/npu_wrapper_v0.jsonc` owns the first-batch register offsets
   and snapshot/clear semantics.
2. `npu_v0_opsched.sv` aggregates the stable counter subset and publishes a
   completed-job snapshot.
3. The core exposes explicit perf events; both descriptor and legacy launch
   paths now populate core counters without wrapper dependence on core FSM
   encodings.
4. Firmware reads the full summary/identity CSR set through MMIO after every
   descriptor completion; `soc_cpu_tb.sv` captures those actual bus read
   values and emits the production `PERF_JOB` record.
5. `soc_cpu_tb.sv` separately checks the CSR backing snapshot against minimal
   event-reference counters; that check is validation only, not report input.
6. `soc_tb.sv` verifies legacy-path MMIO readback, nonzero matmul counting, and
   idle clear behavior.

The formal performance interface is now a CPU/firmware read of defined
CSR/perf registers. Detailed phase timelines are no longer formal production
measurements unless a future architectural event/trace contract is added.
