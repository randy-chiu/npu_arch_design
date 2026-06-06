# Uop Scheduler RTL

`npu_v0_uop_scheduler.sv` owns common uop fetch, decode, local-operation
dispatch, Matrix-engine issue, and Matrix completion wait for the current
`op=0`/mixed-matmul program path.

The scheduler is separate from the command processor, which owns descriptor,
movement, and K-chunk sequencing. Execution engines do not decode the common
uop stream.

Current limitations:

- vector/reduction/SFU primitive issue has not migrated to this scheduler;
- no out-of-order scheduling, scoreboarding, response queue, or macro-op
  expansion;
- detailed scheduler span placement remains derived, while active/wait totals
  are measured.
