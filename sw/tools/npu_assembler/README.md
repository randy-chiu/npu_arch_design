# NPU Assembler

Host-side encoding from NPU uop streams into binary instruction words lives
here.

Current Phase 0 entry:

```text
phase0.py
```

It encodes compiler-emitted JSON uops into the temporary 32-bit instruction
format consumed by `hw/npu_core/rtl/npu_v0_top.sv` and by the firmware/RTL
fixture generation flow.
