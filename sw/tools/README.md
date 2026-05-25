# Development-Host Tools

This directory is for software that runs on the development host:

- CPU toolchain integration, such as RISC-V GCC selection;
- NPU graph-to-program compiler code;
- NPU micro-op assembler code;
- Python simulators and golden helpers;
- temporary fixture generation for RTL and SoC simulation.
- PPA result normalization, baseline comparison, and report generation.

The current `npu_phase0` package remains here as a compatibility package while
the compiler, assembler, simulator, and fixture code are split into clearer
tool modules.
