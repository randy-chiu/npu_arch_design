# Software Layout

Software is split by where it runs and what it controls.

- `sw/soc_cpu`: bare-metal software that executes on the SoC CPU, including
  NPU-wrapper drivers, runtime code, boot code, and firmware apps.
- `sw/npu_core`: code or programs consumed by the NPU core itself, such as
  operator programs and later NPU-side operator implementations.
- `sw/tools`: development-host tools, including graph compilers, NPU
  assemblers, simulators, fixture generators, CPU toolchain integration, and
  PPA result normalization/reporting.
