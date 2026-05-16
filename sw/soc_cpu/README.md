# SoC CPU Software

This directory is for bare-metal firmware that runs on the SoC CPU.

Current contents:

- `boot/start.S`: reset entry, stack setup, and `main()` call.
- `runtime/npu_driver.*`: CPU-side NPU wrapper MMIO driver.
- `apps/soc_cpu_smoke/main.c`: firmware smoke app that launches matmul and
  softmax through the NPU wrapper.

The linker script is generated from `arch/configs/soc_v0.jsonc`:

```text
build/soc/soc_v0.ld
```

Build with:

```text
make firmware-smoke
```

When `riscv-none-elf-gcc`, `riscv32-unknown-elf-gcc`, or
`riscv64-unknown-elf-gcc` is available, the Makefile builds this real firmware.
Without a toolchain, it falls back to the temporary Python RV32I emitter so the
simulation path remains usable.
