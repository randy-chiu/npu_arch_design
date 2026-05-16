# Firmware Smoke Program

The preferred PicoRV32 SoC smoke firmware now lives as normal bare-metal source
code under:

```text
sw/soc_cpu/boot/start.S
sw/soc_cpu/runtime/npu_driver.c
sw/soc_cpu/apps/soc_cpu_smoke/main.c
```

The linker script is generated from `arch/configs/soc_v0.jsonc`:

```text
build/soc/soc_v0.ld
```

Generate it with:

```text
make firmware-smoke
```

Outputs:

```text
build/firmware/soc_cpu_smoke.hex
build/firmware/soc_cpu_smoke.elf
build/firmware/soc_cpu_smoke.dump
```

`soc_cpu_smoke.hex` is loaded by `boot_rom` in `cpu-soc-sim`.
`soc_cpu_smoke.dump` is the readable disassembly when a RISC-V GCC toolchain is
available.

In the current simulation, `soc_cpu_smoke.hex` is the full smoke firmware image,
not only a minimal boot stub. It contains startup code, the NPU MMIO driver,
`main()`, and generated test data. PicoRV32 executes it in place from the boot
ROM address range.

A production-like SoC may instead keep only first-stage boot code in ROM, then
load user firmware from flash or another non-volatile image into SRAM/DRAM
before jumping to it. That loader flow is not modeled yet.

If no RISC-V GCC is installed, `make firmware-smoke` falls back to
`sw/tools/firmware/emit_soc_cpu_smoke.py`. In that mode it also emits
`build/firmware/soc_cpu_smoke.S`, a generated RV32I assembly listing.

Both paths write NPU wrapper MMIO windows, start the NPU by writing
`CTRL.start`, poll `STATUS.done`, check outputs, and write the simulation
test-status register.
