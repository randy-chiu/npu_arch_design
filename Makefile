.PHONY: test demo validate-arch soc-spec npu-wrapper-spec rtl-fixtures firmware-data firmware-smoke-generated firmware-smoke-c firmware-smoke npu-core-sim rtl-sim soc-sim cpu-soc-sim

PYTHONPATH := sw/tools
ARCH := arch/configs/npu_v0.jsonc
SOC := arch/configs/soc_v0.jsonc
NPU_WRAPPER := arch/configs/npu_wrapper_v0.jsonc
RISCV_GCC ?= $(firstword $(shell command -v riscv-none-elf-gcc 2>/dev/null) $(shell command -v riscv32-unknown-elf-gcc 2>/dev/null) $(shell command -v riscv64-unknown-elf-gcc 2>/dev/null))
RISCV_OBJCOPY ?= $(patsubst %-gcc,%-objcopy,$(RISCV_GCC))
RISCV_OBJDUMP ?= $(patsubst %-gcc,%-objdump,$(RISCV_GCC))
RISCV_CFLAGS := -march=rv32i -mabi=ilp32 -mcmodel=medlow -msmall-data-limit=0 -ffreestanding -fno-pic -nostdlib -nostartfiles -Os -Wall -Wextra
RISCV_INCLUDES := -I build/soc -I build/npu_wrapper -I build/firmware -I sw/soc_cpu/runtime

validate-arch:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli validate-arch --arch $(ARCH)

demo:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli demo --arch $(ARCH)

rtl-fixtures:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli emit-rtl-fixtures --arch $(ARCH) --out-dir build/rtl_fixture

soc-spec:
	python sw/tools/soc/emit_soc_spec.py --soc $(SOC) --out build/soc/soc_v0_addr.svh --header-out build/soc/soc_v0_addr.h --linker-out build/soc/soc_v0.ld

npu-wrapper-spec:
	python sw/tools/npu_wrapper/emit_npu_wrapper_spec.py --spec $(NPU_WRAPPER) --svh-out build/npu_wrapper/npu_v0_regs.svh --header-out build/npu_wrapper/npu_v0_regs.h

firmware-data: rtl-fixtures
	python sw/tools/firmware/emit_soc_cpu_smoke_data.py --fixtures build/rtl_fixture --out build/firmware/soc_cpu_smoke_data.h

firmware-smoke-generated: rtl-fixtures soc-spec npu-wrapper-spec
	python sw/tools/firmware/emit_soc_cpu_smoke.py --soc $(SOC) --wrapper $(NPU_WRAPPER) --fixtures build/rtl_fixture --out build/firmware/soc_cpu_smoke.hex

firmware-smoke-c: rtl-fixtures soc-spec npu-wrapper-spec firmware-data
	@test -n "$(RISCV_GCC)" || (echo "No RISC-V bare-metal GCC found. Install riscv-none-elf-gcc, riscv32-unknown-elf-gcc, or riscv64-unknown-elf-gcc."; exit 1)
	mkdir -p build/firmware
	$(RISCV_GCC) $(RISCV_CFLAGS) $(RISCV_INCLUDES) -T build/soc/soc_v0.ld -Wl,-Map=build/firmware/soc_cpu_smoke.map -o build/firmware/soc_cpu_smoke.elf \
		sw/soc_cpu/boot/start.S \
		sw/soc_cpu/runtime/npu_driver.c \
		sw/soc_cpu/apps/soc_cpu_smoke/main.c
	$(RISCV_OBJCOPY) -O binary build/firmware/soc_cpu_smoke.elf build/firmware/soc_cpu_smoke.bin
	$(RISCV_OBJDUMP) -d build/firmware/soc_cpu_smoke.elf > build/firmware/soc_cpu_smoke.dump
	python sw/tools/firmware/bin_to_readmemh.py --in build/firmware/soc_cpu_smoke.bin --out build/firmware/soc_cpu_smoke.hex --words 8192

ifneq ($(RISCV_GCC),)
firmware-smoke: firmware-smoke-c
else
firmware-smoke: firmware-smoke-generated
endif

test:
	PYTHONPATH=$(PYTHONPATH) python -m unittest discover -s test -v

refresh-references:
	PYTHONPATH=$(PYTHONPATH) python scripts/refresh_references.py --output references/discovered_references.md

npu-core-sim: rtl-fixtures
	mkdir -p build
	iverilog -g2012 -I build/rtl_fixture -o build/npu_v0_tb hw/npu_core/rtl/npu_v0_top.sv hw/npu_core/tb/npu_v0_tb.sv
	vvp build/npu_v0_tb

rtl-sim: npu-core-sim

soc-sim: rtl-fixtures soc-spec npu-wrapper-spec
	mkdir -p build/soc
	iverilog -g2012 -I build/rtl_fixture -I build/soc -I build/npu_wrapper -I hw/npu_wrapper/rtl -o build/soc/soc_tb \
		hw/npu_core/rtl/npu_v0_top.sv \
		hw/npu_wrapper/rtl/npu_v0_opsched.sv \
		hw/soc/rtl/bus/simple_bus.sv \
		hw/soc/rtl/mem/boot_rom.sv \
		hw/soc/rtl/mem/simple_sram.sv \
		hw/soc/rtl/debug/test_status.sv \
		hw/soc/rtl/soc_top.sv \
		hw/soc/tb/soc_tb.sv
	vvp build/soc/soc_tb

cpu-soc-sim: firmware-smoke
	mkdir -p build/soc
	iverilog -g2012 -I build/rtl_fixture -I build/soc -I build/npu_wrapper -I hw/npu_wrapper/rtl -o build/soc/soc_cpu_tb \
		hw/soc/cpu/third_party/picorv32/picorv32.v \
		hw/soc/cpu/rtl/picorv32_native_cpu.sv \
		hw/npu_core/rtl/npu_v0_top.sv \
		hw/npu_wrapper/rtl/npu_v0_opsched.sv \
		hw/soc/rtl/bus/simple_bus.sv \
		hw/soc/rtl/mem/boot_rom.sv \
		hw/soc/rtl/mem/simple_sram.sv \
		hw/soc/rtl/debug/test_status.sv \
		hw/soc/rtl/soc_cpu_top.sv \
		hw/soc/tb/soc_cpu_tb.sv
	vvp build/soc/soc_cpu_tb
