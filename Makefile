.PHONY: test test-full demo digits-demo validate-arch transformer-config soc-spec npu-wrapper-spec rtl-fixtures firmware-data firmware-smoke-generated firmware-smoke-c firmware-smoke primitive-engines-sim npu-core-sim npu-subsystem-elab rtl-sim soc-sim cpu-soc-sim cpu-soc-quick cpu-soc-transformer cpu-soc-cnn-full cpu-soc-all perf-report perf-l0-quick perf-l0-transformer perf-l0-cnn-full perf-l0-all ppa-l0-from-perf ppa-l0-report validate-ppa-l0

PYTHONPATH := sw/tools
ARCH := arch/configs/npu_v0.jsonc
SOC := arch/configs/soc_v0.jsonc
NPU_WRAPPER := arch/configs/npu_wrapper_v0.jsonc
TRANSFORMER_CONFIG := arch/configs/npu_transformer_v1.jsonc
PPA_AREA_PROXY := arch/configs/ppa/area_model_v0.jsonc
PPA_ENERGY_PROXY := arch/configs/ppa/energy_model_v0.jsonc
PPA_BASELINE := ppa/baselines/l0/npu_v0_a2_serial_k_stream_l0.json
RISCV_GCC ?= $(firstword $(shell command -v riscv-none-elf-gcc 2>/dev/null) $(shell command -v riscv32-unknown-elf-gcc 2>/dev/null) $(shell command -v riscv64-unknown-elf-gcc 2>/dev/null))
RISCV_OBJCOPY ?= $(patsubst %-gcc,%-objcopy,$(RISCV_GCC))
RISCV_OBJDUMP ?= $(patsubst %-gcc,%-objdump,$(RISCV_GCC))
RISCV_CFLAGS := -march=rv32i -mabi=ilp32 -mcmodel=medlow -msmall-data-limit=0 -ffreestanding -fno-pic -nostdlib -nostartfiles -Os -Wall -Wextra
RISCV_INCLUDES := -I build/soc -I build/npu_wrapper -I build/firmware -I sw/soc_cpu/runtime
FIRMWARE_ROM_WORDS ?= 2097152
WORKLOAD_PROFILE ?= quick

validate-arch:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli validate-arch --arch $(ARCH)

demo:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli demo --arch $(ARCH)

digits-demo:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli digits-demo --arch $(ARCH) --label 2

rtl-fixtures:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli emit-rtl-fixtures --arch $(ARCH) --out-dir build/rtl_fixture

transformer-config:
	PYTHONPATH=$(PYTHONPATH) python -m transformer.emit_transformer_config --config $(TRANSFORMER_CONFIG) --out-dir build/generated

soc-spec:
	python sw/tools/soc/emit_soc_spec.py --soc $(SOC) --out build/soc/soc_v0_addr.svh --header-out build/soc/soc_v0_addr.h --linker-out build/soc/soc_v0.ld

npu-wrapper-spec:
	python sw/tools/npu_wrapper/emit_npu_wrapper_spec.py --spec $(NPU_WRAPPER) --svh-out build/npu_wrapper/npu_v0_regs.svh --header-out build/npu_wrapper/npu_v0_regs.h

firmware-data: rtl-fixtures
	python sw/tools/firmware/emit_soc_cpu_smoke_data.py --fixtures build/rtl_fixture --out build/firmware/soc_cpu_smoke_data.h --manifest-out build/ppa/data/workload_manifest.json --workload-profile $(WORKLOAD_PROFILE)

firmware-smoke-generated: rtl-fixtures soc-spec npu-wrapper-spec
	python sw/tools/firmware/emit_soc_cpu_smoke.py --soc $(SOC) --wrapper $(NPU_WRAPPER) --fixtures build/rtl_fixture --out build/firmware/soc_cpu_smoke.hex --manifest-out build/ppa/data/workload_manifest.json

firmware-smoke-c: rtl-fixtures soc-spec npu-wrapper-spec firmware-data
	@test -n "$(RISCV_GCC)" || (echo "No RISC-V bare-metal GCC found. Install riscv-none-elf-gcc, riscv32-unknown-elf-gcc, or riscv64-unknown-elf-gcc."; exit 1)
	mkdir -p build/firmware
	$(RISCV_GCC) $(RISCV_CFLAGS) $(RISCV_INCLUDES) -T build/soc/soc_v0.ld -Wl,-Map=build/firmware/soc_cpu_smoke.map -o build/firmware/soc_cpu_smoke.elf \
		sw/soc_cpu/boot/start.S \
		sw/soc_cpu/runtime/npu_driver.c \
		sw/soc_cpu/apps/soc_cpu_smoke/main.c
	$(RISCV_OBJCOPY) -O binary build/firmware/soc_cpu_smoke.elf build/firmware/soc_cpu_smoke.bin
	$(RISCV_OBJDUMP) -d build/firmware/soc_cpu_smoke.elf > build/firmware/soc_cpu_smoke.dump
	python sw/tools/firmware/bin_to_readmemh.py --in build/firmware/soc_cpu_smoke.bin --out build/firmware/soc_cpu_smoke.hex --words $(FIRMWARE_ROM_WORDS)

ifneq ($(RISCV_GCC),)
firmware-smoke: firmware-smoke-c
else
firmware-smoke: firmware-smoke-generated
endif

test:
	PYTHONPATH=$(PYTHONPATH) python -m unittest discover -s test -v

test-full:
	$(MAKE) test
	$(MAKE) cpu-soc-all

refresh-references:
	PYTHONPATH=$(PYTHONPATH) python scripts/refresh_references.py --output references/discovered_references.md

primitive-engines-sim: transformer-config
	mkdir -p build
	iverilog -g2012 -o build/primitive_engines_tb \
		build/generated/npu_transformer_v1_config_pkg.sv \
		hw/npu_core/rtl/vector/vector_engine.sv \
		hw/npu_core/rtl/reduction/reduction_engine.sv \
		hw/npu_core/rtl/sfu/sfu_lut.sv \
		hw/npu_core/rtl/transformer_primitive_engines.sv \
		hw/npu_core/tb/primitive_engines_tb.sv
	vvp build/primitive_engines_tb
	iverilog -g2012 -o build/primitive_handshake_tb \
		hw/npu_core/rtl/vector/vector_engine.sv \
		hw/npu_core/rtl/reduction/reduction_engine.sv \
		hw/npu_core/rtl/sfu/sfu_lut.sv \
		hw/npu_core/rtl/primitive_handshake_shims.sv \
		hw/npu_core/tb/primitive_handshake_tb.sv
	vvp build/primitive_handshake_tb

npu-core-sim: rtl-fixtures
	mkdir -p build
	iverilog -g2012 -I build/rtl_fixture -o build/npu_v0_tb \
		hw/npu_core/rtl/vector/vector_engine.sv \
		hw/npu_core/rtl/reduction/reduction_engine.sv \
		hw/npu_core/rtl/sfu/sfu_lut.sv \
		hw/npu_core/rtl/matrix/accumulator_file.sv \
		hw/npu_core/rtl/matrix/matmul_array.sv \
		hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv \
		hw/npu_core/rtl/npu_v0_compute_cluster.sv \
		hw/npu_core/tb/npu_v0_tb.sv
	vvp build/npu_v0_tb

npu-subsystem-elab: rtl-fixtures soc-spec npu-wrapper-spec
	mkdir -p build/ppa/elab
	iverilog -g2012 -t null -s npu_subsystem_top -I build/rtl_fixture -I build/soc -I build/npu_wrapper -I hw/npu_wrapper/rtl \
		hw/npu_core/rtl/matrix/matmul_array.sv \
		hw/npu_core/rtl/matrix/accumulator_file.sv \
		hw/npu_core/rtl/vector/vector_engine.sv \
		hw/npu_core/rtl/reduction/reduction_engine.sv \
		hw/npu_core/rtl/sfu/sfu_lut.sv \
		hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv \
		hw/npu_core/rtl/npu_v0_compute_cluster.sv \
		hw/npu_core/rtl/memory/npu_v0_data_mover.sv \
		hw/npu_core/rtl/npu_v0_core_system.sv \
		hw/npu_wrapper/rtl/npu_v0_wrapper.sv \
		hw/npu_subsystem/rtl/npu_subsystem_top.sv

rtl-sim: npu-core-sim

soc-sim: rtl-fixtures soc-spec npu-wrapper-spec
	mkdir -p build/soc
	iverilog -g2012 -I build/rtl_fixture -I build/soc -I build/npu_wrapper -I hw/npu_wrapper/rtl -o build/soc/soc_tb \
		hw/npu_core/rtl/matrix/matmul_array.sv \
		hw/npu_core/rtl/matrix/accumulator_file.sv \
		hw/npu_core/rtl/vector/vector_engine.sv \
		hw/npu_core/rtl/reduction/reduction_engine.sv \
		hw/npu_core/rtl/sfu/sfu_lut.sv \
		hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv \
		hw/npu_core/rtl/npu_v0_compute_cluster.sv \
		hw/npu_core/rtl/memory/npu_v0_data_mover.sv \
		hw/npu_core/rtl/npu_v0_core_system.sv \
		hw/npu_wrapper/rtl/npu_v0_wrapper.sv \
		hw/soc/rtl/bus/simple_bus.sv \
		hw/soc/rtl/dma/soc_dma.sv \
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
		hw/npu_core/rtl/matrix/matmul_array.sv \
		hw/npu_core/rtl/matrix/accumulator_file.sv \
		hw/npu_core/rtl/vector/vector_engine.sv \
		hw/npu_core/rtl/reduction/reduction_engine.sv \
		hw/npu_core/rtl/sfu/sfu_lut.sv \
		hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv \
		hw/npu_core/rtl/npu_v0_compute_cluster.sv \
		hw/npu_core/rtl/memory/npu_v0_data_mover.sv \
		hw/npu_core/rtl/npu_v0_core_system.sv \
		hw/npu_wrapper/rtl/npu_v0_wrapper.sv \
		hw/soc/rtl/bus/simple_bus.sv \
		hw/soc/rtl/dma/soc_dma.sv \
		hw/soc/rtl/mem/boot_rom.sv \
		hw/soc/rtl/mem/simple_sram.sv \
		hw/soc/rtl/debug/test_status.sv \
		hw/soc/rtl/soc_cpu_top.sv \
		hw/soc/tb/soc_cpu_tb.sv
	vvp build/soc/soc_cpu_tb

cpu-soc-quick:
	$(MAKE) cpu-soc-sim WORKLOAD_PROFILE=quick

cpu-soc-transformer:
	$(MAKE) cpu-soc-sim WORKLOAD_PROFILE=transformer

cpu-soc-cnn-full:
	$(MAKE) cpu-soc-sim WORKLOAD_PROFILE=cnn-full

cpu-soc-all:
	$(MAKE) cpu-soc-sim WORKLOAD_PROFILE=all

perf-report: firmware-smoke
	mkdir -p build/soc build/ppa/data
	iverilog -g2012 -I build/rtl_fixture -I build/soc -I build/npu_wrapper -I hw/npu_wrapper/rtl -o build/soc/soc_cpu_tb \
		hw/soc/cpu/third_party/picorv32/picorv32.v \
		hw/soc/cpu/rtl/picorv32_native_cpu.sv \
		hw/npu_core/rtl/matrix/matmul_array.sv \
		hw/npu_core/rtl/matrix/accumulator_file.sv \
		hw/npu_core/rtl/vector/vector_engine.sv \
		hw/npu_core/rtl/reduction/reduction_engine.sv \
		hw/npu_core/rtl/sfu/sfu_lut.sv \
		hw/npu_core/rtl/scheduler/npu_v0_uop_scheduler.sv \
		hw/npu_core/rtl/npu_v0_compute_cluster.sv \
		hw/npu_core/rtl/memory/npu_v0_data_mover.sv \
		hw/npu_core/rtl/npu_v0_core_system.sv \
		hw/npu_wrapper/rtl/npu_v0_wrapper.sv \
		hw/soc/rtl/bus/simple_bus.sv \
		hw/soc/rtl/dma/soc_dma.sv \
		hw/soc/rtl/mem/boot_rom.sv \
		hw/soc/rtl/mem/simple_sram.sv \
		hw/soc/rtl/debug/test_status.sv \
		hw/soc/rtl/soc_cpu_top.sv \
		hw/soc/tb/soc_cpu_tb.sv
	vvp build/soc/soc_cpu_tb > build/ppa/data/cpu_soc_perf.log
	python sw/tools/perf/report.py --log build/ppa/data/cpu_soc_perf.log --workload-manifest build/ppa/data/workload_manifest.json --arch-config $(ARCH) --soc-config $(SOC) --json-out build/ppa/data/perf.json

perf-l0-quick:
	$(MAKE) perf-report WORKLOAD_PROFILE=quick

perf-l0-transformer:
	$(MAKE) perf-report WORKLOAD_PROFILE=transformer

perf-l0-cnn-full:
	$(MAKE) perf-report WORKLOAD_PROFILE=cnn-full

perf-l0-all:
	$(MAKE) perf-report WORKLOAD_PROFILE=all

ppa-l0-from-perf:
	rm -rf build/perf build/ppa/proxy build/ppa/report
	mkdir -p build/ppa
	PYTHONPATH=$(PYTHONPATH) python -m ppa.schema_check --json $(PPA_BASELINE)
	PYTHONPATH=$(PYTHONPATH) python -m ppa.report \
		--perf-json build/ppa/data/perf.json \
		--area-config $(PPA_AREA_PROXY) \
		--energy-config $(PPA_ENERGY_PROXY) \
		--baseline-json $(PPA_BASELINE) \
		--json-out build/ppa/ppa.json \
		--html-out build/ppa/ppa_overview.html
	PYTHONPATH=$(PYTHONPATH) python -m ppa.schema_check --json build/ppa/ppa.json

ppa-l0-report: perf-report
	$(MAKE) ppa-l0-from-perf

validate-ppa-l0:
	PYTHONPATH=$(PYTHONPATH) python -m ppa.schema_check --json build/ppa/ppa.json
