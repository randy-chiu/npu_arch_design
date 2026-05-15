.PHONY: test demo validate-arch rtl-fixtures rtl-sim

PYTHONPATH := src
ARCH := arch/configs/npu_v0.jsonc

validate-arch:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli validate-arch --arch $(ARCH)

demo:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli demo --arch $(ARCH)

rtl-fixtures:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli emit-rtl-fixtures --arch $(ARCH) --out-dir build/rtl_fixture

test:
	PYTHONPATH=$(PYTHONPATH) python -m unittest discover -s tests -v

refresh-references:
	PYTHONPATH=$(PYTHONPATH) python scripts/refresh_references.py --output references/discovered_references.md

rtl-sim: rtl-fixtures
	mkdir -p build
	iverilog -g2012 -I build/rtl_fixture -o build/npu_v0_tb hw/rtl/npu_v0_top.sv hw/tb/npu_v0_tb.sv
	vvp build/npu_v0_tb
