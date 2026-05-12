.PHONY: test demo validate-arch

PYTHONPATH := src
ARCH := arch/configs/npu_v0.jsonc

validate-arch:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli validate-arch --arch $(ARCH)

demo:
	PYTHONPATH=$(PYTHONPATH) python -m npu_phase0.cli demo --arch $(ARCH)

test:
	PYTHONPATH=$(PYTHONPATH) python -m unittest discover -s tests -v

refresh-references:
	PYTHONPATH=$(PYTHONPATH) python scripts/refresh_references.py --output references/discovered_references.md

rtl-sim:
	mkdir -p build
	iverilog -g2012 -o build/npu_v0_tb hw/rtl/npu_v0_top.sv hw/tb/npu_v0_tb.sv
	vvp build/npu_v0_tb
