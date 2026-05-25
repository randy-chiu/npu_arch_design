NPU_REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../../../..)

export DESIGN_NAME = npu_v0_top
export PLATFORM = sky130hd

export VERILOG_FILES = \
	$(NPU_REPO_ROOT)/hw/npu_core/rtl/matmul_array.sv \
	$(NPU_REPO_ROOT)/hw/npu_core/rtl/npu_v0_top.sv

export VERILOG_INCLUDE_DIRS = \
	$(NPU_REPO_ROOT)/build/rtl_fixture

export SDC_FILE = $(NPU_REPO_ROOT)/flows/asic/openroad/tops/npu_core/constraint.sdc

# Initial floorplan assumptions for repeatable architecture comparison.
export CORE_UTILIZATION = 35
export CORE_ASPECT_RATIO = 1
export CORE_MARGIN = 2
export PLACE_DENSITY = 0.60
