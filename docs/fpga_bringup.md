# FPGA Bring-Up Notes

## Current Status

The current RTL is a real SystemVerilog hardware model for the Phase 0 NPU
micro-op subset. It passes the local Icarus Verilog smoke test:

```text
make rtl-sim
```

It is not yet directly ready to burn onto an FPGA board as a complete bitstream.

## Important Clarification

`iverilog` compiles RTL for simulation. It does not synthesize FPGA hardware and
does not generate a vendor bitstream.

To run on an FPGA, the RTL must go through the FPGA vendor flow, for example:

- Xilinx Vivado for Xilinx/AMD boards.
- Intel Quartus for Intel FPGA boards.
- Lattice Radiant or open-source flows for supported Lattice devices.

The FPGA flow must perform:

1. RTL synthesis.
2. Place and route.
3. Timing analysis.
4. Bitstream generation.
5. Board programming.

## Expected Future Flow

The intended end-to-end flow is:

```text
graph
  -> compiler
  -> micro-op program file
  -> host/runtime loads input tensors and instruction stream
  -> FPGA wrapper writes NPU memory and instruction memory
  -> start signal launches NPU
  -> done signal indicates completion
  -> host/runtime reads outputs
```

Your understanding is directionally correct:

1. The compiler should emit the NPU instruction stream.
2. The RTL should implement the NPU micro-op execution engine.
3. The FPGA design should contain the NPU RTL plus a board-specific wrapper.
4. The host should load tensors and instructions into the mapped memory space.
5. The host should start the NPU and read results after `done`.

The missing piece is that `iverilog` is only step zero for simulation. A real
FPGA deployment needs a synthesis-ready top-level integration.

## What Is Missing Before Board Execution

### 1. Board Wrapper

Current top module:

```text
npu_v0_top
```

It exposes a simple custom host interface:

```text
host_we
host_addr
host_wdata
host_rdata
start
done
```

A real FPGA board usually needs one of:

- AXI-Lite slave interface.
- Wishbone slave interface.
- UART command loader.
- PCIe BAR interface.
- JTAG/debug bridge.
- Board-specific memory-mapped bus.

### 2. Constraints

The project needs board-specific constraint files:

- clock pin
- reset pin
- I/O pins or bus interface pins
- timing constraints

Examples:

- Xilinx `.xdc`
- Intel `.sdc` / `.qsf`

### 3. Program And Tensor Loader

The compiler currently emits JSON micro-ops for the software simulator. The RTL
currently consumes compact 32-bit encoded micro-ops in instruction memory.

Required next step:

- Add an assembler that converts compiler JSON micro-ops into RTL instruction
  words.
- Emit memory initialization files or host-loader commands.

Possible output formats:

- `.mem` file for FPGA BRAM initialization.
- binary file for host runtime.
- text command script for UART/JTAG loader.

### 4. Runtime/Host Tool

A board execution flow needs a host-side tool that can:

- write input tensor memory
- write instruction memory
- assert `start`
- poll `done`
- read output tensor memory
- compare against golden output

The planned minimal SoC path is documented in:

```text
docs/soc_bringup.md
```

That path uses an existing small CPU softcore, a simple memory-mapped bus,
ROM/SRAM, and an `opsched` NPU operator scheduler so bare-metal firmware can
control the NPU through registers instead of a testbench directly toggling
`start`.

### 5. Synthesis Cleanup

The current RTL is simulation-proven, but it still needs synthesis review:

- confirm task usage is accepted by the target FPGA synthesis tool
- confirm arrays infer BRAM/registers as intended
- replace any simulator-friendly constructs if needed
- add reset and initialization strategy suitable for the board

## Recommended Next Milestone

Create `phase0_fpga_min`:

1. Add RTL assembler for the current micro-op encoding.
2. Add generated `.mem` files for matmul and softmax programs.
3. Add the minimal SoC `opsched` path described in `docs/soc_bringup.md`.
4. Add a synthesis-oriented FPGA wrapper with a simple memory-mapped interface.
5. Add a board-specific target directory after the user chooses the FPGA board.
6. Run vendor synthesis and record resource/timing reports.
