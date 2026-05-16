module picorv32_native_cpu (
    input  logic        clk,
    input  logic        rst_n,

    output logic        bus_req,
    output logic        bus_we,
    output logic [31:0] bus_addr,
    output logic [31:0] bus_wdata,
    input  logic [31:0] bus_rdata,
    input  logic        bus_ready,

    output logic        trap
);
    logic [3:0] mem_wstrb;

    assign bus_we = |mem_wstrb;

    picorv32 #(
        .ENABLE_COUNTERS(0),
        .ENABLE_COUNTERS64(0),
        .ENABLE_REGS_16_31(1),
        .ENABLE_REGS_DUALPORT(1),
        .TWO_STAGE_SHIFT(0),
        .BARREL_SHIFTER(0),
        .COMPRESSED_ISA(0),
        .ENABLE_MUL(0),
        .ENABLE_FAST_MUL(0),
        .ENABLE_DIV(0),
        .PROGADDR_RESET(32'h0000_0000),
        .STACKADDR(32'h0002_fff0)
    ) u_cpu (
        .clk(clk),
        .resetn(rst_n),
        .trap(trap),
        .mem_valid(bus_req),
        .mem_instr(),
        .mem_ready(bus_ready),
        .mem_addr(bus_addr),
        .mem_wdata(bus_wdata),
        .mem_wstrb(mem_wstrb),
        .mem_rdata(bus_rdata),
        .mem_la_read(),
        .mem_la_write(),
        .mem_la_addr(),
        .mem_la_wdata(),
        .mem_la_wstrb(),
        .pcpi_valid(),
        .pcpi_insn(),
        .pcpi_rs1(),
        .pcpi_rs2(),
        .pcpi_wr(1'b0),
        .pcpi_rd(32'h0000_0000),
        .pcpi_wait(1'b0),
        .pcpi_ready(1'b0),
        .irq(32'h0000_0000),
        .eoi(),
        .trace_valid(),
        .trace_data()
    );
endmodule
