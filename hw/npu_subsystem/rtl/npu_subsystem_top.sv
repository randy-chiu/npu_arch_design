module npu_subsystem_top #(
    parameter int CORE_HOST_LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        ctrl_req,
    input  logic        ctrl_we,
    input  logic [11:0] ctrl_addr,
    input  logic [31:0] ctrl_wdata,
    output logic [31:0] ctrl_rdata,
    output logic        ctrl_ready,

    output logic        mem_req,
    output logic [CORE_HOST_LANES-1:0] mem_we,
    output logic [31:0] mem_addr,
    output logic [(CORE_HOST_LANES*32)-1:0] mem_wdata,
    input  logic [(CORE_HOST_LANES*32)-1:0] mem_rdata,
    input  logic        mem_ready
);
    npu_v0_opsched #(
        .CORE_HOST_LANES(CORE_HOST_LANES)
    ) u_opsched (
        .clk(clk),
        .rst_n(rst_n),
        .bus_req(ctrl_req),
        .bus_we(ctrl_we),
        .bus_addr(ctrl_addr),
        .bus_wdata(ctrl_wdata),
        .bus_rdata(ctrl_rdata),
        .bus_ready(ctrl_ready),
        .sram_req(mem_req),
        .sram_we(mem_we),
        .sram_addr(mem_addr),
        .sram_wdata(mem_wdata),
        .sram_rdata(mem_rdata),
        .sram_ready(mem_ready)
    );
endmodule
