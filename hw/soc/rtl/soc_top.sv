module soc_top (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        cpu_req,
    input  logic        cpu_we,
    input  logic [31:0] cpu_addr,
    input  logic [31:0] cpu_wdata,
    output logic [31:0] cpu_rdata,
    output logic        cpu_ready,

    output logic [31:0] sim_status
);
    `include "soc_v0_addr.svh"
    logic        rom_req;
    logic        rom_we;
    logic [31:0] rom_addr;
    logic [31:0] rom_wdata;
    logic [31:0] rom_rdata;
    logic        rom_ready;

    logic        sram_req;
    logic [SOC_SRAM_CPU_LANES-1:0] sram_we;
    logic [31:0] sram_addr;
    logic [SOC_SRAM_CPU_DATA_WIDTH_BITS-1:0] sram_wdata;
    logic [SOC_SRAM_CPU_DATA_WIDTH_BITS-1:0] sram_rdata;
    logic        sram_ready;

    logic        npu_wrapper_req;
    logic        npu_wrapper_we;
    logic [11:0] npu_wrapper_addr;
    logic [31:0] npu_wrapper_wdata;
    logic [31:0] npu_wrapper_rdata;
    logic        npu_wrapper_ready;

    logic        dma_req;
    logic        dma_we;
    logic [11:0] dma_addr;
    logic [31:0] dma_wdata;
    logic [31:0] dma_rdata;
    logic        dma_ready;

    logic        test_req;
    logic        test_we;
    logic [3:0]  test_addr;
    logic [31:0] test_wdata;
    logic [31:0] test_rdata;
    logic        test_ready;

    simple_bus #(
        .SRAM_CPU_LANES(SOC_SRAM_CPU_LANES)
    ) u_bus (
        .m_req(cpu_req),
        .m_we(cpu_we),
        .m_addr(cpu_addr),
        .m_wdata(cpu_wdata),
        .m_rdata(cpu_rdata),
        .m_ready(cpu_ready),
        .rom_req(rom_req),
        .rom_we(rom_we),
        .rom_addr(rom_addr),
        .rom_wdata(rom_wdata),
        .rom_rdata(rom_rdata),
        .rom_ready(rom_ready),
        .sram_req(sram_req),
        .sram_we(sram_we),
        .sram_addr(sram_addr),
        .sram_wdata(sram_wdata),
        .sram_rdata(sram_rdata),
        .sram_ready(sram_ready),
        .npu_wrapper_req(npu_wrapper_req),
        .npu_wrapper_we(npu_wrapper_we),
        .npu_wrapper_addr(npu_wrapper_addr),
        .npu_wrapper_wdata(npu_wrapper_wdata),
        .npu_wrapper_rdata(npu_wrapper_rdata),
        .npu_wrapper_ready(npu_wrapper_ready),
        .dma_req(dma_req),
        .dma_we(dma_we),
        .dma_addr(dma_addr),
        .dma_wdata(dma_wdata),
        .dma_rdata(dma_rdata),
        .dma_ready(dma_ready),
        .test_req(test_req),
        .test_we(test_we),
        .test_addr(test_addr),
        .test_wdata(test_wdata),
        .test_rdata(test_rdata),
        .test_ready(test_ready)
    );

    boot_rom #(
        .DMA_LANES(SOC_SRAM_CPU_LANES)
    ) u_boot_rom (
        .clk(clk),
        .req(rom_req),
        .we(rom_we),
        .addr(rom_addr),
        .wdata(rom_wdata),
        .rdata(rom_rdata),
        .ready(rom_ready),
        .dma_req(1'b0),
        .dma_addr(32'h0000_0000),
        .dma_rdata(),
        .dma_ready()
    );

    simple_sram #(
        .CPU_LANES(SOC_SRAM_CPU_LANES),
        .NPU_LANES(SOC_NPU_SRAM_LANES)
    ) u_sram (
        .clk(clk),
        .rst_n(rst_n),
        .req(sram_req),
        .we(sram_we),
        .addr(sram_addr),
        .wdata(sram_wdata),
        .rdata(sram_rdata),
        .ready(sram_ready),
        .npu_req(1'b0),
        .npu_we('0),
        .npu_addr(32'h0000_0000),
        .npu_wdata('0),
        .npu_rdata(),
        .npu_ready()
    );

    npu_v0_wrapper #(
        .CORE_HOST_LANES(SOC_NPU_CORE_HOST_LANES)
    ) u_npu_wrapper (
        .clk(clk),
        .rst_n(rst_n),
        .bus_req(npu_wrapper_req),
        .bus_we(npu_wrapper_we),
        .bus_addr(npu_wrapper_addr),
        .bus_wdata(npu_wrapper_wdata),
        .bus_rdata(npu_wrapper_rdata),
        .bus_ready(npu_wrapper_ready),
        .sram_req(),
        .sram_we(),
        .sram_addr(),
        .sram_wdata(),
        .sram_rdata('0),
        .sram_ready(1'b0)
    );

    assign dma_rdata = 32'h0000_0000;
    assign dma_ready = dma_req;

    test_status u_test_status (
        .clk(clk),
        .rst_n(rst_n),
        .req(test_req),
        .we(test_we),
        .addr(test_addr),
        .wdata(test_wdata),
        .rdata(test_rdata),
        .ready(test_ready),
        .status(sim_status)
    );
endmodule
