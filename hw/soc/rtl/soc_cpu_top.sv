module soc_cpu_top (
    input  logic        clk,
    input  logic        rst_n,
    output logic [31:0] sim_status,
    output logic        cpu_trap
);
    `include "soc_v0_addr.svh"
    logic        cpu_req;
    logic        cpu_we;
    logic [31:0] cpu_addr;
    logic [31:0] cpu_wdata;
    logic [31:0] cpu_rdata;
    logic        cpu_ready;

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
    logic        dma_rom_req;
    logic [31:0] dma_rom_addr;
    logic [SOC_SRAM_CPU_DATA_WIDTH_BITS-1:0] dma_rom_rdata;
    logic        dma_rom_ready;
    logic        dma_sram_req;
    logic [SOC_SRAM_CPU_LANES-1:0] dma_sram_we;
    logic [31:0] dma_sram_addr;
    logic [SOC_SRAM_CPU_DATA_WIDTH_BITS-1:0] dma_sram_wdata;
    logic        dma_sram_ready;

    logic        npu_sram_req;
    logic [SOC_NPU_SRAM_LANES-1:0] npu_sram_we;
    logic [31:0] npu_sram_addr;
    logic [(SOC_NPU_SRAM_LANES*32)-1:0] npu_sram_wdata;
    logic [(SOC_NPU_SRAM_LANES*32)-1:0] npu_sram_rdata;
    logic        npu_sram_ready;

    logic        test_req;
    logic        test_we;
    logic [3:0]  test_addr;
    logic [31:0] test_wdata;
    logic [31:0] test_rdata;
    logic        test_ready;

    logic        sram_req_mux;
    logic [SOC_SRAM_CPU_LANES-1:0] sram_we_mux;
    logic [31:0] sram_addr_mux;
    logic [SOC_SRAM_CPU_DATA_WIDTH_BITS-1:0] sram_wdata_mux;

    picorv32_native_cpu u_cpu (
        .clk(clk),
        .rst_n(rst_n),
        .bus_req(cpu_req),
        .bus_we(cpu_we),
        .bus_addr(cpu_addr),
        .bus_wdata(cpu_wdata),
        .bus_rdata(cpu_rdata),
        .bus_ready(cpu_ready),
        .trap(cpu_trap)
    );

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
        .WORDS(SOC_BOOT_ROM_SIZE_BYTES / 4),
        .INIT_HEX("build/firmware/soc_cpu_smoke.hex"),
        .DMA_LANES(SOC_SRAM_CPU_LANES)
    ) u_boot_rom (
        .clk(clk),
        .req(rom_req),
        .we(rom_we),
        .addr(rom_addr),
        .wdata(rom_wdata),
        .rdata(rom_rdata),
        .ready(rom_ready),
        .dma_req(dma_rom_req),
        .dma_addr(dma_rom_addr),
        .dma_rdata(dma_rom_rdata),
        .dma_ready(dma_rom_ready)
    );

    assign sram_req_mux = dma_sram_req ? dma_sram_req : sram_req;
    assign sram_we_mux = dma_sram_req ? dma_sram_we : sram_we;
    assign sram_addr_mux = dma_sram_req ? dma_sram_addr : sram_addr;
    assign sram_wdata_mux = dma_sram_req ? dma_sram_wdata : sram_wdata;
    assign dma_sram_ready = dma_sram_req && sram_ready;

    simple_sram #(
        .WORDS(SOC_SRAM_SIZE_BYTES / 4),
        .CPU_LANES(SOC_SRAM_CPU_LANES),
        .NPU_LANES(SOC_NPU_SRAM_LANES)
    ) u_sram (
        .clk(clk),
        .rst_n(rst_n),
        .req(sram_req_mux),
        .we(sram_we_mux),
        .addr(sram_addr_mux),
        .wdata(sram_wdata_mux),
        .rdata(sram_rdata),
        .ready(sram_ready),
        .npu_req(npu_sram_req),
        .npu_we(npu_sram_we),
        .npu_addr(npu_sram_addr - SOC_SRAM_BASE),
        .npu_wdata(npu_sram_wdata),
        .npu_rdata(npu_sram_rdata),
        .npu_ready(npu_sram_ready)
    );

    soc_dma #(
        .LANES(SOC_SRAM_CPU_LANES)
    ) u_dma (
        .clk(clk),
        .rst_n(rst_n),
        .bus_req(dma_req),
        .bus_we(dma_we),
        .bus_addr(dma_addr),
        .bus_wdata(dma_wdata),
        .bus_rdata(dma_rdata),
        .bus_ready(dma_ready),
        .rom_req(dma_rom_req),
        .rom_addr(dma_rom_addr),
        .rom_rdata(dma_rom_rdata),
        .rom_ready(dma_rom_ready),
        .sram_req(dma_sram_req),
        .sram_we(dma_sram_we),
        .sram_addr(dma_sram_addr),
        .sram_wdata(dma_sram_wdata),
        .sram_ready(dma_sram_ready)
    );

    npu_v0_opsched #(
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
        .sram_req(npu_sram_req),
        .sram_we(npu_sram_we),
        .sram_addr(npu_sram_addr),
        .sram_wdata(npu_sram_wdata),
        .sram_rdata(npu_sram_rdata),
        .sram_ready(npu_sram_ready)
    );

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
