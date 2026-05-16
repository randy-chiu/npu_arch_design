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
    logic        sram_we;
    logic [31:0] sram_addr;
    logic [31:0] sram_wdata;
    logic [31:0] sram_rdata;
    logic        sram_ready;

    logic        npu_wrapper_req;
    logic        npu_wrapper_we;
    logic [11:0] npu_wrapper_addr;
    logic [31:0] npu_wrapper_wdata;
    logic [31:0] npu_wrapper_rdata;
    logic        npu_wrapper_ready;

    logic        npu_sram_req;
    logic        npu_sram_we;
    logic [31:0] npu_sram_addr;
    logic [31:0] npu_sram_wdata;
    logic [31:0] npu_sram_rdata;
    logic        npu_sram_ready;

    logic        test_req;
    logic        test_we;
    logic [3:0]  test_addr;
    logic [31:0] test_wdata;
    logic [31:0] test_rdata;
    logic        test_ready;

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

    simple_bus u_bus (
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
        .test_req(test_req),
        .test_we(test_we),
        .test_addr(test_addr),
        .test_wdata(test_wdata),
        .test_rdata(test_rdata),
        .test_ready(test_ready)
    );

    boot_rom #(
        .INIT_HEX("build/firmware/soc_cpu_smoke.hex")
    ) u_boot_rom (
        .clk(clk),
        .req(rom_req),
        .we(rom_we),
        .addr(rom_addr),
        .wdata(rom_wdata),
        .rdata(rom_rdata),
        .ready(rom_ready)
    );

    simple_sram u_sram (
        .clk(clk),
        .rst_n(rst_n),
        .req(sram_req),
        .we(sram_we),
        .addr(sram_addr),
        .wdata(sram_wdata),
        .rdata(sram_rdata),
        .ready(sram_ready),
        .npu_req(npu_sram_req),
        .npu_we(npu_sram_we),
        .npu_addr(npu_sram_addr - SOC_SRAM_BASE),
        .npu_wdata(npu_sram_wdata),
        .npu_rdata(npu_sram_rdata),
        .npu_ready(npu_sram_ready)
    );

    npu_v0_opsched u_npu_wrapper (
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
