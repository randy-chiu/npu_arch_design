module simple_bus (
    input  logic        m_req,
    input  logic        m_we,
    input  logic [31:0] m_addr,
    input  logic [31:0] m_wdata,
    output logic [31:0] m_rdata,
    output logic        m_ready,

    output logic        rom_req,
    output logic        rom_we,
    output logic [31:0] rom_addr,
    output logic [31:0] rom_wdata,
    input  logic [31:0] rom_rdata,
    input  logic        rom_ready,

    output logic        sram_req,
    output logic        sram_we,
    output logic [31:0] sram_addr,
    output logic [31:0] sram_wdata,
    input  logic [31:0] sram_rdata,
    input  logic        sram_ready,

    output logic        npu_wrapper_req,
    output logic        npu_wrapper_we,
    output logic [11:0] npu_wrapper_addr,
    output logic [31:0] npu_wrapper_wdata,
    input  logic [31:0] npu_wrapper_rdata,
    input  logic        npu_wrapper_ready,

    output logic        test_req,
    output logic        test_we,
    output logic [3:0]  test_addr,
    output logic [31:0] test_wdata,
    input  logic [31:0] test_rdata,
    input  logic        test_ready
);
    `include "soc_v0_addr.svh"

    logic sel_rom;
    logic sel_sram;
    logic sel_npu_wrapper;
    logic sel_test;

    assign sel_rom = ((m_addr & SOC_BOOT_ROM_MASK) == SOC_BOOT_ROM_BASE);
    assign sel_sram = ((m_addr & SOC_SRAM_MASK) == SOC_SRAM_BASE);
    assign sel_npu_wrapper = ((m_addr & SOC_NPU_WRAPPER_MASK) == SOC_NPU_WRAPPER_BASE);
    assign sel_test = ((m_addr & SOC_TEST_STATUS_MASK) == SOC_TEST_STATUS_BASE);

    assign rom_req = m_req && sel_rom;
    assign rom_we = m_we;
    assign rom_addr = m_addr - SOC_BOOT_ROM_BASE;
    assign rom_wdata = m_wdata;

    assign sram_req = m_req && sel_sram;
    assign sram_we = m_we;
    assign sram_addr = m_addr - SOC_SRAM_BASE;
    assign sram_wdata = m_wdata;

    assign npu_wrapper_req = m_req && sel_npu_wrapper;
    assign npu_wrapper_we = m_we;
    assign npu_wrapper_addr = m_addr[11:0];
    assign npu_wrapper_wdata = m_wdata;

    assign test_req = m_req && sel_test;
    assign test_we = m_we;
    assign test_addr = m_addr[3:0];
    assign test_wdata = m_wdata;

    always_comb begin
        m_ready = m_req;
        m_rdata = 32'h0000_0000;
        if (sel_rom) begin
            m_ready = rom_ready;
            m_rdata = rom_rdata;
        end else if (sel_sram) begin
            m_ready = sram_ready;
            m_rdata = sram_rdata;
        end else if (sel_npu_wrapper) begin
            m_ready = npu_wrapper_ready;
            m_rdata = npu_wrapper_rdata;
        end else if (sel_test) begin
            m_ready = test_ready;
            m_rdata = test_rdata;
        end
    end
endmodule
