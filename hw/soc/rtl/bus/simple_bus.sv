module simple_bus #(
    parameter int SRAM_CPU_LANES = 1
) (
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
    output logic [SRAM_CPU_LANES-1:0] sram_we,
    output logic [31:0] sram_addr,
    output logic [(SRAM_CPU_LANES*32)-1:0] sram_wdata,
    input  logic [(SRAM_CPU_LANES*32)-1:0] sram_rdata,
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
    logic [31:0] sram_local_addr;
    logic [31:0] sram_lane;
    integer sram_pack_lane;

    assign sel_rom = ((m_addr & SOC_BOOT_ROM_MASK) == SOC_BOOT_ROM_BASE);
    assign sel_sram = ((m_addr & SOC_SRAM_MASK) == SOC_SRAM_BASE);
    assign sel_npu_wrapper = ((m_addr & SOC_NPU_WRAPPER_MASK) == SOC_NPU_WRAPPER_BASE);
    assign sel_test = ((m_addr & SOC_TEST_STATUS_MASK) == SOC_TEST_STATUS_BASE);

    assign rom_req = m_req && sel_rom;
    assign rom_we = m_we;
    assign rom_addr = m_addr - SOC_BOOT_ROM_BASE;
    assign rom_wdata = m_wdata;

    assign sram_local_addr = m_addr - SOC_SRAM_BASE;
    assign sram_lane = (sram_local_addr >> 2) % SRAM_CPU_LANES;
    assign sram_req = m_req && sel_sram;
    assign sram_addr = sram_local_addr - (sram_lane << 2);

    assign npu_wrapper_req = m_req && sel_npu_wrapper;
    assign npu_wrapper_we = m_we;
    assign npu_wrapper_addr = m_addr[11:0];
    assign npu_wrapper_wdata = m_wdata;

    assign test_req = m_req && sel_test;
    assign test_we = m_we;
    assign test_addr = m_addr[3:0];
    assign test_wdata = m_wdata;

    always_comb begin
        sram_we = '0;
        sram_wdata = '0;
        for (sram_pack_lane = 0; sram_pack_lane < SRAM_CPU_LANES; sram_pack_lane = sram_pack_lane + 1) begin
            if (sram_lane == sram_pack_lane[31:0]) begin
                sram_we[sram_pack_lane] = m_we && sel_sram;
                sram_wdata[(sram_pack_lane * 32) +: 32] = m_wdata;
            end
        end
    end

    always_comb begin
        m_ready = m_req;
        m_rdata = 32'h0000_0000;
        if (sel_rom) begin
            m_ready = rom_ready;
            m_rdata = rom_rdata;
        end else if (sel_sram) begin
            m_ready = sram_ready;
            m_rdata = sram_rdata[(sram_lane * 32) +: 32];
        end else if (sel_npu_wrapper) begin
            m_ready = npu_wrapper_ready;
            m_rdata = npu_wrapper_rdata;
        end else if (sel_test) begin
            m_ready = test_ready;
            m_rdata = test_rdata;
        end
    end
endmodule
