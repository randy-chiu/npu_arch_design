module simple_sram #(
    parameter int WORDS = 32768,
    parameter int CPU_LANES = 1,
    parameter int NPU_LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        req,
    input  logic [CPU_LANES-1:0] we,
    input  logic [31:0] addr,
    input  logic [(CPU_LANES*32)-1:0] wdata,
    output logic [(CPU_LANES*32)-1:0] rdata,
    output logic        ready,

    input  logic        npu_req,
    input  logic [NPU_LANES-1:0] npu_we,
    input  logic [31:0] npu_addr,
    input  logic [(NPU_LANES*32)-1:0] npu_wdata,
    output logic [(NPU_LANES*32)-1:0] npu_rdata,
    output logic        npu_ready
);
    // Writable CPU data memory. In the current smoke test this is mainly used
    // for stack and C locals because static code/test data execute from boot
    // ROM. A later NPU fetch model should place runtime tensor buffers,
    // descriptors, and possibly program streams here for the NPU to read.
    logic [31:0] mem [0:WORDS-1];
    logic [$clog2(WORDS)-1:0] word_addr;
    logic [$clog2(WORDS)-1:0] npu_word_addr;
    genvar cpu_lane;
    genvar npu_lane;
    integer cpu_write_lane;
    integer npu_write_lane;

    assign word_addr = addr[$clog2(WORDS)+1:2];
    assign npu_word_addr = npu_addr[$clog2(WORDS)+1:2];
    assign ready = req;
    assign npu_ready = npu_req;

    generate
        for (cpu_lane = 0; cpu_lane < CPU_LANES; cpu_lane = cpu_lane + 1) begin : gen_cpu_read_lane
            assign rdata[(cpu_lane * 32) +: 32] =
                (req && we == '0) ? mem[word_addr + cpu_lane] : 32'h0000_0000;
        end
    endgenerate

    generate
        for (npu_lane = 0; npu_lane < NPU_LANES; npu_lane = npu_lane + 1) begin : gen_npu_read_lane
            assign npu_rdata[(npu_lane * 32) +: 32] =
                (npu_req && npu_we == '0) ? mem[npu_word_addr + npu_lane] : 32'h0000_0000;
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (req) begin
            for (cpu_write_lane = 0; cpu_write_lane < CPU_LANES; cpu_write_lane = cpu_write_lane + 1) begin
                if (we[cpu_write_lane]) begin
                    mem[word_addr + cpu_write_lane] <= wdata[(cpu_write_lane * 32) +: 32];
                end
            end
        end
        if (npu_req) begin
            for (npu_write_lane = 0; npu_write_lane < NPU_LANES; npu_write_lane = npu_write_lane + 1) begin
                if (npu_we[npu_write_lane]) begin
                    mem[npu_word_addr + npu_write_lane] <= npu_wdata[(npu_write_lane * 32) +: 32];
                end
            end
        end
    end
endmodule
