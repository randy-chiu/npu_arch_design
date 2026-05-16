module simple_sram #(
    parameter int WORDS = 32768
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        req,
    input  logic        we,
    input  logic [31:0] addr,
    input  logic [31:0] wdata,
    output logic [31:0] rdata,
    output logic        ready,

    input  logic        npu_req,
    input  logic        npu_we,
    input  logic [31:0] npu_addr,
    input  logic [31:0] npu_wdata,
    output logic [31:0] npu_rdata,
    output logic        npu_ready
);
    // Writable CPU data memory. In the current smoke test this is mainly used
    // for stack and C locals because static code/test data execute from boot
    // ROM. A later NPU fetch model should place runtime tensor buffers,
    // descriptors, and possibly program streams here for the NPU to read.
    logic [31:0] mem [0:WORDS-1];
    logic [$clog2(WORDS)-1:0] word_addr;
    logic [$clog2(WORDS)-1:0] npu_word_addr;

    assign word_addr = addr[$clog2(WORDS)+1:2];
    assign npu_word_addr = npu_addr[$clog2(WORDS)+1:2];
    assign ready = req;
    assign npu_ready = npu_req;
    assign rdata = (req && !we) ? mem[word_addr] : 32'h0000_0000;
    assign npu_rdata = (npu_req && !npu_we) ? mem[npu_word_addr] : 32'h0000_0000;

    always_ff @(posedge clk) begin
        if (req && we) begin
            mem[word_addr] <= wdata;
        end
        if (npu_req && npu_we) begin
            mem[npu_word_addr] <= npu_wdata;
        end
    end
endmodule
