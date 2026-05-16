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
    output logic        ready
);
    logic [31:0] mem [0:WORDS-1];
    logic [$clog2(WORDS)-1:0] word_addr;

    assign word_addr = addr[$clog2(WORDS)+1:2];
    assign ready = req;
    assign rdata = (req && !we) ? mem[word_addr] : 32'h0000_0000;

    always_ff @(posedge clk) begin
        if (req && we) begin
            mem[word_addr] <= wdata;
        end
    end
endmodule
