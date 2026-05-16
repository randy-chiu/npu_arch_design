module boot_rom #(
    parameter int WORDS = 8192,
    parameter string INIT_HEX = ""
) (
    input  logic        clk,
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

    initial begin
        if (INIT_HEX != "") begin
            $readmemh(INIT_HEX, mem);
        end
    end

endmodule
