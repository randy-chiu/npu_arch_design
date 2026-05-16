module test_status (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        req,
    input  logic        we,
    input  logic [3:0]  addr,
    input  logic [31:0] wdata,
    output logic [31:0] rdata,
    output logic        ready,
    output logic [31:0] status
);
    assign ready = req;
    assign rdata = (req && !we && addr == 4'h0) ? status : 32'h0000_0000;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            status <= 32'h0000_0000;
        end else begin
            if (req && we && addr == 4'h0) begin
                status <= wdata;
            end
        end
    end
endmodule
