module reduction_engine #(
    parameter int MAX_LEN = 128,
    parameter int DATA_WIDTH = 32,
    parameter int LEN_WIDTH = 8,
    parameter int RESULT_WIDTH = 64
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic [1:0] op,
    input  logic [LEN_WIDTH-1:0] length,
    input  logic signed [(MAX_LEN*DATA_WIDTH)-1:0] x_flat,
    output logic done,
    output logic active,
    output logic signed [RESULT_WIDTH-1:0] result
);
    localparam logic [1:0] REDUCE_MAX = 2'd0;
    localparam logic [1:0] REDUCE_SUM = 2'd1;
    localparam logic [1:0] REDUCE_SUMSQ = 2'd2;

    integer idx;
    logic signed [DATA_WIDTH-1:0] x_value;
    logic signed [RESULT_WIDTH-1:0] accum;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            active <= 1'b0;
            result <= '0;
        end else begin
            done <= 1'b0;
            active <= start;
            if (start) begin
                accum = '0;
                if (op == REDUCE_MAX && length != '0) begin
                    accum = {{(RESULT_WIDTH-DATA_WIDTH){x_flat[DATA_WIDTH-1]}}, x_flat[0 +: DATA_WIDTH]};
                end
                for (idx = 0; idx < MAX_LEN; idx = idx + 1) begin
                    if (idx < length) begin
                        x_value = x_flat[(idx * DATA_WIDTH) +: DATA_WIDTH];
                        case (op)
                            REDUCE_MAX: begin
                                if ($signed(x_value) > $signed(accum)) begin
                                    accum = {{(RESULT_WIDTH-DATA_WIDTH){x_value[DATA_WIDTH-1]}}, x_value};
                                end
                            end
                            REDUCE_SUM: begin
                                accum = accum + {{(RESULT_WIDTH-DATA_WIDTH){x_value[DATA_WIDTH-1]}}, x_value};
                            end
                            REDUCE_SUMSQ: begin
                                accum = accum + (x_value * x_value);
                            end
                            default: accum = '0;
                        endcase
                    end
                end
                result <= accum;
                done <= 1'b1;
            end
        end
    end
endmodule
