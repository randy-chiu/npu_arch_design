module vector_engine #(
    parameter int LANES = 8,
    parameter int DATA_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic [2:0] op,
    input  logic [LANES-1:0] valid_mask,
    input  logic signed [(LANES*DATA_WIDTH)-1:0] a_flat,
    input  logic signed [(LANES*DATA_WIDTH)-1:0] b_flat,
    input  logic signed [DATA_WIDTH-1:0] scalar,
    input  logic signed [DATA_WIDTH-1:0] clamp_low,
    input  logic signed [DATA_WIDTH-1:0] clamp_high,
    input  logic [4:0] shift,
    output logic done,
    output logic active,
    output logic signed [(LANES*DATA_WIDTH)-1:0] y_flat
);
    localparam logic [2:0] VEC_ADD = 3'd0;
    localparam logic [2:0] VEC_SUB = 3'd1;
    localparam logic [2:0] VEC_MUL = 3'd2;
    localparam logic [2:0] VEC_SCALE = 3'd3;
    localparam logic [2:0] VEC_REQUANT = 3'd4;
    localparam logic [2:0] VEC_CLAMP = 3'd5;

    integer lane;
    logic signed [DATA_WIDTH-1:0] a_value;
    logic signed [DATA_WIDTH-1:0] b_value;
    logic signed [(DATA_WIDTH*2)-1:0] wide_value;
    logic signed [DATA_WIDTH-1:0] result_value;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            active <= 1'b0;
            y_flat <= '0;
        end else begin
            done <= 1'b0;
            active <= start;
            if (start) begin
                for (lane = 0; lane < LANES; lane = lane + 1) begin
                    a_value = a_flat[(lane * DATA_WIDTH) +: DATA_WIDTH];
                    b_value = b_flat[(lane * DATA_WIDTH) +: DATA_WIDTH];
                    result_value = '0;
                    if (valid_mask[lane]) begin
                        case (op)
                            VEC_ADD: result_value = a_value + b_value;
                            VEC_SUB: result_value = a_value - b_value;
                            VEC_MUL: begin
                                wide_value = a_value * b_value;
                                result_value = wide_value[DATA_WIDTH-1:0];
                            end
                            VEC_SCALE: begin
                                wide_value = a_value * scalar;
                                result_value = wide_value >>> shift;
                            end
                            VEC_REQUANT: begin
                                result_value = a_value >>> shift;
                                if (result_value < clamp_low) result_value = clamp_low;
                                if (result_value > clamp_high) result_value = clamp_high;
                            end
                            VEC_CLAMP: begin
                                result_value = a_value;
                                if (result_value < clamp_low) result_value = clamp_low;
                                if (result_value > clamp_high) result_value = clamp_high;
                            end
                            default: result_value = '0;
                        endcase
                    end
                    y_flat[(lane * DATA_WIDTH) +: DATA_WIDTH] <= result_value;
                end
                done <= 1'b1;
            end
        end
    end
endmodule
