module vector_engine #(
    parameter int LANES = 8,
    parameter int DATA_WIDTH = 32,
    parameter int OP_VEC_ADD = 0,
    parameter int OP_VEC_SUB = 1,
    parameter int OP_VEC_MUL = 2,
    parameter int OP_VEC_SCALE = 3,
    parameter int OP_VEC_REQUANT = 4,
    parameter int OP_VEC_CLAMP = 5,
    parameter int OP_VEC_SCALE_FIXED = 6
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
    input  logic signed [DATA_WIDTH-1:0] invalid_value,
    output logic done,
    output logic active,
    output logic signed [(LANES*DATA_WIDTH)-1:0] y_flat
);
    localparam logic [2:0] VEC_ADD = OP_VEC_ADD[2:0];
    localparam logic [2:0] VEC_SUB = OP_VEC_SUB[2:0];
    localparam logic [2:0] VEC_MUL = OP_VEC_MUL[2:0];
    localparam logic [2:0] VEC_SCALE = OP_VEC_SCALE[2:0];
    localparam logic [2:0] VEC_REQUANT = OP_VEC_REQUANT[2:0];
    localparam logic [2:0] VEC_CLAMP = OP_VEC_CLAMP[2:0];
    localparam logic [2:0] VEC_SCALE_FIXED = OP_VEC_SCALE_FIXED[2:0];

    integer lane;
    logic signed [DATA_WIDTH-1:0] a_value;
    logic signed [DATA_WIDTH-1:0] b_value;
    logic signed [(DATA_WIDTH*2)-1:0] wide_value;
    logic signed [(DATA_WIDTH*2)-1:0] shifted_wide;
    logic [(DATA_WIDTH*2)-1:0] magnitude;
    logic [(DATA_WIDTH*2)-1:0] rounding_offset;
    logic signed [(DATA_WIDTH*2)-1:0] clamp_low_wide;
    logic signed [(DATA_WIDTH*2)-1:0] clamp_high_wide;
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
                            VEC_SCALE_FIXED: begin
                                wide_value = a_value * scalar;
                                rounding_offset = (shift == 0) ? '0 :
                                    ({{((DATA_WIDTH*2)-1){1'b0}}, 1'b1} << (shift - 1'b1));
                                if (shift == 0) begin
                                    shifted_wide = wide_value;
                                end else if (wide_value >= 0) begin
                                    shifted_wide = $signed(wide_value + rounding_offset) >>> shift;
                                end else begin
                                    magnitude = -wide_value;
                                    shifted_wide = -$signed((magnitude + rounding_offset) >> shift);
                                end
                                clamp_low_wide = {{DATA_WIDTH{clamp_low[DATA_WIDTH-1]}}, clamp_low};
                                clamp_high_wide = {{DATA_WIDTH{clamp_high[DATA_WIDTH-1]}}, clamp_high};
                                if (shifted_wide < clamp_low_wide) result_value = clamp_low;
                                else if (shifted_wide > clamp_high_wide) result_value = clamp_high;
                                else result_value = shifted_wide[DATA_WIDTH-1:0];
                            end
                            default: result_value = '0;
                        endcase
                    end else begin
                        result_value = invalid_value;
                    end
                    y_flat[(lane * DATA_WIDTH) +: DATA_WIDTH] <= result_value;
                end
                done <= 1'b1;
            end
        end
    end
endmodule
