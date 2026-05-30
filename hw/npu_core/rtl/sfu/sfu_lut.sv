module sfu_lut #(
    parameter int DATA_WIDTH = 32,
    parameter int EXP_INPUT_SCALE = 32,
    parameter int EXP_LUT_ENTRIES = 257,
    parameter int EXP_OUTPUT_Q = 15,
    parameter int RECIP_OUTPUT_Q = 24,
    parameter int RSQRT_OUTPUT_Q = 24,
    parameter int BRINGUP_EXP_SEG_0 = 32767,
    parameter int BRINGUP_EXP_SEG_1 = 12055,
    parameter int BRINGUP_EXP_SEG_2 = 4435,
    parameter int BRINGUP_EXP_SEG_3 = 1632,
    parameter int BRINGUP_EXP_SEG_4 = 600,
    parameter int BRINGUP_EXP_SEG_5 = 221,
    parameter int BRINGUP_EXP_SEG_6 = 81,
    parameter int BRINGUP_EXP_SEG_7 = 30,
    parameter int BRINGUP_EXP_SEG_8 = 11,
    parameter int OP_SFU_EXP = 0,
    parameter int OP_SFU_RECIP = 1,
    parameter int OP_SFU_RSQRT = 2
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic [1:0] op,
    input  logic signed [DATA_WIDTH-1:0] x,
    output logic done,
    output logic active,
    output logic [DATA_WIDTH-1:0] y
);
    localparam logic [1:0] SFU_EXP = OP_SFU_EXP[1:0];
    localparam logic [1:0] SFU_RECIP = OP_SFU_RECIP[1:0];
    localparam logic [1:0] SFU_RSQRT = OP_SFU_RSQRT[1:0];
    localparam int EXP_CLAMP_MIN = -8 * EXP_INPUT_SCALE;
    localparam int EXP_SEGMENT_WIDTH = EXP_INPUT_SCALE;
    localparam int EXP_SEGMENT_ROUND = EXP_SEGMENT_WIDTH / 2;
    localparam int RSQRT_ROOT_BITS = DATA_WIDTH / 2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            active <= 1'b0;
            y <= '0;
        end else begin
            done <= 1'b0;
            active <= start;
            if (start) begin
                case (op)
                    SFU_EXP: y <= {16'h0, exp_q15(x)};
                    SFU_RECIP: y <= recip_q24(x[31:0]);
                    SFU_RSQRT: y <= rsqrt_q24(x[31:0]);
                    default: y <= '0;
                endcase
                done <= 1'b1;
            end
        end
    end

    function automatic [15:0] exp_q15(input logic signed [DATA_WIDTH-1:0] value);
        logic signed [DATA_WIDTH-1:0] clamped;
        integer segment;
        begin
            if (value > 0) clamped = 0;
            else if (value < EXP_CLAMP_MIN) clamped = EXP_CLAMP_MIN;
            else clamped = value;
            segment = (-clamped + EXP_SEGMENT_ROUND) / EXP_SEGMENT_WIDTH;
            case (segment)
                0: exp_q15 = BRINGUP_EXP_SEG_0[15:0];
                1: exp_q15 = BRINGUP_EXP_SEG_1[15:0];
                2: exp_q15 = BRINGUP_EXP_SEG_2[15:0];
                3: exp_q15 = BRINGUP_EXP_SEG_3[15:0];
                4: exp_q15 = BRINGUP_EXP_SEG_4[15:0];
                5: exp_q15 = BRINGUP_EXP_SEG_5[15:0];
                6: exp_q15 = BRINGUP_EXP_SEG_6[15:0];
                7: exp_q15 = BRINGUP_EXP_SEG_7[15:0];
                default: exp_q15 = BRINGUP_EXP_SEG_8[15:0];
            endcase
        end
    endfunction

    function automatic [31:0] recip_q24(input logic [31:0] value);
        begin
            recip_q24 = (value == 0) ? 32'h0 : ((32'd1 << RECIP_OUTPUT_Q) / value);
        end
    endfunction

    function automatic [31:0] rsqrt_q24(input logic [31:0] value);
        logic [31:0] root;
        begin
            root = isqrt(value);
            rsqrt_q24 = (root == 0) ? 32'h0 : ((32'd1 << RSQRT_OUTPUT_Q) / root);
        end
    endfunction

    function automatic [31:0] isqrt(input logic [31:0] value);
        integer bit_idx;
        logic [31:0] candidate;
        logic [31:0] root;
        begin
            root = 32'h0;
            for (bit_idx = RSQRT_ROOT_BITS - 1; bit_idx >= 0; bit_idx = bit_idx - 1) begin
                candidate = root | (32'h1 << bit_idx);
                if (candidate * candidate <= value) begin
                    root = candidate;
                end
            end
            isqrt = root;
        end
    endfunction
endmodule
