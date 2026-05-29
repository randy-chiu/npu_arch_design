module sfu_lut #(
    parameter int DATA_WIDTH = 32
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
    localparam logic [1:0] SFU_EXP = 2'd0;
    localparam logic [1:0] SFU_RECIP = 2'd1;
    localparam logic [1:0] SFU_RSQRT = 2'd2;

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
            else if (value < -256) clamped = -256;
            else clamped = value;
            segment = (-clamped + 16) / 32;
            case (segment)
                0: exp_q15 = 16'd32767;
                1: exp_q15 = 16'd12055;
                2: exp_q15 = 16'd4435;
                3: exp_q15 = 16'd1632;
                4: exp_q15 = 16'd600;
                5: exp_q15 = 16'd221;
                6: exp_q15 = 16'd81;
                7: exp_q15 = 16'd30;
                default: exp_q15 = 16'd11;
            endcase
        end
    endfunction

    function automatic [31:0] recip_q24(input logic [31:0] value);
        begin
            recip_q24 = (value == 0) ? 32'h0 : (32'd16777216 / value);
        end
    endfunction

    function automatic [31:0] rsqrt_q24(input logic [31:0] value);
        logic [31:0] root;
        begin
            root = isqrt(value);
            rsqrt_q24 = (root == 0) ? 32'h0 : (32'd16777216 / root);
        end
    endfunction

    function automatic [31:0] isqrt(input logic [31:0] value);
        integer bit_idx;
        logic [31:0] candidate;
        logic [31:0] root;
        begin
            root = 32'h0;
            for (bit_idx = 15; bit_idx >= 0; bit_idx = bit_idx - 1) begin
                candidate = root | (32'h1 << bit_idx);
                if (candidate * candidate <= value) begin
                    root = candidate;
                end
            end
            isqrt = root;
        end
    endfunction
endmodule
