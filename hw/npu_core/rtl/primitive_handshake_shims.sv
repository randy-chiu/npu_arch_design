module vector_engine_handshake #(
    parameter int LANES = 8,
    parameter int DATA_WIDTH = 32,
    parameter int OP_VEC_ADD = 0,
    parameter int OP_VEC_SUB = 1,
    parameter int OP_VEC_MUL = 2,
    parameter int OP_VEC_SCALE = 3,
    parameter int OP_VEC_REQUANT = 4,
    parameter int OP_VEC_CLAMP = 5,
    parameter int OP_VEC_SCALE_FIXED = 6,
    parameter int COUNTER_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic cmd_valid,
    output logic cmd_ready,
    input  logic [2:0] cmd_op,
    input  logic [LANES-1:0] cmd_valid_mask,
    input  logic signed [(LANES*DATA_WIDTH)-1:0] cmd_a_flat,
    input  logic signed [(LANES*DATA_WIDTH)-1:0] cmd_b_flat,
    input  logic signed [DATA_WIDTH-1:0] cmd_scalar,
    input  logic signed [DATA_WIDTH-1:0] cmd_clamp_low,
    input  logic signed [DATA_WIDTH-1:0] cmd_clamp_high,
    input  logic [4:0] cmd_shift,
    output logic rsp_valid,
    input  logic rsp_ready,
    output logic signed [(LANES*DATA_WIDTH)-1:0] rsp_y_flat,
    output logic [COUNTER_WIDTH-1:0] active_cycles,
    output logic [COUNTER_WIDTH-1:0] input_stall_cycles,
    output logic [COUNTER_WIDTH-1:0] output_stall_cycles,
    output logic [COUNTER_WIDTH-1:0] idle_cycles,
    output logic [COUNTER_WIDTH-1:0] accepted_ops,
    output logic [COUNTER_WIDTH-1:0] accepted_lane_ops
);
    logic engine_start;
    logic engine_done;
    logic engine_active;
    logic engine_busy;
    logic [2:0] op;
    logic [LANES-1:0] valid_mask;
    logic signed [(LANES*DATA_WIDTH)-1:0] a_flat;
    logic signed [(LANES*DATA_WIDTH)-1:0] b_flat;
    logic signed [DATA_WIDTH-1:0] scalar;
    logic signed [DATA_WIDTH-1:0] clamp_low;
    logic signed [DATA_WIDTH-1:0] clamp_high;
    logic [4:0] shift;
    logic signed [(LANES*DATA_WIDTH)-1:0] engine_y_flat;

    assign cmd_ready = !engine_busy && !rsp_valid;

    function automatic [COUNTER_WIDTH-1:0] count_valid_lanes(
        input logic [LANES-1:0] mask
    );
        integer lane;
        begin
            count_valid_lanes = '0;
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                count_valid_lanes = count_valid_lanes + mask[lane];
            end
        end
    endfunction

    vector_engine #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH),
        .OP_VEC_ADD(OP_VEC_ADD),
        .OP_VEC_SUB(OP_VEC_SUB),
        .OP_VEC_MUL(OP_VEC_MUL),
        .OP_VEC_SCALE(OP_VEC_SCALE),
        .OP_VEC_REQUANT(OP_VEC_REQUANT),
        .OP_VEC_CLAMP(OP_VEC_CLAMP),
        .OP_VEC_SCALE_FIXED(OP_VEC_SCALE_FIXED)
    ) u_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(engine_start),
        .op(op),
        .valid_mask(valid_mask),
        .a_flat(a_flat),
        .b_flat(b_flat),
        .scalar(scalar),
        .clamp_low(clamp_low),
        .clamp_high(clamp_high),
        .shift(shift),
        .invalid_value('0),
        .done(engine_done),
        .active(engine_active),
        .y_flat(engine_y_flat)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            engine_start <= 1'b0;
            engine_busy <= 1'b0;
            rsp_valid <= 1'b0;
            rsp_y_flat <= '0;
            active_cycles <= '0;
            input_stall_cycles <= '0;
            output_stall_cycles <= '0;
            idle_cycles <= '0;
            accepted_ops <= '0;
            accepted_lane_ops <= '0;
        end else begin
            engine_start <= 1'b0;
            if (engine_busy || rsp_valid || (cmd_valid && cmd_ready)) active_cycles <= active_cycles + 1'b1;
            if (cmd_valid && !cmd_ready) input_stall_cycles <= input_stall_cycles + 1'b1;
            if (rsp_valid && !rsp_ready) output_stall_cycles <= output_stall_cycles + 1'b1;
            if (!engine_busy && !rsp_valid && !(cmd_valid && cmd_ready)) idle_cycles <= idle_cycles + 1'b1;
            if (rsp_valid && rsp_ready) begin
                rsp_valid <= 1'b0;
            end
            if (cmd_valid && cmd_ready) begin
                op <= cmd_op;
                valid_mask <= cmd_valid_mask;
                a_flat <= cmd_a_flat;
                b_flat <= cmd_b_flat;
                scalar <= cmd_scalar;
                clamp_low <= cmd_clamp_low;
                clamp_high <= cmd_clamp_high;
                shift <= cmd_shift;
                engine_start <= 1'b1;
                engine_busy <= 1'b1;
                accepted_ops <= accepted_ops + 1'b1;
                accepted_lane_ops <= accepted_lane_ops + count_valid_lanes(cmd_valid_mask);
            end
            if (engine_done) begin
                rsp_y_flat <= engine_y_flat;
                rsp_valid <= 1'b1;
                engine_busy <= 1'b0;
            end
        end
    end
endmodule

module reduction_engine_handshake #(
    parameter int MAX_LEN = 128,
    parameter int DATA_WIDTH = 32,
    parameter int LEN_WIDTH = 8,
    parameter int RESULT_WIDTH = 64,
    parameter int OP_REDUCE_MAX = 0,
    parameter int OP_REDUCE_SUM = 1,
    parameter int OP_REDUCE_SUMSQ = 2,
    parameter int COUNTER_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic cmd_valid,
    output logic cmd_ready,
    input  logic [1:0] cmd_op,
    input  logic [LEN_WIDTH-1:0] cmd_length,
    input  logic signed [(MAX_LEN*DATA_WIDTH)-1:0] cmd_x_flat,
    output logic rsp_valid,
    input  logic rsp_ready,
    output logic signed [RESULT_WIDTH-1:0] rsp_result,
    output logic [COUNTER_WIDTH-1:0] active_cycles,
    output logic [COUNTER_WIDTH-1:0] input_stall_cycles,
    output logic [COUNTER_WIDTH-1:0] output_stall_cycles,
    output logic [COUNTER_WIDTH-1:0] idle_cycles,
    output logic [COUNTER_WIDTH-1:0] accepted_ops,
    output logic [COUNTER_WIDTH-1:0] accepted_element_ops
);
    logic engine_start;
    logic engine_done;
    logic engine_active;
    logic engine_busy;
    logic [1:0] op;
    logic [LEN_WIDTH-1:0] length;
    logic signed [(MAX_LEN*DATA_WIDTH)-1:0] x_flat;
    logic signed [RESULT_WIDTH-1:0] engine_result;

    assign cmd_ready = !engine_busy && !rsp_valid;

    reduction_engine #(
        .MAX_LEN(MAX_LEN),
        .DATA_WIDTH(DATA_WIDTH),
        .LEN_WIDTH(LEN_WIDTH),
        .RESULT_WIDTH(RESULT_WIDTH),
        .OP_REDUCE_MAX(OP_REDUCE_MAX),
        .OP_REDUCE_SUM(OP_REDUCE_SUM),
        .OP_REDUCE_SUMSQ(OP_REDUCE_SUMSQ)
    ) u_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(engine_start),
        .op(op),
        .length(length),
        .valid_mask({MAX_LEN{1'b1}}),
        .x_flat(x_flat),
        .done(engine_done),
        .active(engine_active),
        .result(engine_result)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            engine_start <= 1'b0;
            engine_busy <= 1'b0;
            rsp_valid <= 1'b0;
            rsp_result <= '0;
            active_cycles <= '0;
            input_stall_cycles <= '0;
            output_stall_cycles <= '0;
            idle_cycles <= '0;
            accepted_ops <= '0;
            accepted_element_ops <= '0;
        end else begin
            engine_start <= 1'b0;
            if (engine_busy || rsp_valid || (cmd_valid && cmd_ready)) active_cycles <= active_cycles + 1'b1;
            if (cmd_valid && !cmd_ready) input_stall_cycles <= input_stall_cycles + 1'b1;
            if (rsp_valid && !rsp_ready) output_stall_cycles <= output_stall_cycles + 1'b1;
            if (!engine_busy && !rsp_valid && !(cmd_valid && cmd_ready)) idle_cycles <= idle_cycles + 1'b1;
            if (rsp_valid && rsp_ready) begin
                rsp_valid <= 1'b0;
            end
            if (cmd_valid && cmd_ready) begin
                op <= cmd_op;
                length <= cmd_length;
                x_flat <= cmd_x_flat;
                engine_start <= 1'b1;
                engine_busy <= 1'b1;
                accepted_ops <= accepted_ops + 1'b1;
                accepted_element_ops <= accepted_element_ops + cmd_length;
            end
            if (engine_done) begin
                rsp_result <= engine_result;
                rsp_valid <= 1'b1;
                engine_busy <= 1'b0;
            end
        end
    end
endmodule

module sfu_lut_handshake #(
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
    parameter int OP_SFU_RSQRT = 2,
    parameter int COUNTER_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic cmd_valid,
    output logic cmd_ready,
    input  logic [1:0] cmd_op,
    input  logic signed [DATA_WIDTH-1:0] cmd_x,
    output logic rsp_valid,
    input  logic rsp_ready,
    output logic [DATA_WIDTH-1:0] rsp_y,
    output logic [COUNTER_WIDTH-1:0] active_cycles,
    output logic [COUNTER_WIDTH-1:0] input_stall_cycles,
    output logic [COUNTER_WIDTH-1:0] output_stall_cycles,
    output logic [COUNTER_WIDTH-1:0] idle_cycles,
    output logic [COUNTER_WIDTH-1:0] exp_ops,
    output logic [COUNTER_WIDTH-1:0] recip_ops,
    output logic [COUNTER_WIDTH-1:0] rsqrt_ops
);
    logic engine_start;
    logic engine_done;
    logic engine_active;
    logic engine_busy;
    logic [1:0] op;
    logic signed [DATA_WIDTH-1:0] x;
    logic [DATA_WIDTH-1:0] engine_y;

    assign cmd_ready = !engine_busy && !rsp_valid;

    sfu_lut #(
        .DATA_WIDTH(DATA_WIDTH),
        .EXP_INPUT_SCALE(EXP_INPUT_SCALE),
        .EXP_LUT_ENTRIES(EXP_LUT_ENTRIES),
        .EXP_OUTPUT_Q(EXP_OUTPUT_Q),
        .RECIP_OUTPUT_Q(RECIP_OUTPUT_Q),
        .RSQRT_OUTPUT_Q(RSQRT_OUTPUT_Q),
        .BRINGUP_EXP_SEG_0(BRINGUP_EXP_SEG_0),
        .BRINGUP_EXP_SEG_1(BRINGUP_EXP_SEG_1),
        .BRINGUP_EXP_SEG_2(BRINGUP_EXP_SEG_2),
        .BRINGUP_EXP_SEG_3(BRINGUP_EXP_SEG_3),
        .BRINGUP_EXP_SEG_4(BRINGUP_EXP_SEG_4),
        .BRINGUP_EXP_SEG_5(BRINGUP_EXP_SEG_5),
        .BRINGUP_EXP_SEG_6(BRINGUP_EXP_SEG_6),
        .BRINGUP_EXP_SEG_7(BRINGUP_EXP_SEG_7),
        .BRINGUP_EXP_SEG_8(BRINGUP_EXP_SEG_8),
        .OP_SFU_EXP(OP_SFU_EXP),
        .OP_SFU_RECIP(OP_SFU_RECIP),
        .OP_SFU_RSQRT(OP_SFU_RSQRT)
    ) u_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(engine_start),
        .op(op),
        .x(x),
        .done(engine_done),
        .active(engine_active),
        .y(engine_y)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            engine_start <= 1'b0;
            engine_busy <= 1'b0;
            rsp_valid <= 1'b0;
            rsp_y <= '0;
            active_cycles <= '0;
            input_stall_cycles <= '0;
            output_stall_cycles <= '0;
            idle_cycles <= '0;
            exp_ops <= '0;
            recip_ops <= '0;
            rsqrt_ops <= '0;
        end else begin
            engine_start <= 1'b0;
            if (engine_busy || rsp_valid || (cmd_valid && cmd_ready)) active_cycles <= active_cycles + 1'b1;
            if (cmd_valid && !cmd_ready) input_stall_cycles <= input_stall_cycles + 1'b1;
            if (rsp_valid && !rsp_ready) output_stall_cycles <= output_stall_cycles + 1'b1;
            if (!engine_busy && !rsp_valid && !(cmd_valid && cmd_ready)) idle_cycles <= idle_cycles + 1'b1;
            if (rsp_valid && rsp_ready) begin
                rsp_valid <= 1'b0;
            end
            if (cmd_valid && cmd_ready) begin
                op <= cmd_op;
                x <= cmd_x;
                engine_start <= 1'b1;
                engine_busy <= 1'b1;
                if (cmd_op == OP_SFU_EXP[1:0]) exp_ops <= exp_ops + 1'b1;
                if (cmd_op == OP_SFU_RECIP[1:0]) recip_ops <= recip_ops + 1'b1;
                if (cmd_op == OP_SFU_RSQRT[1:0]) rsqrt_ops <= rsqrt_ops + 1'b1;
            end
            if (engine_done) begin
                rsp_y <= engine_y;
                rsp_valid <= 1'b1;
                engine_busy <= 1'b0;
            end
        end
    end
endmodule
