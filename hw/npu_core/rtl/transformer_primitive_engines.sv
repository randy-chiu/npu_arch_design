module transformer_primitive_engines (
    input  logic clk,
    input  logic rst_n,

    input  logic vector_start,
    input  logic [2:0] vector_op,
    input  logic [npu_transformer_v1_config_pkg::CFG_VECTOR_LANES-1:0] vector_valid_mask,
    input  logic signed [(npu_transformer_v1_config_pkg::CFG_VECTOR_LANES*npu_transformer_v1_config_pkg::CFG_VECTOR_DATA_WIDTH)-1:0] vector_a_flat,
    input  logic signed [(npu_transformer_v1_config_pkg::CFG_VECTOR_LANES*npu_transformer_v1_config_pkg::CFG_VECTOR_DATA_WIDTH)-1:0] vector_b_flat,
    input  logic signed [npu_transformer_v1_config_pkg::CFG_VECTOR_DATA_WIDTH-1:0] vector_scalar,
    input  logic signed [npu_transformer_v1_config_pkg::CFG_VECTOR_DATA_WIDTH-1:0] vector_clamp_low,
    input  logic signed [npu_transformer_v1_config_pkg::CFG_VECTOR_DATA_WIDTH-1:0] vector_clamp_high,
    input  logic [4:0] vector_shift,
    output logic vector_done,
    output logic vector_active,
    output logic signed [(npu_transformer_v1_config_pkg::CFG_VECTOR_LANES*npu_transformer_v1_config_pkg::CFG_VECTOR_DATA_WIDTH)-1:0] vector_y_flat,

    input  logic reduction_start,
    input  logic [1:0] reduction_op,
    input  logic [7:0] reduction_length,
    input  logic signed [(npu_transformer_v1_config_pkg::CFG_REDUCTION_MAX_LEN*npu_transformer_v1_config_pkg::CFG_REDUCTION_DATA_WIDTH)-1:0] reduction_x_flat,
    output logic reduction_done,
    output logic reduction_active,
    output logic signed [npu_transformer_v1_config_pkg::CFG_REDUCTION_RESULT_WIDTH-1:0] reduction_result,

    input  logic sfu_start,
    input  logic [1:0] sfu_op,
    input  logic signed [npu_transformer_v1_config_pkg::CFG_SFU_DATA_WIDTH-1:0] sfu_x,
    output logic sfu_done,
    output logic sfu_active,
    output logic [npu_transformer_v1_config_pkg::CFG_SFU_DATA_WIDTH-1:0] sfu_y
);
    import npu_transformer_v1_config_pkg::*;

    vector_engine #(
        .LANES(CFG_VECTOR_LANES),
        .DATA_WIDTH(CFG_VECTOR_DATA_WIDTH),
        .OP_VEC_ADD(CFG_VEC_ADD),
        .OP_VEC_SUB(CFG_VEC_SUB),
        .OP_VEC_MUL(CFG_VEC_MUL),
        .OP_VEC_SCALE(CFG_VEC_SCALE),
        .OP_VEC_REQUANT(CFG_VEC_REQUANT),
        .OP_VEC_CLAMP(CFG_VEC_CLAMP),
        .OP_VEC_SCALE_FIXED(CFG_VEC_SCALE_FIXED)
    ) u_vector_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(vector_start),
        .op(vector_op),
        .valid_mask(vector_valid_mask),
        .a_flat(vector_a_flat),
        .b_flat(vector_b_flat),
        .scalar(vector_scalar),
        .clamp_low(vector_clamp_low),
        .clamp_high(vector_clamp_high),
        .shift(vector_shift),
        .invalid_value('0),
        .done(vector_done),
        .active(vector_active),
        .y_flat(vector_y_flat)
    );

    reduction_engine #(
        .MAX_LEN(CFG_REDUCTION_MAX_LEN),
        .DATA_WIDTH(CFG_REDUCTION_DATA_WIDTH),
        .RESULT_WIDTH(CFG_REDUCTION_RESULT_WIDTH),
        .OP_REDUCE_MAX(CFG_REDUCE_MAX),
        .OP_REDUCE_SUM(CFG_REDUCE_SUM),
        .OP_REDUCE_SUMSQ(CFG_REDUCE_SUMSQ)
    ) u_reduction_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(reduction_start),
        .op(reduction_op),
        .length(reduction_length),
        .valid_mask({CFG_REDUCTION_MAX_LEN{1'b1}}),
        .x_flat(reduction_x_flat),
        .done(reduction_done),
        .active(reduction_active),
        .result(reduction_result)
    );

    sfu_lut #(
        .DATA_WIDTH(CFG_SFU_DATA_WIDTH),
        .EXP_INPUT_SCALE(CFG_SFU_EXP_INPUT_SCALE),
        .EXP_LUT_ENTRIES(CFG_SFU_EXP_LUT_ENTRIES),
        .EXP_OUTPUT_Q(CFG_SFU_EXP_OUTPUT_Q),
        .RECIP_OUTPUT_Q(CFG_SFU_RECIP_OUTPUT_Q),
        .RSQRT_OUTPUT_Q(CFG_SFU_RSQRT_OUTPUT_Q),
        .BRINGUP_EXP_SEG_0(CFG_SFU_BRINGUP_EXP_SEG_0),
        .BRINGUP_EXP_SEG_1(CFG_SFU_BRINGUP_EXP_SEG_1),
        .BRINGUP_EXP_SEG_2(CFG_SFU_BRINGUP_EXP_SEG_2),
        .BRINGUP_EXP_SEG_3(CFG_SFU_BRINGUP_EXP_SEG_3),
        .BRINGUP_EXP_SEG_4(CFG_SFU_BRINGUP_EXP_SEG_4),
        .BRINGUP_EXP_SEG_5(CFG_SFU_BRINGUP_EXP_SEG_5),
        .BRINGUP_EXP_SEG_6(CFG_SFU_BRINGUP_EXP_SEG_6),
        .BRINGUP_EXP_SEG_7(CFG_SFU_BRINGUP_EXP_SEG_7),
        .BRINGUP_EXP_SEG_8(CFG_SFU_BRINGUP_EXP_SEG_8),
        .OP_SFU_EXP(CFG_SFU_EXP),
        .OP_SFU_RECIP(CFG_SFU_RECIP),
        .OP_SFU_RSQRT(CFG_SFU_RSQRT)
    ) u_sfu_lut (
        .clk(clk),
        .rst_n(rst_n),
        .start(sfu_start),
        .op(sfu_op),
        .x(sfu_x),
        .done(sfu_done),
        .active(sfu_active),
        .y(sfu_y)
    );
endmodule
