module npu_v0_top #(
    parameter int CORE_HOST_LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [1:0]  op,          // 0: uop program, 1: attention softmax v1, 2: matrix u16s8 q15.
    output logic        done,
    output logic        perf_active,
    output logic        perf_fetch_active,
    output logic        perf_matmul_active,
    output logic        perf_done_active,

    input  logic [CORE_HOST_LANES-1:0] host_we,
    input  logic [11:0] host_addr,
    input  logic [(CORE_HOST_LANES*32)-1:0] host_wdata,
    output logic [(CORE_HOST_LANES*32)-1:0] host_rdata
);
    `include "npu_v0_spec.svh"

    typedef enum logic [4:0] {
        ST_IDLE,
        ST_FETCH,
        ST_MATMUL,
        ST_DONE,
        ST_ATTN_PREPARE,
        ST_ATTN_REDMAX_START,
        ST_ATTN_REDMAX_WAIT,
        ST_ATTN_VSUB_START,
        ST_ATTN_VSUB_WAIT,
        ST_ATTN_VCLAMP_START,
        ST_ATTN_VCLAMP_WAIT,
        ST_ATTN_EXP_START,
        ST_ATTN_EXP_WAIT,
        ST_ATTN_REDSUM_PREPARE,
        ST_ATTN_REDSUM_START,
        ST_ATTN_REDSUM_WAIT,
        ST_ATTN_RECIP_START,
        ST_ATTN_RECIP_WAIT,
        ST_ATTN_NORM_START,
        ST_ATTN_NORM_WAIT
    } state_t;

    localparam logic [7:0] RTL_SOFTMAX_LEN_U8 = RTL_SOFTMAX_LEN;

    logic [15:0]        dram_a [0:RTL_MATMUL_ELEMS-1];
    logic [15:0]        dram_a_bank1 [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_b [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_b_bank1 [0:RTL_MATMUL_ELEMS-1];
    logic signed [31:0] dram_c [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_x [0:RTL_SOFTMAX_LEN-1];
    logic [31:0]        dram_y [0:RTL_SOFTMAX_LEN-1];

    logic [15:0]        spad_a [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  spad_b [0:RTL_MATMUL_ELEMS-1];
    logic signed [15:0] vec_buf [0:RTL_SOFTMAX_LEN-1];
    logic signed [15:0] scalar_max;
    logic [15:0]        scalar_sum;
    logic [31:0]        instr_mem [0:RTL_HOST_PROGRAM_WORDS-1];

    state_t state;
    logic [$clog2(RTL_HOST_PROGRAM_WORDS)-1:0] pc;
    logic [31:0] instr;
    logic [3:0] opcode;
    logic [3:0] arg0;
    logic [3:0] arg1;
    logic matmul_start;
    logic matmul_done;
    logic matmul_accumulate_enable;
    logic host_write_bank;
    logic compute_bank_select;
    logic compute_bank_active;
    logic [(RTL_MATMUL_ELEMS*16)-1:0] matmul_a_flat;
    logic [(RTL_MATMUL_ELEMS*8)-1:0]  matmul_b_flat;
    logic [(RTL_MATMUL_ELEMS*32)-1:0] matmul_result_flat;
    logic [(RTL_MATMUL_ELEMS*32)-1:0] acc_read_data_flat;
    logic acc_clear_request;
    logic acc_read_enable;
    logic acc_write_enable;
    logic [31:0] acc_read_count;
    logic [31:0] acc_write_count;
    logic [31:0] acc_clear_count;
    logic [31:0] acc_residency_cycles;
    logic [31:0] acc_spill_count;
    logic primitive_softmax_start;
    logic [2:0] primitive_vector_op;
    logic [7:0] primitive_vector_valid_mask;
    logic signed [(RTL_SOFTMAX_LEN*32)-1:0] primitive_vector_a_flat;
    logic signed [(RTL_SOFTMAX_LEN*32)-1:0] primitive_vector_b_flat;
    logic signed [(RTL_SOFTMAX_LEN*32)-1:0] primitive_vector_y_flat;
    logic signed [31:0] primitive_vector_scalar;
    logic signed [31:0] primitive_vector_clamp_low;
    logic signed [31:0] primitive_vector_clamp_high;
    logic [4:0] primitive_vector_shift;
    logic primitive_vector_done;
    logic primitive_vector_active;
    logic primitive_reduction_start;
    logic [1:0] primitive_reduction_op;
    logic signed [(RTL_SOFTMAX_LEN*32)-1:0] primitive_reduction_x_flat;
    logic signed [63:0] primitive_reduction_result;
    logic primitive_reduction_done;
    logic primitive_reduction_active;
    logic primitive_sfu_start;
    logic [1:0] primitive_sfu_op;
    logic signed [31:0] primitive_sfu_x;
    logic [31:0] primitive_sfu_y;
    logic primitive_sfu_done;
    logic primitive_sfu_active;
    logic [3:0] primitive_lane_idx;

    integer idx;
    integer host_lane_idx;
    integer host_read_lane_idx;
    logic [11:0] host_lane_addr;
    logic [11:0] host_read_lane_addr;

    genvar matmul_flat_idx;
    generate
        for (matmul_flat_idx = 0; matmul_flat_idx < RTL_MATMUL_ELEMS; matmul_flat_idx = matmul_flat_idx + 1) begin : gen_matmul_flat
            assign matmul_a_flat[(matmul_flat_idx * 16) +: 16] = spad_a[matmul_flat_idx];
            assign matmul_b_flat[(matmul_flat_idx * 8) +: 8] = spad_b[matmul_flat_idx];
        end
    endgenerate

    assign opcode = instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB];
    assign arg0 = instr[UOP_ARG0_MSB:UOP_ARG0_LSB];
    assign arg1 = instr[UOP_ARG1_MSB:UOP_ARG1_LSB];
    assign perf_active = start || state != ST_IDLE;
    assign perf_fetch_active = state == ST_FETCH;
    assign perf_matmul_active = state == ST_MATMUL;
    assign perf_done_active = state == ST_DONE;
    assign acc_read_enable =
        (state == ST_FETCH) &&
        (instr_mem[pc][UOP_OPCODE_MSB:UOP_OPCODE_LSB] == UOP_STORE) &&
        (instr_mem[pc][UOP_ARG0_MSB:UOP_ARG0_LSB] == TENSOR_C) &&
        (instr_mem[pc][UOP_ARG1_MSB:UOP_ARG1_LSB] == BUF_ACC);
    assign acc_write_enable = (state == ST_MATMUL) && matmul_done;

    matmul_array #(
        .M(RTL_MATMUL_M),
        .N(RTL_MATMUL_N),
        .K(RTL_MATMUL_K)
    ) u_matmul_array (
        .clk(clk),
        .rst_n(rst_n),
        .start(matmul_start),
        .mixed_u16s8_q15(op == 2'd2),
        .done(matmul_done),
        .a_flat(matmul_a_flat),
        .b_flat(matmul_b_flat),
        .result_flat(matmul_result_flat)
    );

    accumulator_file #(
        .TILE_ELEMS(RTL_MATMUL_ELEMS),
        .ACC_WIDTH(32),
        .BANKS(2),
        .COUNTER_WIDTH(32)
    ) u_accumulator_file (
        .clk(clk),
        .rst_n(rst_n),
        .bank_select(1'b0),
        .clear(acc_clear_request),
        .read_enable(acc_read_enable),
        .write_enable(acc_write_enable),
        .accumulate_enable(matmul_accumulate_enable),
        .write_data_flat(matmul_result_flat),
        .read_data_flat(acc_read_data_flat),
        .acc_read_count(acc_read_count),
        .acc_write_count(acc_write_count),
        .acc_clear_count(acc_clear_count),
        .acc_residency_cycles(acc_residency_cycles),
        .acc_spill_count(acc_spill_count)
    );

    vector_engine #(
        .LANES(RTL_SOFTMAX_LEN),
        .DATA_WIDTH(32),
        .OP_VEC_ADD(0),
        .OP_VEC_SUB(1),
        .OP_VEC_MUL(2),
        .OP_VEC_SCALE(3),
        .OP_VEC_REQUANT(4),
        .OP_VEC_CLAMP(5)
    ) u_attention_vector_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(primitive_softmax_start),
        .op(primitive_vector_op),
        .valid_mask(primitive_vector_valid_mask),
        .a_flat(primitive_vector_a_flat),
        .b_flat(primitive_vector_b_flat),
        .scalar(primitive_vector_scalar),
        .clamp_low(primitive_vector_clamp_low),
        .clamp_high(primitive_vector_clamp_high),
        .shift(primitive_vector_shift),
        .done(primitive_vector_done),
        .active(primitive_vector_active),
        .y_flat(primitive_vector_y_flat)
    );

    reduction_engine #(
        .MAX_LEN(RTL_SOFTMAX_LEN),
        .DATA_WIDTH(32),
        .RESULT_WIDTH(64),
        .OP_REDUCE_MAX(0),
        .OP_REDUCE_SUM(1),
        .OP_REDUCE_SUMSQ(2)
    ) u_attention_reduction_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(primitive_reduction_start),
        .op(primitive_reduction_op),
        .length(RTL_SOFTMAX_LEN_U8),
        .x_flat(primitive_reduction_x_flat),
        .done(primitive_reduction_done),
        .active(primitive_reduction_active),
        .result(primitive_reduction_result)
    );

    sfu_lut #(
        .DATA_WIDTH(32),
        .EXP_INPUT_SCALE(32),
        .EXP_LUT_ENTRIES(257),
        .EXP_OUTPUT_Q(15),
        .RECIP_OUTPUT_Q(24),
        .RSQRT_OUTPUT_Q(24),
        .OP_SFU_EXP(0),
        .OP_SFU_RECIP(1),
        .OP_SFU_RSQRT(2)
    ) u_attention_sfu_lut (
        .clk(clk),
        .rst_n(rst_n),
        .start(primitive_sfu_start),
        .op(primitive_sfu_op),
        .x(primitive_sfu_x),
        .done(primitive_sfu_done),
        .active(primitive_sfu_active),
        .y(primitive_sfu_y)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (idx = 0; idx < RTL_MATMUL_ELEMS; idx = idx + 1) begin
                dram_a[idx] <= '0;
                dram_a_bank1[idx] <= '0;
                dram_b[idx] <= '0;
                dram_b_bank1[idx] <= '0;
                dram_c[idx] <= '0;
                spad_a[idx] <= '0;
                spad_b[idx] <= '0;
            end
            for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                dram_x[idx] <= '0;
                dram_y[idx] <= '0;
                vec_buf[idx] <= '0;
            end
            for (idx = 0; idx < RTL_HOST_PROGRAM_WORDS; idx = idx + 1) begin
                instr_mem[idx] <= '0;
            end
            matmul_accumulate_enable <= 1'b0;
            host_write_bank <= 1'b0;
            compute_bank_select <= 1'b0;
            acc_clear_request <= 1'b0;
            primitive_softmax_start <= 1'b0;
            primitive_reduction_start <= 1'b0;
            primitive_sfu_start <= 1'b0;
            primitive_vector_op <= '0;
            primitive_vector_valid_mask <= '0;
            primitive_vector_a_flat <= '0;
            primitive_vector_b_flat <= '0;
            primitive_vector_scalar <= '0;
            primitive_vector_clamp_low <= '0;
            primitive_vector_clamp_high <= '0;
            primitive_vector_shift <= '0;
            primitive_reduction_op <= '0;
            primitive_reduction_x_flat <= '0;
            primitive_sfu_op <= '0;
            primitive_sfu_x <= '0;
            primitive_lane_idx <= '0;
        end else begin
            acc_clear_request <= 1'b0;
            for (host_lane_idx = 0; host_lane_idx < CORE_HOST_LANES; host_lane_idx = host_lane_idx + 1) begin
                if (host_we[host_lane_idx]) begin
                    host_lane_addr = host_addr + host_lane_idx[11:0];
                    if (host_lane_addr >= RTL_HOST_A_BASE &&
                        host_lane_addr < RTL_HOST_A_BASE + RTL_HOST_A_WORDS) begin
                        if (host_write_bank) begin
                            dram_a_bank1[host_lane_addr - RTL_HOST_A_BASE] <= host_wdata[(host_lane_idx * 32) +: 16];
                        end else begin
                            dram_a[host_lane_addr - RTL_HOST_A_BASE] <= host_wdata[(host_lane_idx * 32) +: 16];
                        end
                    end else if (host_lane_addr >= RTL_HOST_B_BASE &&
                                 host_lane_addr < RTL_HOST_B_BASE + RTL_HOST_B_WORDS) begin
                        if (host_write_bank) begin
                            dram_b_bank1[host_lane_addr - RTL_HOST_B_BASE] <= host_wdata[(host_lane_idx * 32) +: 8];
                        end else begin
                            dram_b[host_lane_addr - RTL_HOST_B_BASE] <= host_wdata[(host_lane_idx * 32) +: 8];
                        end
                    end else if (state == ST_IDLE && host_lane_addr >= RTL_HOST_X_BASE &&
                                 host_lane_addr < RTL_HOST_X_BASE + RTL_HOST_X_WORDS) begin
                        dram_x[host_lane_addr - RTL_HOST_X_BASE] <= host_wdata[(host_lane_idx * 32) +: 8];
                    end else if (state == ST_IDLE && host_lane_addr >= RTL_HOST_PROGRAM_BASE &&
                                 host_lane_addr < RTL_HOST_PROGRAM_BASE + RTL_HOST_PROGRAM_WORDS) begin
                        instr_mem[host_lane_addr - RTL_HOST_PROGRAM_BASE] <= host_wdata[(host_lane_idx * 32) +: 32];
                    end else if (host_lane_addr == RTL_HOST_CONTROL_BASE) begin
                        matmul_accumulate_enable <= host_wdata[(host_lane_idx * 32) + RTL_CTRL_ACCUMULATE_ENABLE_BIT];
                        host_write_bank <= host_wdata[(host_lane_idx * 32) + RTL_CTRL_HOST_WRITE_BANK_BIT];
                        compute_bank_select <= host_wdata[(host_lane_idx * 32) + RTL_CTRL_COMPUTE_BANK_SELECT_BIT];
                        if (host_wdata[(host_lane_idx * 32) + RTL_CTRL_ACCUMULATOR_CLEAR_BIT]) begin
                            acc_clear_request <= 1'b1;
                        end
                    end
                end
            end
        end
    end

    always @* begin
        host_rdata = '0;
        for (host_read_lane_idx = 0; host_read_lane_idx < CORE_HOST_LANES; host_read_lane_idx = host_read_lane_idx + 1) begin
            host_read_lane_addr = host_addr + host_read_lane_idx[11:0];
            if (host_read_lane_addr >= RTL_HOST_C_BASE &&
                host_read_lane_addr < RTL_HOST_C_BASE + RTL_HOST_C_WORDS) begin
                host_rdata[(host_read_lane_idx * 32) +: 32] = dram_c[host_read_lane_addr - RTL_HOST_C_BASE];
            end else if (host_read_lane_addr >= RTL_HOST_Y_BASE &&
                         host_read_lane_addr < RTL_HOST_Y_BASE + RTL_HOST_Y_WORDS) begin
                host_rdata[(host_read_lane_idx * 32) +: 32] = dram_y[host_read_lane_addr - RTL_HOST_Y_BASE];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            pc <= '0;
            instr <= '0;
            matmul_start <= 1'b0;
            scalar_max <= '0;
            scalar_sum <= '0;
            compute_bank_active <= 1'b0;
        end else begin
            done <= 1'b0;
            matmul_start <= 1'b0;
            primitive_softmax_start <= 1'b0;
            primitive_reduction_start <= 1'b0;
            primitive_sfu_start <= 1'b0;
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        compute_bank_active <= compute_bank_select;
                        pc <= 4'h0;
                        state <= (op == 2'd1) ? ST_ATTN_PREPARE : ST_FETCH;
                    end
                end

                ST_FETCH: begin
                    instr <= instr_mem[pc];
                    pc <= pc + 1'b1;
                    case (instr_mem[pc][UOP_OPCODE_MSB:UOP_OPCODE_LSB])
                        UOP_LOAD: begin
                            run_load(
                                instr_mem[pc][UOP_ARG0_MSB:UOP_ARG0_LSB],
                                instr_mem[pc][UOP_ARG1_MSB:UOP_ARG1_LSB]
                            );
                        end
                        UOP_STORE: begin
                            run_store(
                                instr_mem[pc][UOP_ARG0_MSB:UOP_ARG0_LSB],
                                instr_mem[pc][UOP_ARG1_MSB:UOP_ARG1_LSB]
                            );
                        end
                        UOP_MATMUL: begin
                            matmul_start <= 1'b1;
                            state <= ST_MATMUL;
                        end
                        UOP_VREDMAX: begin
                            run_vredmax();
                        end
                        UOP_VSUB: begin
                            run_vsub();
                        end
                        UOP_VEXP: begin
                            run_vexp();
                        end
                        UOP_VREDSUM: begin
                            run_vredsum();
                        end
                        UOP_VDIV: begin
                            run_vdiv();
                        end
                        UOP_HALT: begin
                            state <= ST_DONE;
                        end
                        default: begin
                            state <= ST_DONE;
                        end
                    endcase
                end

                ST_MATMUL: begin
                    if (matmul_done) begin
                        state <= ST_FETCH;
                    end
                end

                ST_DONE: begin
                    done <= 1'b1;
                    if (!start) begin
                        state <= ST_IDLE;
                    end
                end

                ST_ATTN_PREPARE: begin
                    for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                        primitive_vector_a_flat[(idx * 32) +: 32] <= {{24{dram_x[idx][7]}}, dram_x[idx]};
                        primitive_reduction_x_flat[(idx * 32) +: 32] <= {{24{dram_x[idx][7]}}, dram_x[idx]};
                    end
                    primitive_vector_valid_mask <= 8'hff;
                    state <= ST_ATTN_REDMAX_START;
                end

                ST_ATTN_REDMAX_START: begin
                    primitive_reduction_op <= 2'd0;
                    primitive_reduction_start <= 1'b1;
                    state <= ST_ATTN_REDMAX_WAIT;
                end

                ST_ATTN_REDMAX_WAIT: begin
                    if (primitive_reduction_done) begin
                        for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                            primitive_vector_b_flat[(idx * 32) +: 32] <= primitive_reduction_result[31:0];
                        end
                        state <= ST_ATTN_VSUB_START;
                    end
                end

                ST_ATTN_VSUB_START: begin
                    primitive_vector_op <= 3'd1;
                    primitive_softmax_start <= 1'b1;
                    state <= ST_ATTN_VSUB_WAIT;
                end

                ST_ATTN_VSUB_WAIT: begin
                    if (primitive_vector_done) begin
                        primitive_vector_a_flat <= primitive_vector_y_flat;
                        state <= ST_ATTN_VCLAMP_START;
                    end
                end

                ST_ATTN_VCLAMP_START: begin
                    primitive_vector_op <= 3'd5;
                    primitive_vector_clamp_low <= -32'sd256;
                    primitive_vector_clamp_high <= 32'sd0;
                    primitive_softmax_start <= 1'b1;
                    state <= ST_ATTN_VCLAMP_WAIT;
                end

                ST_ATTN_VCLAMP_WAIT: begin
                    if (primitive_vector_done) begin
                        primitive_vector_a_flat <= primitive_vector_y_flat;
                        primitive_lane_idx <= 4'h0;
                        state <= ST_ATTN_EXP_START;
                    end
                end

                ST_ATTN_EXP_START: begin
                    primitive_sfu_op <= 2'd0;
                    primitive_sfu_x <= primitive_vector_a_flat[(primitive_lane_idx * 32) +: 32];
                    primitive_sfu_start <= 1'b1;
                    state <= ST_ATTN_EXP_WAIT;
                end

                ST_ATTN_EXP_WAIT: begin
                    if (primitive_sfu_done) begin
                        primitive_vector_a_flat[(primitive_lane_idx * 32) +: 32] <= primitive_sfu_y;
                        if (primitive_lane_idx == RTL_SOFTMAX_LEN - 1) begin
                            state <= ST_ATTN_REDSUM_PREPARE;
                        end else begin
                            primitive_lane_idx <= primitive_lane_idx + 1'b1;
                            state <= ST_ATTN_EXP_START;
                        end
                    end
                end

                ST_ATTN_REDSUM_PREPARE: begin
                    primitive_reduction_x_flat <= primitive_vector_a_flat;
                    state <= ST_ATTN_REDSUM_START;
                end

                ST_ATTN_REDSUM_START: begin
                    primitive_reduction_op <= 2'd1;
                    primitive_reduction_start <= 1'b1;
                    state <= ST_ATTN_REDSUM_WAIT;
                end

                ST_ATTN_REDSUM_WAIT: begin
                    if (primitive_reduction_done) begin
                        primitive_sfu_op <= 2'd1;
                        primitive_sfu_x <= primitive_reduction_result[31:0];
                        state <= ST_ATTN_RECIP_START;
                    end
                end

                ST_ATTN_RECIP_START: begin
                    primitive_sfu_start <= 1'b1;
                    state <= ST_ATTN_RECIP_WAIT;
                end

                ST_ATTN_RECIP_WAIT: begin
                    if (primitive_sfu_done) begin
                        primitive_vector_scalar <= primitive_sfu_y;
                        primitive_vector_shift <= 5'd9;
                        state <= ST_ATTN_NORM_START;
                    end
                end

                ST_ATTN_NORM_START: begin
                    primitive_vector_op <= 3'd3;
                    primitive_softmax_start <= 1'b1;
                    state <= ST_ATTN_NORM_WAIT;
                end

                ST_ATTN_NORM_WAIT: begin
                    if (primitive_vector_done) begin
                        for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                            dram_y[idx] <= primitive_vector_y_flat[(idx * 32) +: 32];
                        end
                        state <= ST_DONE;
                    end
                end
            endcase
        end
    end

    task automatic run_load(input logic [3:0] tensor, input logic [3:0] buffer);
        integer l;
        begin
            if (tensor == TENSOR_A && buffer == BUF_SPAD_A) begin
                if (compute_bank_active) begin
                    for (l = 0; l < RTL_MATMUL_ELEMS; l = l + 1) spad_a[l] = dram_a_bank1[l];
                end else begin
                    for (l = 0; l < RTL_MATMUL_ELEMS; l = l + 1) spad_a[l] = dram_a[l];
                end
            end else if (tensor == TENSOR_B && buffer == BUF_SPAD_B) begin
                if (compute_bank_active) begin
                    for (l = 0; l < RTL_MATMUL_ELEMS; l = l + 1) spad_b[l] = dram_b_bank1[l];
                end else begin
                    for (l = 0; l < RTL_MATMUL_ELEMS; l = l + 1) spad_b[l] = dram_b[l];
                end
            end else if (tensor == TENSOR_X && buffer == BUF_VEC) begin
                for (l = 0; l < RTL_SOFTMAX_LEN; l = l + 1) vec_buf[l] = {{8{dram_x[l][7]}}, dram_x[l]};
            end
        end
    endtask

    task automatic run_store(input logic [3:0] tensor, input logic [3:0] buffer);
        integer s;
        begin
            if (tensor == TENSOR_C && buffer == BUF_ACC) begin
                for (s = 0; s < RTL_MATMUL_ELEMS; s = s + 1) begin
                    dram_c[s] = $signed(acc_read_data_flat[(s * 32) +: 32]);
                end
            end else if (tensor == TENSOR_Y && buffer == BUF_VEC) begin
                for (s = 0; s < RTL_SOFTMAX_LEN; s = s + 1) dram_y[s] = vec_buf[s][7:0];
            end
        end
    endtask

    task automatic run_vredmax;
        integer v;
        begin
            scalar_max = vec_buf[0];
            for (v = 1; v < RTL_SOFTMAX_LEN; v = v + 1) begin
                if (vec_buf[v] > scalar_max) scalar_max = vec_buf[v];
            end
        end
    endtask

    task automatic run_vsub;
        integer v;
        begin
            for (v = 0; v < RTL_SOFTMAX_LEN; v = v + 1) begin
                vec_buf[v] = vec_buf[v] - scalar_max;
            end
        end
    endtask

    task automatic run_vexp;
        integer v;
        begin
            for (v = 0; v < RTL_SOFTMAX_LEN; v = v + 1) begin
                vec_buf[v] = {8'h0, exp_lut_q8(vec_buf[v][8:0])};
            end
        end
    endtask

    task automatic run_vredsum;
        integer v;
        begin
            scalar_sum = 16'h0;
            for (v = 0; v < RTL_SOFTMAX_LEN; v = v + 1) begin
                scalar_sum = scalar_sum + vec_buf[v][7:0];
            end
        end
    endtask

    task automatic run_vdiv;
        integer v;
        logic [23:0] norm_prod;
        logic [23:0] quotient;
        begin
            for (v = 0; v < RTL_SOFTMAX_LEN; v = v + 1) begin
                norm_prod = {16'h0, vec_buf[v][7:0]} * 24'd255;
                quotient = (scalar_sum == 0) ? 24'h0 : (norm_prod / {8'h0, scalar_sum});
                vec_buf[v] = {8'h0, quotient[7:0]};
            end
        end
    endtask

    function automatic [7:0] exp_lut_q8(input logic signed [8:0] delta);
        begin
            if (delta >= 0) exp_lut_q8 = 8'd255;
            else if (delta == -1) exp_lut_q8 = 8'd94;
            else if (delta == -2) exp_lut_q8 = 8'd35;
            else if (delta == -3) exp_lut_q8 = 8'd13;
            else if (delta == -4) exp_lut_q8 = 8'd5;
            else if (delta == -5) exp_lut_q8 = 8'd2;
            else if (delta == -6) exp_lut_q8 = 8'd1;
            else exp_lut_q8 = 8'd0;
        end
    endfunction
endmodule
