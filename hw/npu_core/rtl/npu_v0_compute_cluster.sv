module npu_v0_compute_cluster #(
    parameter int CORE_HOST_LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [1:0]  op,          // 0: uop, 1: attention softmax, 2: matrix u16s8 q15, 3: attention scale/mask.
    input  logic        output_store_enable,
    output logic        done,
    output logic        perf_active,
    output logic        perf_fetch_active,
    output logic        perf_matmul_active,
    output logic        perf_done_active,
    output logic        perf_uop_sched_active,
    output logic        perf_uop_sched_wait,
    output logic        perf_matrix_active,
    output logic        perf_local_active,

    input  logic [CORE_HOST_LANES-1:0] host_we,
    input  logic [11:0] host_addr,
    input  logic [(CORE_HOST_LANES*32)-1:0] host_wdata,
    output logic [(CORE_HOST_LANES*32)-1:0] host_rdata
);
    `include "npu_v0_spec.svh"

    typedef enum logic [1:0] {
        LOCAL_ROUTE_IDLE,
        LOCAL_ROUTE_START,
        LOCAL_ROUTE_WAIT
    } local_route_state_t;

    localparam logic [7:0] RTL_SOFTMAX_LEN_U8 = RTL_SOFTMAX_LEN;

    logic [15:0]        dram_a [0:RTL_MATMUL_ELEMS-1];
    logic [15:0]        dram_a_bank1 [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_b [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_b_bank1 [0:RTL_MATMUL_ELEMS-1];
    logic signed [31:0] dram_c [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_x [0:RTL_SOFTMAX_LEN-1];
    logic [31:0]        dram_y [0:RTL_SOFTMAX_LEN-1];

    logic signed [15:0] vec_buf [0:RTL_SOFTMAX_LEN-1];
    logic signed [15:0] scalar_max;
    logic [15:0]        scalar_sum;
    logic [31:0]        instr_mem [0:RTL_HOST_PROGRAM_WORDS-1];

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
    logic uop_sched_done;
    logic uop_sched_load_valid;
    logic uop_sched_store_valid;
    logic uop_sched_exec_valid;
    logic [3:0] uop_sched_opcode;
    logic [3:0] uop_sched_tensor;
    logic [3:0] uop_sched_buffer;
    logic uop_sched_matrix_start;
    logic uop_sched_local_exec_done;
    logic [$clog2(RTL_HOST_PROGRAM_WORDS)-1:0] uop_sched_program_addr;
    logic matrix_datapath_active;
    logic [3:0] primitive_lane_idx;
    logic [3:0] primitive_row_idx;
    local_route_state_t local_route_state;
    logic [3:0] local_route_opcode;
    logic [3:0] local_route_row;
    logic [3:0] local_route_lane;

    integer idx;
    integer host_lane_idx;
    integer host_read_lane_idx;
    logic [11:0] host_lane_addr;
    logic [11:0] host_read_lane_addr;

    genvar matmul_flat_idx;
    generate
        for (matmul_flat_idx = 0; matmul_flat_idx < RTL_MATMUL_ELEMS; matmul_flat_idx = matmul_flat_idx + 1) begin : gen_matmul_flat
            assign matmul_a_flat[(matmul_flat_idx * 16) +: 16] =
                compute_bank_active ? dram_a_bank1[matmul_flat_idx] : dram_a[matmul_flat_idx];
            assign matmul_b_flat[(matmul_flat_idx * 8) +: 8] =
                compute_bank_active ? dram_b_bank1[matmul_flat_idx] : dram_b[matmul_flat_idx];
        end
    endgenerate

    assign perf_local_active =
        (uop_sched_load_valid &&
         uop_sched_tensor != TENSOR_A &&
         uop_sched_tensor != TENSOR_B) ||
        (uop_sched_store_valid && output_store_enable) ||
        uop_sched_exec_valid ||
        acc_write_enable;
    assign perf_active =
        local_route_state != LOCAL_ROUTE_IDLE ||
        perf_local_active || matrix_datapath_active;
    assign perf_fetch_active = perf_uop_sched_active;
    assign perf_matmul_active = matrix_datapath_active;
    assign perf_done_active = uop_sched_done;
    assign acc_read_enable =
        uop_sched_store_valid && output_store_enable &&
        uop_sched_tensor == TENSOR_C &&
        uop_sched_buffer == BUF_ACC;
    assign acc_write_enable = matmul_done;

    npu_v0_uop_scheduler u_uop_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(uop_sched_done),
        .program_addr(uop_sched_program_addr),
        .program_rdata(instr_mem[uop_sched_program_addr]),
        .local_load_valid(uop_sched_load_valid),
        .local_store_valid(uop_sched_store_valid),
        .local_exec_valid(uop_sched_exec_valid),
        .local_opcode(uop_sched_opcode),
        .local_tensor(uop_sched_tensor),
        .local_buffer(uop_sched_buffer),
        .local_exec_blocking(
            (op == 2'd3 && uop_sched_opcode == UOP_VSCALE_FIXED) ||
            (op == 2'd1 && (
                uop_sched_opcode == UOP_VREDMAX ||
                uop_sched_opcode == UOP_VSUB ||
                uop_sched_opcode == UOP_VCLAMP ||
                uop_sched_opcode == UOP_VEXP ||
                uop_sched_opcode == UOP_VREDSUM ||
                uop_sched_opcode == UOP_VDIV ||
                uop_sched_opcode == UOP_VNORM
            ))
        ),
        .local_exec_done(uop_sched_local_exec_done),
        .matrix_start(uop_sched_matrix_start),
        .matrix_done(matmul_done),
        .perf_active(perf_uop_sched_active),
        .perf_wait(perf_uop_sched_wait)
    );

    matmul_array #(
        .M(RTL_MATMUL_M),
        .N(RTL_MATMUL_N),
        .K(RTL_MATMUL_K)
    ) u_matmul_array (
        .clk(clk),
        .rst_n(rst_n),
        .start(uop_sched_matrix_start),
        .mixed_u16s8_q15(op == 2'd2),
        .done(matmul_done),
        .perf_active(matrix_datapath_active),
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
        .OP_VEC_CLAMP(5),
        .OP_VEC_SCALE_FIXED(6)
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
        .EXP_INPUT_SCALE(RTL_SFU_EXP_INPUT_SCALE),
        .EXP_LUT_ENTRIES(RTL_SFU_EXP_LUT_ENTRIES),
        .EXP_OUTPUT_Q(RTL_SFU_EXP_OUTPUT_Q),
        .RECIP_OUTPUT_Q(RTL_SFU_RECIP_OUTPUT_Q),
        .RSQRT_OUTPUT_Q(RTL_SFU_RSQRT_OUTPUT_Q),
        .BRINGUP_EXP_SEG_0(RTL_SFU_BRINGUP_EXP_SEG_0),
        .BRINGUP_EXP_SEG_1(RTL_SFU_BRINGUP_EXP_SEG_1),
        .BRINGUP_EXP_SEG_2(RTL_SFU_BRINGUP_EXP_SEG_2),
        .BRINGUP_EXP_SEG_3(RTL_SFU_BRINGUP_EXP_SEG_3),
        .BRINGUP_EXP_SEG_4(RTL_SFU_BRINGUP_EXP_SEG_4),
        .BRINGUP_EXP_SEG_5(RTL_SFU_BRINGUP_EXP_SEG_5),
        .BRINGUP_EXP_SEG_6(RTL_SFU_BRINGUP_EXP_SEG_6),
        .BRINGUP_EXP_SEG_7(RTL_SFU_BRINGUP_EXP_SEG_7),
        .BRINGUP_EXP_SEG_8(RTL_SFU_BRINGUP_EXP_SEG_8),
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
            primitive_row_idx <= '0;
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
                    end else if (host_lane_addr >= RTL_HOST_X_BASE &&
                                 host_lane_addr < RTL_HOST_X_BASE + RTL_HOST_X_WORDS) begin
                        dram_x[host_lane_addr - RTL_HOST_X_BASE] <= host_wdata[(host_lane_idx * 32) +: 8];
                    end else if (host_lane_addr >= RTL_HOST_C_BASE &&
                                 host_lane_addr < RTL_HOST_C_BASE + RTL_HOST_C_WORDS) begin
                        dram_c[host_lane_addr - RTL_HOST_C_BASE] <= host_wdata[(host_lane_idx * 32) +: 32];
                    end else if (host_lane_addr >= RTL_HOST_PROGRAM_BASE &&
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
            done <= 1'b0;
            scalar_max <= '0;
            scalar_sum <= '0;
            compute_bank_active <= 1'b0;
            local_route_state <= LOCAL_ROUTE_IDLE;
            local_route_opcode <= '0;
            local_route_row <= '0;
            local_route_lane <= '0;
            uop_sched_local_exec_done <= 1'b0;
        end else begin
            done <= 1'b0;
            uop_sched_local_exec_done <= 1'b0;
            primitive_softmax_start <= 1'b0;
            primitive_reduction_start <= 1'b0;
            primitive_sfu_start <= 1'b0;
            if (uop_sched_load_valid) begin
                run_load(uop_sched_tensor, uop_sched_buffer);
            end
            if (uop_sched_store_valid && output_store_enable) begin
                run_store(uop_sched_tensor, uop_sched_buffer);
            end
            if (uop_sched_exec_valid) begin
                if (op == 2'd0) begin
                    case (uop_sched_opcode)
                        UOP_VREDMAX: run_vredmax();
                        UOP_VSUB: run_vsub();
                        UOP_VEXP: run_vexp();
                        UOP_VREDSUM: run_vredsum();
                        UOP_VDIV: run_vdiv();
                        default: begin
                        end
                    endcase
                end else begin
                local_route_opcode <= uop_sched_opcode;
                local_route_row <= uop_sched_tensor;
                local_route_lane <= uop_sched_buffer;
                primitive_row_idx <= uop_sched_tensor;
                primitive_lane_idx <= uop_sched_buffer;
                primitive_vector_valid_mask <= {RTL_SOFTMAX_LEN{1'b1}};
                case (uop_sched_opcode)
                    UOP_VREDMAX: begin
                        primitive_reduction_op <= 2'd0;
                        for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                            primitive_vector_a_flat[(idx * 32) +: 32] <=
                                dram_c[(uop_sched_tensor * RTL_SOFTMAX_LEN) + idx];
                            primitive_reduction_x_flat[(idx * 32) +: 32] <=
                                dram_c[(uop_sched_tensor * RTL_SOFTMAX_LEN) + idx];
                        end
                    end
                    UOP_VSUB: primitive_vector_op <= 3'd1;
                    UOP_VCLAMP: begin
                        primitive_vector_op <= 3'd5;
                        primitive_vector_clamp_low <= RTL_SOFTMAX_CLAMP_LOW;
                        primitive_vector_clamp_high <= RTL_SOFTMAX_CLAMP_HIGH;
                    end
                    UOP_VEXP: begin
                        primitive_sfu_op <= 2'd0;
                        primitive_sfu_x <= primitive_vector_a_flat[(uop_sched_buffer * 32) +: 32];
                    end
                    UOP_VREDSUM: begin
                        primitive_reduction_op <= 2'd1;
                        primitive_reduction_x_flat <= primitive_vector_a_flat;
                    end
                    UOP_VDIV: begin
                        primitive_sfu_op <= 2'd1;
                        primitive_sfu_x <= primitive_reduction_result[31:0];
                    end
                    UOP_VNORM: begin
                        primitive_vector_op <= 3'd3;
                        primitive_vector_shift <= RTL_SOFTMAX_NORMALIZE_SHIFT;
                    end
                    UOP_VSCALE_FIXED: begin
                        for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                            primitive_vector_a_flat[(idx * 32) +: 32] <=
                                dram_c[(uop_sched_tensor * RTL_SOFTMAX_LEN) + idx];
                        end
                        primitive_vector_op <= 3'd6;
                        primitive_vector_scalar <= RTL_SCORE_SCALE_MULTIPLIER;
                        primitive_vector_shift <= RTL_SCORE_SCALE_SHIFT;
                        primitive_vector_clamp_low <= 32'sh8000_0000;
                        primitive_vector_clamp_high <= 32'sh7fff_ffff;
                    end
                    default: begin
                    end
                endcase
                local_route_state <= LOCAL_ROUTE_START;
                end
            end
            case (local_route_state)
                LOCAL_ROUTE_IDLE: begin
                end
                LOCAL_ROUTE_START: begin
                    case (local_route_opcode)
                        UOP_VREDMAX, UOP_VREDSUM: primitive_reduction_start <= 1'b1;
                        UOP_VEXP, UOP_VDIV: primitive_sfu_start <= 1'b1;
                        default: primitive_softmax_start <= 1'b1;
                    endcase
                    local_route_state <= LOCAL_ROUTE_WAIT;
                end
                LOCAL_ROUTE_WAIT: begin
                    case (local_route_opcode)
                        UOP_VREDMAX: begin
                            if (primitive_reduction_done) begin
                                for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                                    primitive_vector_b_flat[(idx * 32) +: 32] <=
                                        primitive_reduction_result[31:0];
                                end
                                uop_sched_local_exec_done <= 1'b1;
                                local_route_state <= LOCAL_ROUTE_IDLE;
                            end
                        end
                        UOP_VSUB, UOP_VCLAMP: begin
                            if (primitive_vector_done) begin
                                primitive_vector_a_flat <= primitive_vector_y_flat;
                                uop_sched_local_exec_done <= 1'b1;
                                local_route_state <= LOCAL_ROUTE_IDLE;
                            end
                        end
                        UOP_VEXP: begin
                            if (primitive_sfu_done) begin
                                primitive_vector_a_flat[(local_route_lane * 32) +: 32] <= primitive_sfu_y;
                                uop_sched_local_exec_done <= 1'b1;
                                local_route_state <= LOCAL_ROUTE_IDLE;
                            end
                        end
                        UOP_VREDSUM: begin
                            if (primitive_reduction_done) begin
                                uop_sched_local_exec_done <= 1'b1;
                                local_route_state <= LOCAL_ROUTE_IDLE;
                            end
                        end
                        UOP_VDIV: begin
                            if (primitive_sfu_done) begin
                                primitive_vector_scalar <= primitive_sfu_y;
                                uop_sched_local_exec_done <= 1'b1;
                                local_route_state <= LOCAL_ROUTE_IDLE;
                            end
                        end
                        UOP_VNORM, UOP_VSCALE_FIXED: begin
                            if (primitive_vector_done) begin
                                for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                                    dram_c[(local_route_row * RTL_SOFTMAX_LEN) + idx] <=
                                        primitive_vector_y_flat[(idx * 32) +: 32];
                                end
                                uop_sched_local_exec_done <= 1'b1;
                                local_route_state <= LOCAL_ROUTE_IDLE;
                            end
                        end
                        default: begin
                            uop_sched_local_exec_done <= 1'b1;
                            local_route_state <= LOCAL_ROUTE_IDLE;
                        end
                    endcase
                end
                default: local_route_state <= LOCAL_ROUTE_IDLE;
            endcase
            if (start) begin
                compute_bank_active <= compute_bank_select;
            end
            if (uop_sched_done) begin
                done <= 1'b1;
            end
        end
    end

    task automatic run_load(input logic [3:0] tensor, input logic [3:0] buffer);
        integer l;
        begin
            if (tensor == TENSOR_X && buffer == BUF_VEC) begin
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
