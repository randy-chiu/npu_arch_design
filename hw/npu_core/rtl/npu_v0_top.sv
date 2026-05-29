module npu_v0_top #(
    parameter int CORE_HOST_LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        op,          // Reserved. Phase 0 execution is driven by uops.
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

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_FETCH,
        ST_MATMUL,
        ST_DONE
    } state_t;

    logic signed [7:0]  dram_a [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_a_bank1 [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_b [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_b_bank1 [0:RTL_MATMUL_ELEMS-1];
    logic signed [31:0] dram_c [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_x [0:RTL_SOFTMAX_LEN-1];
    logic [7:0]         dram_y [0:RTL_SOFTMAX_LEN-1];

    logic signed [7:0]  spad_a [0:RTL_MATMUL_ELEMS-1];
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
    logic [(RTL_MATMUL_ELEMS*8)-1:0]  matmul_a_flat;
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

    integer idx;
    integer host_lane_idx;
    integer host_read_lane_idx;
    logic [11:0] host_lane_addr;
    logic [11:0] host_read_lane_addr;

    genvar matmul_flat_idx;
    generate
        for (matmul_flat_idx = 0; matmul_flat_idx < RTL_MATMUL_ELEMS; matmul_flat_idx = matmul_flat_idx + 1) begin : gen_matmul_flat
            assign matmul_a_flat[(matmul_flat_idx * 8) +: 8] = spad_a[matmul_flat_idx];
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
        end else begin
            acc_clear_request <= 1'b0;
            for (host_lane_idx = 0; host_lane_idx < CORE_HOST_LANES; host_lane_idx = host_lane_idx + 1) begin
                if (host_we[host_lane_idx]) begin
                    host_lane_addr = host_addr + host_lane_idx[11:0];
                    if (host_lane_addr >= RTL_HOST_A_BASE &&
                        host_lane_addr < RTL_HOST_A_BASE + RTL_HOST_A_WORDS) begin
                        if (host_write_bank) begin
                            dram_a_bank1[host_lane_addr - RTL_HOST_A_BASE] <= host_wdata[(host_lane_idx * 32) +: 8];
                        end else begin
                            dram_a[host_lane_addr - RTL_HOST_A_BASE] <= host_wdata[(host_lane_idx * 32) +: 8];
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
                host_rdata[(host_read_lane_idx * 32) +: 32] = {24'h0, dram_y[host_read_lane_addr - RTL_HOST_Y_BASE]};
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
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        compute_bank_active <= compute_bank_select;
                        pc <= 4'h0;
                        state <= ST_FETCH;
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
