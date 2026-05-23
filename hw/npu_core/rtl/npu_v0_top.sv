module npu_v0_top #(
    parameter int CORE_HOST_LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        op,          // Reserved. Phase 0 execution is driven by uops.
    output logic        done,

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
    logic signed [7:0]  dram_b [0:RTL_MATMUL_ELEMS-1];
    logic signed [31:0] dram_c [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  dram_x [0:RTL_SOFTMAX_LEN-1];
    logic [7:0]         dram_y [0:RTL_SOFTMAX_LEN-1];

    logic signed [7:0]  spad_a [0:RTL_MATMUL_ELEMS-1];
    logic signed [7:0]  spad_b [0:RTL_MATMUL_ELEMS-1];
    logic signed [31:0] acc_buf [0:RTL_MATMUL_ELEMS-1];
    logic signed [15:0] vec_buf [0:RTL_SOFTMAX_LEN-1];
    logic signed [15:0] scalar_max;
    logic [15:0]        scalar_sum;
    logic [31:0]        instr_mem [0:15];

    state_t state;
    logic [3:0] pc;
    logic [31:0] instr;
    logic [3:0] opcode;
    logic [3:0] arg0;
    logic [3:0] arg1;
    logic matmul_start;
    logic matmul_done;
    logic matmul_accumulate_enable;
    logic [(RTL_MATMUL_ELEMS*8)-1:0]  matmul_a_flat;
    logic [(RTL_MATMUL_ELEMS*8)-1:0]  matmul_b_flat;
    logic [(RTL_MATMUL_ELEMS*32)-1:0] matmul_result_flat;

    integer idx;
    integer commit_idx;
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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (idx = 0; idx < RTL_MATMUL_ELEMS; idx = idx + 1) begin
                dram_a[idx] <= '0;
                dram_b[idx] <= '0;
                dram_c[idx] <= '0;
                spad_a[idx] <= '0;
                spad_b[idx] <= '0;
                acc_buf[idx] <= '0;
            end
            for (idx = 0; idx < RTL_SOFTMAX_LEN; idx = idx + 1) begin
                dram_x[idx] <= '0;
                dram_y[idx] <= '0;
                vec_buf[idx] <= '0;
            end
            for (idx = 0; idx < 16; idx = idx + 1) begin
                instr_mem[idx] <= '0;
            end
            matmul_accumulate_enable <= 1'b0;
        end else if (state == ST_IDLE) begin
            for (host_lane_idx = 0; host_lane_idx < CORE_HOST_LANES; host_lane_idx = host_lane_idx + 1) begin
                if (host_we[host_lane_idx]) begin
                    host_lane_addr = host_addr + host_lane_idx[11:0];
                    if (host_lane_addr < 12'h040) begin
                        dram_a[host_lane_addr[5:0]] <= host_wdata[(host_lane_idx * 32) +: 8];
                    end else if (host_lane_addr >= 12'h100 && host_lane_addr < 12'h140) begin
                        dram_b[host_lane_addr[5:0]] <= host_wdata[(host_lane_idx * 32) +: 8];
                    end else if (host_lane_addr >= 12'h300 && host_lane_addr < 12'h308) begin
                        dram_x[host_lane_addr[2:0]] <= host_wdata[(host_lane_idx * 32) +: 8];
                    end else if (host_lane_addr >= 12'h400 && host_lane_addr < 12'h410) begin
                        instr_mem[host_lane_addr[3:0]] <= host_wdata[(host_lane_idx * 32) +: 32];
                    end else if (host_lane_addr == 12'h500) begin
                        matmul_accumulate_enable <= host_wdata[(host_lane_idx * 32)];
                        if (host_wdata[(host_lane_idx * 32) + 1]) begin
                            for (idx = 0; idx < RTL_MATMUL_ELEMS; idx = idx + 1) begin
                                acc_buf[idx] <= '0;
                            end
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
            if (host_read_lane_addr >= 12'h200 && host_read_lane_addr < 12'h240) begin
                host_rdata[(host_read_lane_idx * 32) +: 32] = dram_c[host_read_lane_addr[5:0]];
            end else if (host_read_lane_addr >= 12'h380 && host_read_lane_addr < 12'h388) begin
                host_rdata[(host_read_lane_idx * 32) +: 32] = {24'h0, dram_y[host_read_lane_addr[2:0]]};
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
        end else begin
            done <= 1'b0;
            matmul_start <= 1'b0;
            case (state)
                ST_IDLE: begin
                    if (start) begin
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
                        for (commit_idx = 0; commit_idx < RTL_MATMUL_ELEMS; commit_idx = commit_idx + 1) begin
                            if (matmul_accumulate_enable) begin
                                acc_buf[commit_idx] <= acc_buf[commit_idx] + $signed(matmul_result_flat[(commit_idx * 32) +: 32]);
                            end else begin
                                acc_buf[commit_idx] <= $signed(matmul_result_flat[(commit_idx * 32) +: 32]);
                            end
                        end
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
                for (l = 0; l < RTL_MATMUL_ELEMS; l = l + 1) spad_a[l] = dram_a[l];
            end else if (tensor == TENSOR_B && buffer == BUF_SPAD_B) begin
                for (l = 0; l < RTL_MATMUL_ELEMS; l = l + 1) spad_b[l] = dram_b[l];
            end else if (tensor == TENSOR_X && buffer == BUF_VEC) begin
                for (l = 0; l < RTL_SOFTMAX_LEN; l = l + 1) vec_buf[l] = {{8{dram_x[l][7]}}, dram_x[l]};
            end
        end
    endtask

    task automatic run_store(input logic [3:0] tensor, input logic [3:0] buffer);
        integer s;
        begin
            if (tensor == TENSOR_C && buffer == BUF_ACC) begin
                for (s = 0; s < RTL_MATMUL_ELEMS; s = s + 1) dram_c[s] = acc_buf[s];
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
