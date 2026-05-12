module npu_v0_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        op,          // Reserved. Phase 0 execution is driven by uops.
    output logic        done,

    input  logic        host_we,
    input  logic [11:0] host_addr,
    input  logic [31:0] host_wdata,
    output logic [31:0] host_rdata
);
    localparam [3:0] UOP_LOAD    = 4'h1;
    localparam [3:0] UOP_STORE   = 4'h2;
    localparam [3:0] UOP_MATMUL  = 4'h3;
    localparam [3:0] UOP_VREDMAX = 4'h4;
    localparam [3:0] UOP_VSUB    = 4'h5;
    localparam [3:0] UOP_VEXP    = 4'h6;
    localparam [3:0] UOP_VREDSUM = 4'h7;
    localparam [3:0] UOP_VDIV    = 4'h8;
    localparam [3:0] UOP_HALT    = 4'hf;

    localparam [3:0] TENSOR_A = 4'h0;
    localparam [3:0] TENSOR_B = 4'h1;
    localparam [3:0] TENSOR_C = 4'h2;
    localparam [3:0] TENSOR_X = 4'h3;
    localparam [3:0] TENSOR_Y = 4'h4;

    localparam [3:0] BUF_SPAD_A = 4'h0;
    localparam [3:0] BUF_SPAD_B = 4'h1;
    localparam [3:0] BUF_ACC    = 4'h2;
    localparam [3:0] BUF_VEC    = 4'h3;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_FETCH,
        ST_MATMUL,
        ST_DONE
    } state_t;

    logic signed [7:0]  dram_a [0:63];
    logic signed [7:0]  dram_b [0:63];
    logic signed [31:0] dram_c [0:63];
    logic signed [7:0]  dram_x [0:7];
    logic [7:0]         dram_y [0:7];

    logic signed [7:0]  spad_a [0:63];
    logic signed [7:0]  spad_b [0:63];
    logic signed [31:0] acc_buf [0:63];
    logic signed [15:0] vec_buf [0:7];
    logic signed [15:0] scalar_max;
    logic [15:0]        scalar_sum;
    logic [31:0]        instr_mem [0:15];

    state_t state;
    logic [3:0] pc;
    logic [31:0] instr;
    logic [3:0] opcode;
    logic [3:0] arg0;
    logic [3:0] arg1;
    logic [6:0] i_idx;
    logic [6:0] j_idx;
    logic [6:0] k_idx;
    logic signed [31:0] acc;

    integer idx;

    assign opcode = instr[31:28];
    assign arg0 = instr[27:24];
    assign arg1 = instr[23:20];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (idx = 0; idx < 64; idx = idx + 1) begin
                dram_a[idx] <= '0;
                dram_b[idx] <= '0;
                dram_c[idx] <= '0;
                spad_a[idx] <= '0;
                spad_b[idx] <= '0;
                acc_buf[idx] <= '0;
            end
            for (idx = 0; idx < 8; idx = idx + 1) begin
                dram_x[idx] <= '0;
                dram_y[idx] <= '0;
                vec_buf[idx] <= '0;
            end
            for (idx = 0; idx < 16; idx = idx + 1) begin
                instr_mem[idx] <= '0;
            end
        end else if (host_we && state == ST_IDLE) begin
            if (host_addr < 12'h040) begin
                dram_a[host_addr[5:0]] <= host_wdata[7:0];
            end else if (host_addr >= 12'h100 && host_addr < 12'h140) begin
                dram_b[host_addr[5:0]] <= host_wdata[7:0];
            end else if (host_addr >= 12'h300 && host_addr < 12'h308) begin
                dram_x[host_addr[2:0]] <= host_wdata[7:0];
            end else if (host_addr >= 12'h400 && host_addr < 12'h410) begin
                instr_mem[host_addr[3:0]] <= host_wdata;
            end
        end
    end

    always @* begin
        host_rdata = 32'h0;
        if (host_addr >= 12'h200 && host_addr < 12'h240) begin
            host_rdata = dram_c[host_addr[5:0]];
        end else if (host_addr >= 12'h380 && host_addr < 12'h388) begin
            host_rdata = {24'h0, dram_y[host_addr[2:0]]};
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            pc <= '0;
            instr <= '0;
            i_idx <= '0;
            j_idx <= '0;
            k_idx <= '0;
            acc <= '0;
            scalar_max <= '0;
            scalar_sum <= '0;
        end else begin
            done <= 1'b0;
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
                    case (instr_mem[pc][31:28])
                        UOP_LOAD: begin
                            run_load(instr_mem[pc][27:24], instr_mem[pc][23:20]);
                        end
                        UOP_STORE: begin
                            run_store(instr_mem[pc][27:24], instr_mem[pc][23:20]);
                        end
                        UOP_MATMUL: begin
                            i_idx <= '0;
                            j_idx <= '0;
                            k_idx <= '0;
                            acc <= '0;
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
                    acc <= acc + spad_a[(i_idx * 8) + k_idx] * spad_b[(k_idx * 8) + j_idx];
                    if (k_idx == 7) begin
                        acc_buf[(i_idx * 8) + j_idx] <= acc + spad_a[(i_idx * 8) + k_idx] * spad_b[(k_idx * 8) + j_idx];
                        acc <= '0;
                        k_idx <= '0;
                        if (j_idx == 7) begin
                            j_idx <= '0;
                            if (i_idx == 7) begin
                                i_idx <= '0;
                                state <= ST_FETCH;
                            end else begin
                                i_idx <= i_idx + 1'b1;
                            end
                        end else begin
                            j_idx <= j_idx + 1'b1;
                        end
                    end else begin
                        k_idx <= k_idx + 1'b1;
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
                for (l = 0; l < 64; l = l + 1) spad_a[l] = dram_a[l];
            end else if (tensor == TENSOR_B && buffer == BUF_SPAD_B) begin
                for (l = 0; l < 64; l = l + 1) spad_b[l] = dram_b[l];
            end else if (tensor == TENSOR_X && buffer == BUF_VEC) begin
                for (l = 0; l < 8; l = l + 1) vec_buf[l] = {{8{dram_x[l][7]}}, dram_x[l]};
            end
        end
    endtask

    task automatic run_store(input logic [3:0] tensor, input logic [3:0] buffer);
        integer s;
        begin
            if (tensor == TENSOR_C && buffer == BUF_ACC) begin
                for (s = 0; s < 64; s = s + 1) dram_c[s] = acc_buf[s];
            end else if (tensor == TENSOR_Y && buffer == BUF_VEC) begin
                for (s = 0; s < 8; s = s + 1) dram_y[s] = vec_buf[s][7:0];
            end
        end
    endtask

    task automatic run_vredmax;
        integer v;
        begin
            scalar_max = vec_buf[0];
            for (v = 1; v < 8; v = v + 1) begin
                if (vec_buf[v] > scalar_max) scalar_max = vec_buf[v];
            end
        end
    endtask

    task automatic run_vsub;
        integer v;
        begin
            for (v = 0; v < 8; v = v + 1) begin
                vec_buf[v] = vec_buf[v] - scalar_max;
            end
        end
    endtask

    task automatic run_vexp;
        integer v;
        begin
            for (v = 0; v < 8; v = v + 1) begin
                vec_buf[v] = {8'h0, exp_lut_q8(vec_buf[v][8:0])};
            end
        end
    endtask

    task automatic run_vredsum;
        integer v;
        begin
            scalar_sum = 16'h0;
            for (v = 0; v < 8; v = v + 1) begin
                scalar_sum = scalar_sum + vec_buf[v][7:0];
            end
        end
    endtask

    task automatic run_vdiv;
        integer v;
        logic [23:0] norm_prod;
        logic [23:0] quotient;
        begin
            for (v = 0; v < 8; v = v + 1) begin
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
