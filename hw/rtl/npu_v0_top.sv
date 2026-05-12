module npu_v0_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic        op,
    output logic        done,

    input  logic        host_we,
    input  logic [11:0] host_addr,
    input  logic [31:0] host_wdata,
    output logic [31:0] host_rdata
);
    localparam OP_MATMUL  = 1'b0;
    localparam OP_SOFTMAX = 1'b1;

    logic signed [7:0]  mat_a [0:63];
    logic signed [7:0]  mat_b [0:63];
    logic signed [31:0] mat_c [0:63];
    logic signed [7:0]  sm_x  [0:7];
    logic [7:0]         sm_y  [0:7];

    logic busy;
    logic active_op;
    logic [6:0] i_idx;
    logic [6:0] j_idx;
    logic [6:0] k_idx;
    logic signed [31:0] acc;

    integer idx;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (idx = 0; idx < 64; idx = idx + 1) begin
                mat_a[idx] <= '0;
                mat_b[idx] <= '0;
                mat_c[idx] <= '0;
            end
            for (idx = 0; idx < 8; idx = idx + 1) begin
                sm_x[idx] <= '0;
                sm_y[idx] <= '0;
            end
        end else if (host_we && !busy) begin
            if (host_addr < 12'h040) begin
                mat_a[host_addr[5:0]] <= host_wdata[7:0];
            end else if (host_addr >= 12'h100 && host_addr < 12'h140) begin
                mat_b[host_addr[5:0]] <= host_wdata[7:0];
            end else if (host_addr >= 12'h300 && host_addr < 12'h308) begin
                sm_x[host_addr[2:0]] <= host_wdata[7:0];
            end
        end
    end

    always_comb begin
        host_rdata = 32'h0;
        if (host_addr >= 12'h200 && host_addr < 12'h240) begin
            host_rdata = mat_c[host_addr[5:0]];
        end else if (host_addr >= 12'h380 && host_addr < 12'h388) begin
            host_rdata = {24'h0, sm_y[host_addr[2:0]]};
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            active_op <= OP_MATMUL;
            i_idx <= '0;
            j_idx <= '0;
            k_idx <= '0;
            acc <= '0;
        end else begin
            done <= 1'b0;

            if (start && !busy) begin
                busy <= 1'b1;
                active_op <= op;
                i_idx <= '0;
                j_idx <= '0;
                k_idx <= '0;
                acc <= '0;
                if (op == OP_SOFTMAX) begin
                    run_softmax();
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end else if (busy && active_op == OP_MATMUL) begin
                acc <= acc + mat_a[(i_idx * 8) + k_idx] * mat_b[(k_idx * 8) + j_idx];

                if (k_idx == 7) begin
                    mat_c[(i_idx * 8) + j_idx] <= acc + mat_a[(i_idx * 8) + k_idx] * mat_b[(k_idx * 8) + j_idx];
                    acc <= '0;
                    k_idx <= '0;
                    if (j_idx == 7) begin
                        j_idx <= '0;
                        if (i_idx == 7) begin
                            i_idx <= '0;
                            busy <= 1'b0;
                            done <= 1'b1;
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
        end
    end

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

    task automatic run_softmax;
        logic signed [7:0] max_v;
        logic [7:0] exp_v [0:7];
        logic [11:0] sum_v;
        integer s;
        begin
            max_v = sm_x[0];
            for (s = 1; s < 8; s = s + 1) begin
                if (sm_x[s] > max_v) max_v = sm_x[s];
            end
            sum_v = 12'd0;
            for (s = 0; s < 8; s = s + 1) begin
                exp_v[s] = exp_lut_q8(sm_x[s] - max_v);
                sum_v = sum_v + exp_v[s];
            end
            for (s = 0; s < 8; s = s + 1) begin
                sm_y[s] = (sum_v == 0) ? 8'd0 : ((exp_v[s] * 8'd255) / sum_v);
            end
        end
    endtask
endmodule

