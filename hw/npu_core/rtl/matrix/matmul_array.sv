module matmul_array #(
    parameter int M = 8,
    parameter int N = 8,
    parameter int K = 8
) (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  logic mixed_u16s8_q15,
    output logic done,
    output logic perf_active,

    input  logic [(M*K*16)-1:0]  a_flat,
    input  logic [(K*N*8)-1:0]   b_flat,
    output logic [(M*N*32)-1:0]  result_flat
);
    localparam int OUT_ELEMS = M * N;

    logic active;
    logic mixed_mode_active;
    logic [$clog2(K)-1:0] k_idx;
    logic signed [31:0] result [0:OUT_ELEMS-1];
    integer i;
    integer j;
    integer r;

    assign perf_active = active;

    function automatic logic signed [7:0] a_s8_at(input int row, input int k);
        begin
            a_s8_at = a_flat[(((row * K) + k) * 16) +: 8];
        end
    endfunction

    function automatic logic [15:0] a_u16_at(input int row, input int k);
        begin
            a_u16_at = a_flat[(((row * K) + k) * 16) +: 16];
        end
    endfunction

    function automatic logic signed [7:0] b_at(input int k, input int col);
        begin
            b_at = b_flat[(((k * N) + col) * 8) +: 8];
        end
    endfunction

    genvar out_idx;
    generate
        for (out_idx = 0; out_idx < OUT_ELEMS; out_idx = out_idx + 1) begin : gen_result_flat
            assign result_flat[(out_idx * 32) +: 32] =
                mixed_mode_active ? (result[out_idx] >>> 15) : result[out_idx];
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            mixed_mode_active <= 1'b0;
            done <= 1'b0;
            k_idx <= '0;
            for (r = 0; r < OUT_ELEMS; r = r + 1) begin
                result[r] <= '0;
            end
        end else begin
            done <= 1'b0;
            if (start && !active) begin
                active <= 1'b1;
                mixed_mode_active <= mixed_u16s8_q15;
                k_idx <= '0;
                for (r = 0; r < OUT_ELEMS; r = r + 1) begin
                    result[r] <= '0;
                end
            end else if (active) begin
                for (i = 0; i < M; i = i + 1) begin
                    for (j = 0; j < N; j = j + 1) begin
                        result[(i * N) + j] <=
                            result[(i * N) + j] +
                            (mixed_mode_active ?
                                ($signed({1'b0, a_u16_at(i, k_idx)}) * b_at(k_idx, j)) :
                                (a_s8_at(i, k_idx) * b_at(k_idx, j)));
                    end
                end

                if (k_idx == K - 1) begin
                    active <= 1'b0;
                    done <= 1'b1;
                    k_idx <= '0;
                end else begin
                    k_idx <= k_idx + 1'b1;
                end
            end
        end
    end
endmodule
