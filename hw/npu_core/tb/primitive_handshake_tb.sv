module primitive_handshake_tb;
    localparam int LANES = 4;
    localparam int DATA_WIDTH = 32;
    localparam int MAX_LEN = 8;

    logic clk;
    logic rst_n;

    logic vector_cmd_valid;
    logic vector_cmd_ready;
    logic [2:0] vector_cmd_op;
    logic [LANES-1:0] vector_cmd_valid_mask;
    logic signed [(LANES*DATA_WIDTH)-1:0] vector_cmd_a_flat;
    logic signed [(LANES*DATA_WIDTH)-1:0] vector_cmd_b_flat;
    logic signed [DATA_WIDTH-1:0] vector_cmd_scalar;
    logic signed [DATA_WIDTH-1:0] vector_cmd_clamp_low;
    logic signed [DATA_WIDTH-1:0] vector_cmd_clamp_high;
    logic [4:0] vector_cmd_shift;
    logic vector_rsp_valid;
    logic vector_rsp_ready;
    logic signed [(LANES*DATA_WIDTH)-1:0] vector_rsp_y_flat;
    logic [31:0] vector_active_cycles;
    logic [31:0] vector_input_stall_cycles;
    logic [31:0] vector_output_stall_cycles;
    logic [31:0] vector_idle_cycles;
    logic [31:0] vector_accepted_ops;
    logic [31:0] vector_accepted_lane_ops;

    logic reduction_cmd_valid;
    logic reduction_cmd_ready;
    logic [1:0] reduction_cmd_op;
    logic [7:0] reduction_cmd_length;
    logic signed [(MAX_LEN*DATA_WIDTH)-1:0] reduction_cmd_x_flat;
    logic reduction_rsp_valid;
    logic reduction_rsp_ready;
    logic signed [63:0] reduction_rsp_result;
    logic [31:0] reduction_active_cycles;
    logic [31:0] reduction_input_stall_cycles;
    logic [31:0] reduction_output_stall_cycles;
    logic [31:0] reduction_idle_cycles;
    logic [31:0] reduction_accepted_ops;
    logic [31:0] reduction_accepted_element_ops;

    logic sfu_cmd_valid;
    logic sfu_cmd_ready;
    logic [1:0] sfu_cmd_op;
    logic signed [DATA_WIDTH-1:0] sfu_cmd_x;
    logic sfu_rsp_valid;
    logic sfu_rsp_ready;
    logic [DATA_WIDTH-1:0] sfu_rsp_y;
    logic [31:0] sfu_active_cycles;
    logic [31:0] sfu_input_stall_cycles;
    logic [31:0] sfu_output_stall_cycles;
    logic [31:0] sfu_idle_cycles;
    logic [31:0] sfu_exp_ops;
    logic [31:0] sfu_recip_ops;
    logic [31:0] sfu_rsqrt_ops;

    vector_engine_handshake #(
        .LANES(LANES),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_vector (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(vector_cmd_valid),
        .cmd_ready(vector_cmd_ready),
        .cmd_op(vector_cmd_op),
        .cmd_valid_mask(vector_cmd_valid_mask),
        .cmd_a_flat(vector_cmd_a_flat),
        .cmd_b_flat(vector_cmd_b_flat),
        .cmd_scalar(vector_cmd_scalar),
        .cmd_clamp_low(vector_cmd_clamp_low),
        .cmd_clamp_high(vector_cmd_clamp_high),
        .cmd_shift(vector_cmd_shift),
        .rsp_valid(vector_rsp_valid),
        .rsp_ready(vector_rsp_ready),
        .rsp_y_flat(vector_rsp_y_flat),
        .active_cycles(vector_active_cycles),
        .input_stall_cycles(vector_input_stall_cycles),
        .output_stall_cycles(vector_output_stall_cycles),
        .idle_cycles(vector_idle_cycles),
        .accepted_ops(vector_accepted_ops),
        .accepted_lane_ops(vector_accepted_lane_ops)
    );

    reduction_engine_handshake #(
        .MAX_LEN(MAX_LEN),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_reduction (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(reduction_cmd_valid),
        .cmd_ready(reduction_cmd_ready),
        .cmd_op(reduction_cmd_op),
        .cmd_length(reduction_cmd_length),
        .cmd_x_flat(reduction_cmd_x_flat),
        .rsp_valid(reduction_rsp_valid),
        .rsp_ready(reduction_rsp_ready),
        .rsp_result(reduction_rsp_result),
        .active_cycles(reduction_active_cycles),
        .input_stall_cycles(reduction_input_stall_cycles),
        .output_stall_cycles(reduction_output_stall_cycles),
        .idle_cycles(reduction_idle_cycles),
        .accepted_ops(reduction_accepted_ops),
        .accepted_element_ops(reduction_accepted_element_ops)
    );

    sfu_lut_handshake u_sfu (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(sfu_cmd_valid),
        .cmd_ready(sfu_cmd_ready),
        .cmd_op(sfu_cmd_op),
        .cmd_x(sfu_cmd_x),
        .rsp_valid(sfu_rsp_valid),
        .rsp_ready(sfu_rsp_ready),
        .rsp_y(sfu_rsp_y),
        .active_cycles(sfu_active_cycles),
        .input_stall_cycles(sfu_input_stall_cycles),
        .output_stall_cycles(sfu_output_stall_cycles),
        .idle_cycles(sfu_idle_cycles),
        .exp_ops(sfu_exp_ops),
        .recip_ops(sfu_recip_ops),
        .rsqrt_ops(sfu_rsqrt_ops)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic check(input logic condition, input string message);
        begin
            if (!condition) begin
                $display("FAIL %s", message);
                $fatal(1);
            end
        end
    endtask

    task automatic wait_vector_response;
        begin
            while (!vector_rsp_valid) begin
                @(posedge clk);
                #1;
            end
        end
    endtask

    task automatic wait_reduction_response;
        begin
            while (!reduction_rsp_valid) begin
                @(posedge clk);
                #1;
            end
        end
    endtask

    task automatic wait_sfu_response;
        begin
            while (!sfu_rsp_valid) begin
                @(posedge clk);
                #1;
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        vector_cmd_valid = 1'b0;
        vector_rsp_ready = 1'b0;
        reduction_cmd_valid = 1'b0;
        reduction_rsp_ready = 1'b0;
        sfu_cmd_valid = 1'b0;
        sfu_rsp_ready = 1'b0;
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        #1;

        check(vector_cmd_ready && reduction_cmd_ready && sfu_cmd_ready,
              "all shims ready after reset");
        check(!vector_rsp_valid && !reduction_rsp_valid && !sfu_rsp_valid,
              "no stale response after reset");

        run_vector_handshake_test();
        run_reduction_handshake_test();
        run_sfu_handshake_and_reset_test();

        $display("PASS primitive valid/ready handshake tests");
        $finish;
    end

    task automatic run_vector_handshake_test;
        logic signed [(LANES*DATA_WIDTH)-1:0] held_result;
        begin
            vector_cmd_op = 3'd0;
            vector_cmd_valid_mask = 4'hf;
            vector_cmd_a_flat = {32'sd4, 32'sd3, 32'sd2, 32'sd1};
            vector_cmd_b_flat = {32'sd40, 32'sd30, 32'sd20, 32'sd10};
            vector_cmd_scalar = '0;
            vector_cmd_clamp_low = -32'sd128;
            vector_cmd_clamp_high = 32'sd127;
            vector_cmd_shift = '0;
            vector_cmd_valid = 1'b1;
            @(posedge clk);
            #1;
            check(!vector_cmd_ready, "vector blocks a second command while active");
            vector_cmd_valid = 1'b0;
            vector_cmd_a_flat = '0;
            vector_cmd_b_flat = '0;
            wait_vector_response();
            check(!vector_cmd_ready, "vector blocks input while response is stalled");
            check(vector_rsp_y_flat[0 +: DATA_WIDTH] == 32'sd11,
                  "vector captured command payload on acceptance");

            vector_cmd_a_flat = {32'sd8, 32'sd7, 32'sd6, 32'sd5};
            vector_cmd_b_flat = {32'sd4, 32'sd3, 32'sd2, 32'sd1};
            vector_cmd_valid = 1'b1;
            held_result = vector_rsp_y_flat;
            repeat (2) begin
                @(posedge clk);
                #1;
                check(vector_rsp_valid && vector_rsp_y_flat == held_result &&
                      !vector_cmd_ready,
                      "vector holds response and stalls stable next command");
            end
            vector_rsp_ready = 1'b1;
            @(posedge clk);
            #1;
            vector_rsp_ready = 1'b0;
            check(!vector_rsp_valid && vector_cmd_ready,
                  "vector retires response and exposes command ready");
            @(posedge clk);
            #1;
            vector_cmd_valid = 1'b0;
            wait_vector_response();
            check(vector_rsp_y_flat[0 +: DATA_WIDTH] == 32'sd6,
                  "vector accepts held command after input stall");
            vector_rsp_ready = 1'b1;
            @(posedge clk);
            #1;
            vector_rsp_ready = 1'b0;
            check(vector_accepted_ops == 32'd2 &&
                  vector_accepted_lane_ops == 32'd8,
                  "vector accepted-op counters");
            check(vector_active_cycles > 0 && vector_input_stall_cycles > 0 &&
                  vector_output_stall_cycles > 0 && vector_idle_cycles > 0,
                  "vector cycle counters");
        end
    endtask

    task automatic run_reduction_handshake_test;
        logic signed [63:0] held_result;
        begin
            reduction_cmd_op = 2'd1;
            reduction_cmd_length = 8'd4;
            reduction_cmd_x_flat = '0;
            reduction_cmd_x_flat[0 +: DATA_WIDTH] = 32'sd2;
            reduction_cmd_x_flat[DATA_WIDTH +: DATA_WIDTH] = -32'sd3;
            reduction_cmd_x_flat[(2*DATA_WIDTH) +: DATA_WIDTH] = 32'sd7;
            reduction_cmd_x_flat[(3*DATA_WIDTH) +: DATA_WIDTH] = 32'sd5;
            reduction_cmd_valid = 1'b1;
            @(posedge clk);
            #1;
            reduction_cmd_valid = 1'b0;
            wait_reduction_response();
            check(reduction_rsp_result == 64'sd11, "reduction response value");
            held_result = reduction_rsp_result;
            @(posedge clk);
            #1;
            check(reduction_rsp_valid && reduction_rsp_result == held_result,
                  "reduction holds stalled response stable");
            reduction_rsp_ready = 1'b1;
            @(posedge clk);
            #1;
            reduction_rsp_ready = 1'b0;
            check(!reduction_rsp_valid && reduction_cmd_ready,
                  "reduction retires response");
            check(reduction_accepted_ops == 32'd1 &&
                  reduction_accepted_element_ops == 32'd4,
                  "reduction accepted-op counters");
            check(reduction_active_cycles > 0 &&
                  reduction_output_stall_cycles > 0 &&
                  reduction_idle_cycles > 0,
                  "reduction cycle counters");
        end
    endtask

    task automatic run_sfu_handshake_and_reset_test;
        begin
            sfu_cmd_op = 2'd0;
            sfu_cmd_x = -32'sd32;
            sfu_cmd_valid = 1'b1;
            @(posedge clk);
            #1;
            sfu_cmd_valid = 1'b0;
            wait_sfu_response();
            check(sfu_rsp_y == 32'd12055, "SFU response value");
            check(!sfu_cmd_ready, "SFU blocks input while response is stalled");
            @(posedge clk);
            #1;
            check(sfu_rsp_valid && sfu_rsp_y == 32'd12055,
                  "SFU holds stalled response stable");
            check(sfu_exp_ops == 32'd1 && sfu_recip_ops == 0 &&
                  sfu_rsqrt_ops == 0 && sfu_active_cycles > 0 &&
                  sfu_output_stall_cycles > 0,
                  "SFU event counters");

            rst_n = 1'b0;
            @(posedge clk);
            #1;
            rst_n = 1'b1;
            #1;
            check(!sfu_rsp_valid && sfu_cmd_ready,
                  "reset clears SFU response without stale replay");
            check(sfu_active_cycles == 0 && sfu_input_stall_cycles == 0 &&
                  sfu_output_stall_cycles == 0 && sfu_idle_cycles == 0 &&
                  sfu_exp_ops == 0 && sfu_recip_ops == 0 &&
                  sfu_rsqrt_ops == 0,
                  "reset clears SFU counters");
        end
    endtask
endmodule
