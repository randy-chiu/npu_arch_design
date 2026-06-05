module primitive_engines_tb;
    import npu_transformer_v1_config_pkg::*;

    localparam int LANES = CFG_VECTOR_LANES;
    localparam int MAX_LEN = CFG_REDUCTION_MAX_LEN;
    localparam int DATA_WIDTH = CFG_VECTOR_DATA_WIDTH;
    localparam int REDUCTION_DATA_WIDTH = CFG_REDUCTION_DATA_WIDTH;
    localparam int REDUCTION_RESULT_WIDTH = CFG_REDUCTION_RESULT_WIDTH;
    localparam int SFU_DATA_WIDTH = CFG_SFU_DATA_WIDTH;

    localparam logic [2:0] VEC_ADD = CFG_VEC_ADD[2:0];
    localparam logic [2:0] VEC_SUB = CFG_VEC_SUB[2:0];
    localparam logic [2:0] VEC_MUL = CFG_VEC_MUL[2:0];
    localparam logic [2:0] VEC_SCALE = CFG_VEC_SCALE[2:0];
    localparam logic [2:0] VEC_REQUANT = CFG_VEC_REQUANT[2:0];
    localparam logic [2:0] VEC_CLAMP = CFG_VEC_CLAMP[2:0];
    localparam logic [2:0] VEC_REQUANT_V2 = CFG_VEC_REQUANT_V2[2:0];

    localparam logic [1:0] REDUCE_MAX = CFG_REDUCE_MAX[1:0];
    localparam logic [1:0] REDUCE_SUM = CFG_REDUCE_SUM[1:0];
    localparam logic [1:0] REDUCE_SUMSQ = CFG_REDUCE_SUMSQ[1:0];

    localparam logic [1:0] SFU_EXP = CFG_SFU_EXP[1:0];
    localparam logic [1:0] SFU_RECIP = CFG_SFU_RECIP[1:0];
    localparam logic [1:0] SFU_RSQRT = CFG_SFU_RSQRT[1:0];

    logic clk;
    logic rst_n;

    logic vector_start;
    logic [2:0] vector_op;
    logic [LANES-1:0] vector_valid_mask;
    logic signed [(LANES*DATA_WIDTH)-1:0] vector_a_flat;
    logic signed [(LANES*DATA_WIDTH)-1:0] vector_b_flat;
    logic signed [DATA_WIDTH-1:0] vector_scalar;
    logic signed [DATA_WIDTH-1:0] vector_clamp_low;
    logic signed [DATA_WIDTH-1:0] vector_clamp_high;
    logic [4:0] vector_shift;
    logic vector_done;
    logic vector_active;
    logic signed [(LANES*DATA_WIDTH)-1:0] vector_y_flat;

    localparam int SMALL_VECTOR_LANES = 4;
    localparam int SMALL_VECTOR_DATA_WIDTH = 16;
    logic small_vector_start;
    logic [2:0] small_vector_op;
    logic [SMALL_VECTOR_LANES-1:0] small_vector_valid_mask;
    logic signed [(SMALL_VECTOR_LANES*SMALL_VECTOR_DATA_WIDTH)-1:0] small_vector_a_flat;
    logic signed [(SMALL_VECTOR_LANES*SMALL_VECTOR_DATA_WIDTH)-1:0] small_vector_b_flat;
    logic signed [SMALL_VECTOR_DATA_WIDTH-1:0] small_vector_scalar;
    logic signed [SMALL_VECTOR_DATA_WIDTH-1:0] small_vector_clamp_low;
    logic signed [SMALL_VECTOR_DATA_WIDTH-1:0] small_vector_clamp_high;
    logic [4:0] small_vector_shift;
    logic small_vector_done;
    logic small_vector_active;
    logic signed [(SMALL_VECTOR_LANES*SMALL_VECTOR_DATA_WIDTH)-1:0] small_vector_y_flat;

    logic reduction_start;
    logic [1:0] reduction_op;
    logic [7:0] reduction_length;
    logic signed [(MAX_LEN*REDUCTION_DATA_WIDTH)-1:0] reduction_x_flat;
    logic reduction_done;
    logic reduction_active;
    logic signed [REDUCTION_RESULT_WIDTH-1:0] reduction_result;

    logic sfu_start;
    logic [1:0] sfu_op;
    logic signed [SFU_DATA_WIDTH-1:0] sfu_x;
    logic sfu_done;
    logic sfu_active;
    logic [SFU_DATA_WIDTH-1:0] sfu_y;

    transformer_primitive_engines u_transformer_primitive_engines (
        .clk(clk),
        .rst_n(rst_n),

        .vector_start(vector_start),
        .vector_op(vector_op),
        .vector_valid_mask(vector_valid_mask),
        .vector_a_flat(vector_a_flat),
        .vector_b_flat(vector_b_flat),
        .vector_scalar(vector_scalar),
        .vector_clamp_low(vector_clamp_low),
        .vector_clamp_high(vector_clamp_high),
        .vector_shift(vector_shift),
        .vector_done(vector_done),
        .vector_active(vector_active),
        .vector_y_flat(vector_y_flat),

        .reduction_start(reduction_start),
        .reduction_op(reduction_op),
        .reduction_length(reduction_length),
        .reduction_x_flat(reduction_x_flat),
        .reduction_done(reduction_done),
        .reduction_active(reduction_active),
        .reduction_result(reduction_result),

        .sfu_start(sfu_start),
        .sfu_op(sfu_op),
        .sfu_x(sfu_x),
        .sfu_done(sfu_done),
        .sfu_active(sfu_active),
        .sfu_y(sfu_y)
    );

    vector_engine #(
        .LANES(SMALL_VECTOR_LANES),
        .DATA_WIDTH(SMALL_VECTOR_DATA_WIDTH),
        .OP_VEC_ADD(CFG_VEC_ADD),
        .OP_VEC_SUB(CFG_VEC_SUB),
        .OP_VEC_MUL(CFG_VEC_MUL),
        .OP_VEC_SCALE(CFG_VEC_SCALE),
        .OP_VEC_REQUANT(CFG_VEC_REQUANT),
        .OP_VEC_CLAMP(CFG_VEC_CLAMP),
        .OP_VEC_REQUANT_V2(CFG_VEC_REQUANT_V2)
    ) u_small_vector_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(small_vector_start),
        .op(small_vector_op),
        .valid_mask(small_vector_valid_mask),
        .a_flat(small_vector_a_flat),
        .b_flat(small_vector_b_flat),
        .scalar(small_vector_scalar),
        .clamp_low(small_vector_clamp_low),
        .clamp_high(small_vector_clamp_high),
        .shift(small_vector_shift),
        .done(small_vector_done),
        .active(small_vector_active),
        .y_flat(small_vector_y_flat)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 1'b0;
        vector_start = 1'b0;
        small_vector_start = 1'b0;
        reduction_start = 1'b0;
        sfu_start = 1'b0;
        vector_a_flat = '0;
        vector_b_flat = '0;
        small_vector_a_flat = '0;
        small_vector_b_flat = '0;
        reduction_x_flat = '0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        run_vector_tests();
        run_vector_parameter_override_test();
        run_reduction_tests();
        run_sfu_tests();
        run_softmax_primitive_sequence_test();
        run_attention_softmax_sequence_test();
        run_rmsnorm_primitive_sequence_test();

        $display("PASS primitive engine RTL tests");
        $finish;
    end

    task automatic set_vec_a(input int lane, input logic signed [31:0] value);
        begin
            vector_a_flat[(lane * DATA_WIDTH) +: DATA_WIDTH] = value;
        end
    endtask

    task automatic set_vec_b(input int lane, input logic signed [31:0] value);
        begin
            vector_b_flat[(lane * DATA_WIDTH) +: DATA_WIDTH] = value;
        end
    endtask

    task automatic check_vec_y(input int lane, input logic signed [31:0] expected);
        logic signed [31:0] actual;
        begin
            actual = vector_y_flat[(lane * DATA_WIDTH) +: DATA_WIDTH];
            if (actual !== expected) begin
                $display("FAIL vector lane %0d actual=%0d expected=%0d", lane, actual, expected);
                $fatal(1);
            end
        end
    endtask

    task automatic launch_vector;
        begin
            vector_start = 1'b1;
            @(posedge clk);
            #1;
            if (!vector_done) begin
                $display("FAIL vector did not assert done");
                $fatal(1);
            end
            vector_start = 1'b0;
            @(posedge clk);
        end
    endtask

    task automatic run_vector_tests;
        integer i;
        begin
            vector_valid_mask = 8'hff;
            vector_shift = 5'd1;
            vector_scalar = 32'sd4;
            vector_clamp_low = -32'sd10;
            vector_clamp_high = 32'sd10;
            for (i = 0; i < LANES; i = i + 1) begin
                set_vec_a(i, i - 4);
                set_vec_b(i, 2 * i);
            end

            vector_op = VEC_SUB;
            launch_vector();
            check_vec_y(0, -32'sd4);
            check_vec_y(7, -32'sd11);

            vector_op = VEC_CLAMP;
            set_vec_a(0, -32'sd256);
            set_vec_a(1, -32'sd8);
            set_vec_a(2, 32'sd2);
            vector_clamp_low = -32'sd8;
            vector_clamp_high = 32'sd0;
            launch_vector();
            check_vec_y(0, -32'sd8);
            check_vec_y(1, -32'sd8);
            check_vec_y(2, 32'sd0);

            vector_op = VEC_SCALE;
            set_vec_a(0, 32'sd7);
            vector_scalar = 32'sd4;
            vector_shift = 5'd1;
            launch_vector();
            check_vec_y(0, 32'sd14);

            vector_op = VEC_MUL;
            set_vec_a(0, -32'sd3);
            set_vec_b(0, 32'sd5);
            launch_vector();
            check_vec_y(0, -32'sd15);

            vector_op = VEC_REQUANT;
            vector_shift = 5'd2;
            vector_clamp_low = -32'sd4;
            vector_clamp_high = 32'sd7;
            set_vec_a(0, -32'sd32);
            set_vec_a(1, -32'sd7);
            set_vec_a(2, 32'sd12);
            set_vec_a(3, 32'sd64);
            launch_vector();
            check_vec_y(0, -32'sd4);
            check_vec_y(1, -32'sd2);
            check_vec_y(2, 32'sd3);
            check_vec_y(3, 32'sd7);

            vector_op = VEC_REQUANT_V2;
            vector_scalar = 32'sd11585;
            vector_shift = 5'd15;
            vector_clamp_low = -32'sd1000;
            vector_clamp_high = 32'sd1000;
            set_vec_a(0, 32'sd1000);
            set_vec_a(1, -32'sd1000);
            set_vec_a(2, 32'sd1);
            set_vec_a(3, -32'sd1);
            launch_vector();
            check_vec_y(0, 32'sd354);
            check_vec_y(1, -32'sd354);
            check_vec_y(2, 32'sd0);
            check_vec_y(3, 32'sd0);

            vector_scalar = 32'sd2;
            vector_shift = 5'd0;
            vector_clamp_low = -32'sd7;
            vector_clamp_high = 32'sd7;
            set_vec_a(0, 32'sd5);
            set_vec_a(1, -32'sd5);
            launch_vector();
            check_vec_y(0, 32'sd7);
            check_vec_y(1, -32'sd7);
        end
    endtask

    task automatic set_small_vec_a(input int lane, input logic signed [15:0] value);
        begin
            small_vector_a_flat[(lane * SMALL_VECTOR_DATA_WIDTH) +: SMALL_VECTOR_DATA_WIDTH] = value;
        end
    endtask

    task automatic check_small_vec_y(input int lane, input logic signed [15:0] expected);
        logic signed [15:0] actual;
        begin
            actual = small_vector_y_flat[(lane * SMALL_VECTOR_DATA_WIDTH) +: SMALL_VECTOR_DATA_WIDTH];
            if (actual !== expected) begin
                $display("FAIL small vector lane %0d actual=%0d expected=%0d", lane, actual, expected);
                $fatal(1);
            end
        end
    endtask

    task automatic launch_small_vector;
        begin
            small_vector_start = 1'b1;
            @(posedge clk);
            #1;
            if (!small_vector_done) begin
                $display("FAIL small vector did not assert done");
                $fatal(1);
            end
            small_vector_start = 1'b0;
            @(posedge clk);
        end
    endtask

    task automatic run_vector_parameter_override_test;
        begin
            small_vector_valid_mask = 4'hf;
            small_vector_scalar = 16'sd0;
            small_vector_clamp_low = -16'sd16;
            small_vector_clamp_high = 16'sd16;
            small_vector_shift = 5'd1;
            set_small_vec_a(0, -16'sd40);
            set_small_vec_a(1, -16'sd3);
            set_small_vec_a(2, 16'sd14);
            set_small_vec_a(3, 16'sd80);
            small_vector_op = VEC_REQUANT;
            launch_small_vector();
            check_small_vec_y(0, -16'sd16);
            check_small_vec_y(1, -16'sd2);
            check_small_vec_y(2, 16'sd7);
            check_small_vec_y(3, 16'sd16);
        end
    endtask

    task automatic set_reduce_x(input int index, input logic signed [31:0] value);
        begin
            reduction_x_flat[(index * REDUCTION_DATA_WIDTH) +: REDUCTION_DATA_WIDTH] = value;
        end
    endtask

    task automatic launch_reduction;
        begin
            reduction_start = 1'b1;
            @(posedge clk);
            #1;
            if (!reduction_done) begin
                $display("FAIL reduction did not assert done");
                $fatal(1);
            end
            reduction_start = 1'b0;
            @(posedge clk);
        end
    endtask

    task automatic run_reduction_tests;
        begin
            reduction_x_flat = '0;
            set_reduce_x(0, -32'sd3);
            set_reduce_x(1, 32'sd5);
            set_reduce_x(2, 32'sd2);
            set_reduce_x(3, -32'sd7);
            reduction_length = 8'd4;

            reduction_op = REDUCE_MAX;
            launch_reduction();
            if (reduction_result !== 64'sd5) begin
                $display("FAIL REDUCE_MAX actual=%0d", reduction_result);
                $fatal(1);
            end

            reduction_op = REDUCE_SUM;
            launch_reduction();
            if (reduction_result !== -64'sd3) begin
                $display("FAIL REDUCE_SUM actual=%0d", reduction_result);
                $fatal(1);
            end

            reduction_op = REDUCE_SUMSQ;
            launch_reduction();
            if (reduction_result !== 64'sd87) begin
                $display("FAIL REDUCE_SUMSQ actual=%0d", reduction_result);
                $fatal(1);
            end
        end
    endtask

    task automatic launch_sfu;
        begin
            sfu_start = 1'b1;
            @(posedge clk);
            #1;
            if (!sfu_done) begin
                $display("FAIL SFU did not assert done");
                $fatal(1);
            end
            sfu_start = 1'b0;
            @(posedge clk);
        end
    endtask

    task automatic run_sfu_tests;
        logic [31:0] exp0;
        logic [31:0] exp_neg8;
        begin
            sfu_op = SFU_EXP;
            sfu_x = 32'sd0;
            launch_sfu();
            exp0 = sfu_y;
            if (exp0 !== 32'd32767) begin
                $display("FAIL SFU_EXP(0) actual=%0d", exp0);
                $fatal(1);
            end

            sfu_x = -32'sd256;
            launch_sfu();
            exp_neg8 = sfu_y;
            if (exp_neg8 >= exp0 || exp_neg8 !== 32'd11) begin
                $display("FAIL SFU_EXP(-256) actual=%0d", exp_neg8);
                $fatal(1);
            end

            sfu_op = SFU_RECIP;
            sfu_x = 32'sd256;
            launch_sfu();
            if (sfu_y !== 32'd65536) begin
                $display("FAIL SFU_RECIP actual=%0d", sfu_y);
                $fatal(1);
            end

            sfu_op = SFU_RSQRT;
            sfu_x = 32'sd256;
            launch_sfu();
            if (sfu_y !== 32'd1048576) begin
                $display("FAIL SFU_RSQRT actual=%0d", sfu_y);
                $fatal(1);
            end
        end
    endtask

    task automatic run_softmax_primitive_sequence_test;
        integer i;
        logic signed [31:0] row [0:3];
        logic signed [31:0] max_value;
        logic [31:0] exp_values [0:3];
        logic [31:0] reciprocal;
        begin
            row[0] = 32'sd32;
            row[1] = 32'sd0;
            row[2] = -32'sd32;
            row[3] = -32'sd512;

            reduction_x_flat = '0;
            for (i = 0; i < 4; i = i + 1) begin
                set_reduce_x(i, row[i]);
            end
            reduction_length = 8'd4;
            reduction_op = REDUCE_MAX;
            launch_reduction();
            max_value = reduction_result[31:0];
            if (max_value !== 32'sd32) begin
                $display("FAIL softmax sequence max actual=%0d", max_value);
                $fatal(1);
            end

            vector_a_flat = '0;
            vector_b_flat = '0;
            vector_valid_mask = 8'h0f;
            for (i = 0; i < 4; i = i + 1) begin
                set_vec_a(i, row[i]);
                set_vec_b(i, max_value);
            end
            vector_op = VEC_SUB;
            launch_vector();
            check_vec_y(0, 32'sd0);
            check_vec_y(1, -32'sd32);
            check_vec_y(2, -32'sd64);
            check_vec_y(3, -32'sd544);

            vector_a_flat = vector_y_flat;
            vector_clamp_low = -32'sd256;
            vector_clamp_high = 32'sd0;
            vector_op = VEC_CLAMP;
            launch_vector();
            check_vec_y(0, 32'sd0);
            check_vec_y(1, -32'sd32);
            check_vec_y(2, -32'sd64);
            check_vec_y(3, -32'sd256);

            reduction_x_flat = '0;
            for (i = 0; i < 4; i = i + 1) begin
                sfu_op = SFU_EXP;
                sfu_x = vector_y_flat[(i * DATA_WIDTH) +: DATA_WIDTH];
                launch_sfu();
                exp_values[i] = sfu_y;
                set_reduce_x(i, sfu_y);
            end
            if (exp_values[0] !== 32'd32767 || exp_values[1] !== 32'd12055 ||
                exp_values[2] !== 32'd4435 || exp_values[3] !== 32'd11) begin
                $display("FAIL softmax sequence exp values");
                $fatal(1);
            end

            reduction_length = 8'd4;
            reduction_op = REDUCE_SUM;
            launch_reduction();
            if (reduction_result !== 64'd49268) begin
                $display("FAIL softmax sequence exp sum actual=%0d", reduction_result);
                $fatal(1);
            end

            sfu_op = SFU_RECIP;
            sfu_x = reduction_result[31:0];
            launch_sfu();
            reciprocal = sfu_y;
            if (reciprocal !== 32'd340) begin
                $display("FAIL softmax sequence reciprocal actual=%0d", reciprocal);
                $fatal(1);
            end

            vector_a_flat = '0;
            for (i = 0; i < 4; i = i + 1) begin
                set_vec_a(i, exp_values[i]);
            end
            vector_scalar = reciprocal;
            vector_shift = 5'd9;
            vector_op = VEC_SCALE;
            launch_vector();
            check_vec_y(0, 32'sd21759);
            check_vec_y(1, 32'sd8005);
            check_vec_y(2, 32'sd2945);
            check_vec_y(3, 32'sd7);
        end
    endtask

    task automatic run_attention_softmax_sequence_test;
        integer i;
        logic signed [31:0] row [0:7];
        logic signed [31:0] max_value;
        logic [31:0] exp_values [0:7];
        logic [31:0] reciprocal;
        begin
            row[0] = 32'sd32;
            row[1] = 32'sd0;
            row[2] = -32'sd32;
            row[3] = -32'sd64;
            row[4] = -32'sd96;
            row[5] = -32'sd128;
            row[6] = -32'sd192;
            row[7] = -32'sd256;

            reduction_x_flat = '0;
            for (i = 0; i < 8; i = i + 1) begin
                set_reduce_x(i, row[i]);
            end
            reduction_length = 8'd8;
            reduction_op = REDUCE_MAX;
            launch_reduction();
            max_value = reduction_result[31:0];
            if (max_value !== 32'sd32) begin
                $display("FAIL attention softmax max actual=%0d", max_value);
                $fatal(1);
            end

            vector_a_flat = '0;
            vector_b_flat = '0;
            vector_valid_mask = 8'hff;
            for (i = 0; i < 8; i = i + 1) begin
                set_vec_a(i, row[i]);
                set_vec_b(i, max_value);
            end
            vector_op = VEC_SUB;
            launch_vector();

            vector_a_flat = vector_y_flat;
            vector_clamp_low = -32'sd256;
            vector_clamp_high = 32'sd0;
            vector_op = VEC_CLAMP;
            launch_vector();
            check_vec_y(0, 32'sd0);
            check_vec_y(1, -32'sd32);
            check_vec_y(2, -32'sd64);
            check_vec_y(3, -32'sd96);
            check_vec_y(4, -32'sd128);
            check_vec_y(5, -32'sd160);
            check_vec_y(6, -32'sd224);
            check_vec_y(7, -32'sd256);

            reduction_x_flat = '0;
            for (i = 0; i < 8; i = i + 1) begin
                sfu_op = SFU_EXP;
                sfu_x = vector_y_flat[(i * DATA_WIDTH) +: DATA_WIDTH];
                launch_sfu();
                exp_values[i] = sfu_y;
                set_reduce_x(i, sfu_y);
            end
            if (exp_values[0] !== 32'd32767 || exp_values[1] !== 32'd12055 ||
                exp_values[2] !== 32'd4435 || exp_values[3] !== 32'd1632 ||
                exp_values[4] !== 32'd600 || exp_values[5] !== 32'd221 ||
                exp_values[6] !== 32'd30 || exp_values[7] !== 32'd11) begin
                $display("FAIL attention softmax exp values");
                $fatal(1);
            end

            reduction_length = 8'd8;
            reduction_op = REDUCE_SUM;
            launch_reduction();
            if (reduction_result !== 64'd51751) begin
                $display("FAIL attention softmax exp sum actual=%0d", reduction_result);
                $fatal(1);
            end

            sfu_op = SFU_RECIP;
            sfu_x = reduction_result[31:0];
            launch_sfu();
            reciprocal = sfu_y;
            if (reciprocal !== 32'd324) begin
                $display("FAIL attention softmax reciprocal actual=%0d", reciprocal);
                $fatal(1);
            end

            vector_a_flat = '0;
            for (i = 0; i < 8; i = i + 1) begin
                set_vec_a(i, exp_values[i]);
            end
            vector_scalar = reciprocal;
            vector_shift = 5'd9;
            vector_op = VEC_SCALE;
            launch_vector();
            check_vec_y(0, 32'sd20735);
            check_vec_y(1, 32'sd7628);
            check_vec_y(2, 32'sd2806);
            check_vec_y(3, 32'sd1032);
            check_vec_y(4, 32'sd379);
            check_vec_y(5, 32'sd139);
            check_vec_y(6, 32'sd18);
            check_vec_y(7, 32'sd6);
        end
    endtask

    task automatic run_rmsnorm_primitive_sequence_test;
        integer i;
        logic [31:0] inv_rms;
        begin
            reduction_x_flat = '0;
            for (i = 0; i < 4; i = i + 1) begin
                set_reduce_x(i, 32'sd8);
            end
            reduction_length = 8'd4;
            reduction_op = REDUCE_SUMSQ;
            launch_reduction();
            if (reduction_result !== 64'd256) begin
                $display("FAIL RMSNorm sequence sumsq actual=%0d", reduction_result);
                $fatal(1);
            end

            sfu_op = SFU_RSQRT;
            sfu_x = reduction_result[31:0];
            launch_sfu();
            inv_rms = sfu_y;
            if (inv_rms !== 32'd1048576) begin
                $display("FAIL RMSNorm sequence rsqrt actual=%0d", inv_rms);
                $fatal(1);
            end

            vector_a_flat = '0;
            vector_valid_mask = 8'h0f;
            for (i = 0; i < 4; i = i + 1) begin
                set_vec_a(i, 32'sd8);
            end
            vector_scalar = inv_rms;
            vector_shift = 5'd20;
            vector_op = VEC_SCALE;
            launch_vector();
            for (i = 0; i < 4; i = i + 1) begin
                check_vec_y(i, 32'sd8);
            end
        end
    endtask
endmodule
