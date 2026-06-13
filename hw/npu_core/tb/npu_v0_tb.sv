module npu_v0_tb;
    `include "npu_v0_spec.svh"
    `include "npu_v0_tb_params.svh"

    logic clk;
    logic rst_n;
    logic start;
    logic [1:0] op;
    logic done;
    logic [3:0] host_we;
    logic [11:0] host_addr;
    logic [127:0] host_wdata;
    logic [127:0] host_rdata;

    npu_v0_compute_cluster dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .op(op),
        .output_store_enable(1'b1),
        .row_mask_enable(1'b1),
        .done(done),
        .host_we(host_we),
        .host_addr(host_addr),
        .host_wdata(host_wdata),
        .host_rdata(host_rdata)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        op = 2'd0;
        host_we = 4'b0000;
        host_addr = 12'h0;
        host_wdata = 128'h0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        run_matmul_fixture_test();
        run_mixed_u16s8_q15_matmul_test();
        run_attention_scale_mask_test();
        run_core_host_lane_smoke();

        $display("PASS npu_v0 RTL generated-fixture tests");
        $finish;
    end

    function automatic [31:0] uop(input logic [3:0] opcode, input logic [3:0] arg0, input logic [3:0] arg1);
        begin
            uop = {opcode, arg0, arg1, 20'h0};
        end
    endfunction

    task automatic host_write(input logic [11:0] addr, input logic [31:0] data);
        begin
            @(posedge clk);
            host_addr <= addr;
            host_wdata <= {96'h0, data};
            host_we <= 4'b0001;
            @(posedge clk);
            host_we <= 4'b0000;
        end
    endtask

    function automatic signed [31:0] scale_score_d8(input logic signed [31:0] value);
        logic signed [63:0] product;
        logic signed [63:0] scaled;
        begin
            product = value * 64'sd11585;
            if (product >= 0) scaled = (product + 64'sd16384) >>> 15;
            else scaled = -(((-product) + 64'sd16384) >>> 15);
            scale_score_d8 = scaled[31:0];
        end
    endfunction

    task automatic host_read_check(input logic [11:0] addr, input logic [31:0] expected);
        begin
            @(posedge clk);
            host_addr <= addr;
            @(posedge clk);
            if (host_rdata[31:0] !== expected) begin
                $display("FAIL addr=%h actual=%0d expected=%0d", addr, host_rdata[31:0], expected);
                $fatal(1);
            end
        end
    endtask

    task automatic host_write4(
        input logic [11:0] addr,
        input logic [31:0] data0,
        input logic [31:0] data1,
        input logic [31:0] data2,
        input logic [31:0] data3
    );
        begin
            @(posedge clk);
            host_addr <= addr;
            host_wdata <= {data3, data2, data1, data0};
            host_we <= 4'b1111;
            @(posedge clk);
            host_we <= 4'b0000;
        end
    endtask

    task automatic host_read4_check(
        input logic [11:0] addr,
        input logic [31:0] expected0,
        input logic [31:0] expected1,
        input logic [31:0] expected2,
        input logic [31:0] expected3
    );
        begin
            @(posedge clk);
            host_addr <= addr;
            @(posedge clk);
            if (host_rdata !== {expected3, expected2, expected1, expected0}) begin
                $display(
                    "FAIL wide read addr=%h actual=%h expected=%h",
                    addr,
                    host_rdata,
                    {expected3, expected2, expected1, expected0}
                );
                $fatal(1);
            end
        end
    endtask

    task automatic launch_and_wait;
        begin
            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            wait(done == 1'b1);
            @(posedge clk);
        end
    endtask

    task automatic run_matmul_fixture_test;
        integer i;
        logic signed [31:0] expected_c [0:MATMUL_OUTPUT_COUNT-1];
        begin
            $readmemh(MATMUL_A_HEX, dut.dram_a);
            $readmemh(MATMUL_B_HEX, dut.dram_b);
            $readmemh(MATMUL_PROGRAM_HEX, dut.instr_mem);
            $readmemh(MATMUL_EXPECTED_C_HEX, expected_c);

            launch_and_wait();
            op <= 2'd0;

            for (i = 0; i < MATMUL_OUTPUT_COUNT; i = i + 1) begin
                if (dut.dram_c[i] !== expected_c[i]) begin
                    $display("FAIL matmul[%0d] actual=%0d expected=%0d", i, dut.dram_c[i], expected_c[i]);
                    $fatal(1);
                end
            end
        end
    endtask

    task automatic run_mixed_u16s8_q15_matmul_test;
        integer i;
        begin
            for (i = 0; i < RTL_MATMUL_ELEMS; i = i + 1) begin
                dut.dram_a[i] = 16'd16384;
                dut.dram_b[i] = 8'd2;
            end
            dut.instr_mem[0] = uop(UOP_LOAD, TENSOR_A, BUF_SPAD_A);
            dut.instr_mem[1] = uop(UOP_LOAD, TENSOR_B, BUF_SPAD_B);
            dut.instr_mem[2] = uop(UOP_MATMUL, 4'h0, 4'h0);
            dut.instr_mem[3] = uop(UOP_STORE, TENSOR_C, BUF_ACC);
            dut.instr_mem[4] = uop(UOP_HALT, 4'h0, 4'h0);
            op <= 2'd2;

            launch_and_wait();
            op <= 2'd0;

            for (i = 0; i < RTL_MATMUL_ELEMS; i = i + 1) begin
                if (dut.dram_c[i] !== 32'd8) begin
                    $display("FAIL mixed u16s8 matmul[%0d] actual=%0d expected=8", i, dut.dram_c[i]);
                    $fatal(1);
                end
            end
        end
    endtask

    task automatic run_attention_scale_mask_test;
        integer i;
        integer row;
        integer lane;
        logic signed [31:0] input_value;
        logic signed [31:0] expected_value;
        logic [7:0] expected_mask;
        begin
            dut.row_mask_words[0] = 32'h0f07_0301;
            dut.row_mask_words[1] = 32'hff7f_3f1f;
            for (i = 0; i < RTL_MATMUL_ELEMS; i = i + 1) begin
                input_value = (i - 32) * 97;
                dut.dram_c[i] = input_value;
            end
            for (i = 0; i < RTL_SOFTMAX_LEN; i = i + 1) begin
                dut.instr_mem[i] = uop(UOP_VSCALE_FIXED, i[3:0], 4'h0);
            end
            dut.instr_mem[RTL_SOFTMAX_LEN] = uop(UOP_HALT, 4'h0, 4'h0);
            op <= 2'd3;
            launch_and_wait();
            op <= 2'd0;

            for (i = 0; i < RTL_MATMUL_ELEMS; i = i + 1) begin
                row = i / RTL_SOFTMAX_LEN;
                lane = i % RTL_SOFTMAX_LEN;
                expected_mask = (row < 4) ?
                    dut.row_mask_words[0][(row * 8) +: 8] :
                    dut.row_mask_words[1][((row - 4) * 8) +: 8];
                input_value = (i - 32) * 97;
                expected_value = expected_mask[lane] ?
                    scale_score_d8(input_value) : RTL_SOFTMAX_NEG_INF;
                if (dut.dram_c[i] !== expected_value) begin
                    $display("FAIL scale_mask[%0d] actual=%0d expected=%0d", i, dut.dram_c[i], expected_value);
                    $fatal(1);
                end
            end
            dut.row_mask_words[0] = 32'hffff_ffff;
            dut.row_mask_words[1] = 32'hffff_ffff;
        end
    endtask

    task automatic run_core_host_lane_smoke;
        begin
            repeat (2) @(posedge clk);
            host_write4(RTL_HOST_A_BASE, 32'h0000_0011, 32'h0000_0022, 32'h0000_0033, 32'h0000_0044);
            @(posedge clk);
            if (dut.dram_a[0] !== 16'h0011 || dut.dram_a[1] !== 16'h0022 ||
                dut.dram_a[2] !== 16'h0033 || dut.dram_a[3] !== 16'h0044) begin
                $display("FAIL wide write to A window");
                $fatal(1);
            end

            dut.dram_c[0] = 32'h0000_0101;
            dut.dram_c[1] = 32'h0000_0202;
            dut.dram_c[2] = 32'h0000_0303;
            dut.dram_c[3] = 32'h0000_0404;
            host_read4_check(RTL_HOST_C_BASE, 32'h0000_0101, 32'h0000_0202, 32'h0000_0303, 32'h0000_0404);
        end
    endtask
endmodule
