module npu_v0_tb;
    `include "npu_v0_spec.svh"
    `include "npu_v0_tb_params.svh"

    logic clk;
    logic rst_n;
    logic start;
    logic op;
    logic done;
    logic [3:0] host_we;
    logic [11:0] host_addr;
    logic [127:0] host_wdata;
    logic [127:0] host_rdata;

    npu_v0_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .op(op),
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
        op = 1'b0;
        host_we = 4'b0000;
        host_addr = 12'h0;
        host_wdata = 128'h0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        run_matmul_fixture_test();
        run_softmax_fixture_test();
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

            for (i = 0; i < MATMUL_OUTPUT_COUNT; i = i + 1) begin
                if (dut.dram_c[i] !== expected_c[i]) begin
                    $display("FAIL matmul[%0d] actual=%0d expected=%0d", i, dut.dram_c[i], expected_c[i]);
                    $fatal(1);
                end
            end
        end
    endtask

    task automatic run_softmax_fixture_test;
        integer i;
        logic [7:0] expected_y [0:SOFTMAX_OUTPUT_COUNT-1];
        begin
            $readmemh(SOFTMAX_X_HEX, dut.dram_x);
            $readmemh(SOFTMAX_PROGRAM_HEX, dut.instr_mem);
            $readmemh(SOFTMAX_EXPECTED_Y_HEX, expected_y);

            launch_and_wait();

            for (i = 0; i < SOFTMAX_OUTPUT_COUNT; i = i + 1) begin
                if (dut.dram_y[i] !== expected_y[i]) begin
                    $display("FAIL softmax[%0d] actual=%0d expected=%0d", i, dut.dram_y[i], expected_y[i]);
                    $fatal(1);
                end
            end
        end
    endtask

    task automatic run_core_host_lane_smoke;
        begin
            repeat (2) @(posedge clk);
            host_write4(12'h000, 32'h0000_0011, 32'h0000_0022, 32'h0000_0033, 32'h0000_0044);
            @(posedge clk);
            if (dut.dram_a[0] !== 8'h11 || dut.dram_a[1] !== 8'h22 ||
                dut.dram_a[2] !== 8'h33 || dut.dram_a[3] !== 8'h44) begin
                $display("FAIL wide write to A window");
                $fatal(1);
            end

            dut.dram_c[0] = 32'h0000_0101;
            dut.dram_c[1] = 32'h0000_0202;
            dut.dram_c[2] = 32'h0000_0303;
            dut.dram_c[3] = 32'h0000_0404;
            host_read4_check(12'h200, 32'h0000_0101, 32'h0000_0202, 32'h0000_0303, 32'h0000_0404);
        end
    endtask
endmodule
