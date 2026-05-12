module npu_v0_tb;
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

    logic clk;
    logic rst_n;
    logic start;
    logic op;
    logic done;
    logic host_we;
    logic [11:0] host_addr;
    logic [31:0] host_wdata;
    logic [31:0] host_rdata;

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
        host_we = 1'b0;
        host_addr = 12'h0;
        host_wdata = 32'h0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        run_matmul_uop_test();
        run_softmax_uop_test();

        $display("PASS npu_v0 RTL micro-op smoke tests");
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
            host_wdata <= data;
            host_we <= 1'b1;
            @(posedge clk);
            host_we <= 1'b0;
        end
    endtask

    task automatic host_read_check(input logic [11:0] addr, input logic [31:0] expected);
        begin
            @(posedge clk);
            host_addr <= addr;
            @(posedge clk);
            if (host_rdata !== expected) begin
                $display("FAIL addr=%h actual=%0d expected=%0d", addr, host_rdata, expected);
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

    task automatic run_matmul_uop_test;
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1) begin
                host_write(12'h000 + i[11:0], 32'd1);
                host_write(12'h100 + i[11:0], 32'd1);
            end

            host_write(12'h400, uop(UOP_LOAD, TENSOR_A, BUF_SPAD_A));
            host_write(12'h401, uop(UOP_LOAD, TENSOR_B, BUF_SPAD_B));
            host_write(12'h402, uop(UOP_MATMUL, 4'h0, 4'h0));
            host_write(12'h403, uop(UOP_STORE, TENSOR_C, BUF_ACC));
            host_write(12'h404, uop(UOP_HALT, 4'h0, 4'h0));

            launch_and_wait();

            host_read_check(12'h200, 32'd8);
            host_read_check(12'h201, 32'd8);
            host_read_check(12'h23f, 32'd8);
        end
    endtask

    task automatic run_softmax_uop_test;
        begin
            host_write(12'h300, 32'd0);
            host_write(12'h301, 32'd0);
            host_write(12'h302, 32'd0);
            host_write(12'h303, 32'd0);
            host_write(12'h304, 32'd0);
            host_write(12'h305, 32'd0);
            host_write(12'h306, 32'd0);
            host_write(12'h307, 32'd0);

            host_write(12'h400, uop(UOP_LOAD, TENSOR_X, BUF_VEC));
            host_write(12'h401, uop(UOP_VREDMAX, BUF_VEC, 4'h0));
            host_write(12'h402, uop(UOP_VSUB, BUF_VEC, BUF_VEC));
            host_write(12'h403, uop(UOP_VEXP, BUF_VEC, BUF_VEC));
            host_write(12'h404, uop(UOP_VREDSUM, BUF_VEC, 4'h0));
            host_write(12'h405, uop(UOP_VDIV, BUF_VEC, BUF_VEC));
            host_write(12'h406, uop(UOP_STORE, TENSOR_Y, BUF_VEC));
            host_write(12'h407, uop(UOP_HALT, 4'h0, 4'h0));

            launch_and_wait();

            host_read_check(12'h380, 32'd31);
            host_read_check(12'h387, 32'd31);
        end
    endtask
endmodule
