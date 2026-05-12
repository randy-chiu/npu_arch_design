module npu_v0_tb;
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

        run_matmul_test();
        run_softmax_test();

        $display("PASS npu_v0 RTL smoke tests");
        $finish;
    end

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

    task automatic run_matmul_test;
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1) begin
                host_write(12'h000 + i[11:0], 32'd1);
                host_write(12'h100 + i[11:0], 32'd1);
            end

            @(posedge clk);
            op <= 1'b0;
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            wait(done == 1'b1);

            host_read_check(12'h200, 32'd8);
            host_read_check(12'h201, 32'd8);
            host_read_check(12'h23f, 32'd8);
        end
    endtask

    task automatic run_softmax_test;
        begin
            host_write(12'h300, 32'd0);
            host_write(12'h301, 32'd0);
            host_write(12'h302, 32'd0);
            host_write(12'h303, 32'd0);
            host_write(12'h304, 32'd0);
            host_write(12'h305, 32'd0);
            host_write(12'h306, 32'd0);
            host_write(12'h307, 32'd0);

            @(posedge clk);
            op <= 1'b1;
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            wait(done == 1'b1);

            host_read_check(12'h380, 32'd31);
            host_read_check(12'h387, 32'd31);
        end
    endtask
endmodule
