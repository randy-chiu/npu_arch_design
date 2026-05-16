module soc_tb;
    `include "npu_v0_tb_params.svh"
    `include "npu_v0_regs.svh"
    `include "soc_v0_addr.svh"

    localparam logic [31:0] NPU_WRAPPER_CTRL = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_CTRL;
    localparam logic [31:0] NPU_WRAPPER_STATUS = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_STATUS;
    localparam logic [31:0] NPU_WRAPPER_A_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_A_BASE;
    localparam logic [31:0] NPU_WRAPPER_B_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_B_BASE;
    localparam logic [31:0] NPU_WRAPPER_C_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_C_BASE;
    localparam logic [31:0] NPU_WRAPPER_X_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_X_BASE;
    localparam logic [31:0] NPU_WRAPPER_Y_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_Y_BASE;
    localparam logic [31:0] NPU_WRAPPER_PROGRAM_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PROGRAM_BASE;

    logic clk;
    logic rst_n;
    logic cpu_req;
    logic cpu_we;
    logic [31:0] cpu_addr;
    logic [31:0] cpu_wdata;
    logic [31:0] cpu_rdata;
    logic cpu_ready;
    logic [31:0] sim_status;

    logic [31:0] matmul_a [0:63];
    logic [31:0] matmul_b [0:63];
    logic [31:0] matmul_program [0:15];
    logic [31:0] matmul_expected_c [0:63];
    logic [31:0] softmax_x [0:7];
    logic [31:0] softmax_program [0:15];
    logic [31:0] softmax_expected_y [0:7];
    logic [31:0] actual_status;

    soc_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .cpu_req(cpu_req),
        .cpu_we(cpu_we),
        .cpu_addr(cpu_addr),
        .cpu_wdata(cpu_wdata),
        .cpu_rdata(cpu_rdata),
        .cpu_ready(cpu_ready),
        .sim_status(sim_status)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 1'b0;
        cpu_req = 1'b0;
        cpu_we = 1'b0;
        cpu_addr = 32'h0000_0000;
        cpu_wdata = 32'h0000_0000;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        run_cpu_controlled_matmul();
        run_cpu_controlled_softmax();
        bus_write(SOC_TEST_STATUS_BASE, 32'h0000_0001);
        bus_read(SOC_TEST_STATUS_BASE, actual_status);

        if (actual_status !== 32'h0000_0001 || sim_status !== 32'h0000_0001) begin
            $display("FAIL test status read=%h sim=%h expected=00000001", actual_status, sim_status);
            $fatal(1);
        end

        $display("PASS minimal SoC opsched smoke test");
        $finish;
    end

    task automatic bus_write(input logic [31:0] addr, input logic [31:0] data);
        begin
            @(posedge clk);
            cpu_addr <= addr;
            cpu_wdata <= data;
            cpu_we <= 1'b1;
            cpu_req <= 1'b1;
            @(posedge clk);
            if (!cpu_ready) begin
                $display("FAIL bus write not ready addr=%h", addr);
                $fatal(1);
            end
            cpu_req <= 1'b0;
            cpu_we <= 1'b0;
        end
    endtask

    task automatic bus_read(input logic [31:0] addr, output logic [31:0] data);
        begin
            @(posedge clk);
            cpu_addr <= addr;
            cpu_we <= 1'b0;
            cpu_req <= 1'b1;
            @(posedge clk);
            if (!cpu_ready) begin
                $display("FAIL bus read not ready addr=%h", addr);
                $fatal(1);
            end
            data = cpu_rdata;
            cpu_req <= 1'b0;
        end
    endtask

    task automatic wait_opsched_done;
        integer timeout;
        logic [31:0] status;
        begin
            timeout = 2000;
            status = 32'h0;
            while (timeout > 0 && status[0] != 1'b1) begin
                bus_read(NPU_WRAPPER_STATUS, status);
                timeout = timeout - 1;
            end
            if (status[0] != 1'b1) begin
                $display("FAIL opsched timeout status=%h", status);
                $fatal(1);
            end
        end
    endtask

    task automatic run_cpu_controlled_matmul;
        integer i;
        logic [31:0] actual;
        begin
            $readmemh(MATMUL_A_HEX, matmul_a);
            $readmemh(MATMUL_B_HEX, matmul_b);
            $readmemh(MATMUL_PROGRAM_HEX, matmul_program);
            $readmemh(MATMUL_EXPECTED_C_HEX, matmul_expected_c);

            for (i = 0; i < MATMUL_OUTPUT_COUNT; i = i + 1) begin
                bus_write(NPU_WRAPPER_A_BASE + (i * 4), matmul_a[i]);
                bus_write(NPU_WRAPPER_B_BASE + (i * 4), matmul_b[i]);
            end
            for (i = 0; i < 16; i = i + 1) begin
                bus_write(NPU_WRAPPER_PROGRAM_BASE + (i * 4), matmul_program[i]);
            end

            bus_write(NPU_WRAPPER_CTRL, 32'h0000_0001);
            wait_opsched_done();

            for (i = 0; i < MATMUL_OUTPUT_COUNT; i = i + 1) begin
                bus_read(NPU_WRAPPER_C_BASE + (i * 4), actual);
                if (actual !== matmul_expected_c[i]) begin
                    $display(
                        "FAIL soc matmul[%0d] actual=%0d expected=%0d",
                        i,
                        actual,
                        matmul_expected_c[i]
                    );
                    $fatal(1);
                end
            end
        end
    endtask

    task automatic run_cpu_controlled_softmax;
        integer i;
        logic [31:0] actual;
        begin
            $readmemh(SOFTMAX_X_HEX, softmax_x);
            $readmemh(SOFTMAX_PROGRAM_HEX, softmax_program);
            $readmemh(SOFTMAX_EXPECTED_Y_HEX, softmax_expected_y);

            for (i = 0; i < SOFTMAX_OUTPUT_COUNT; i = i + 1) begin
                bus_write(NPU_WRAPPER_X_BASE + (i * 4), softmax_x[i]);
            end
            for (i = 0; i < 16; i = i + 1) begin
                bus_write(NPU_WRAPPER_PROGRAM_BASE + (i * 4), softmax_program[i]);
            end

            bus_write(NPU_WRAPPER_CTRL, 32'h0000_0001);
            wait_opsched_done();

            for (i = 0; i < SOFTMAX_OUTPUT_COUNT; i = i + 1) begin
                bus_read(NPU_WRAPPER_Y_BASE + (i * 4), actual);
                if (actual[7:0] !== softmax_expected_y[i][7:0]) begin
                    $display(
                        "FAIL soc softmax[%0d] actual=%0d expected=%0d",
                        i,
                        actual[7:0],
                        softmax_expected_y[i][7:0]
                    );
                    $fatal(1);
                end
            end
        end
    endtask
endmodule
