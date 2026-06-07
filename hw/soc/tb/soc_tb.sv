module soc_tb;
    `include "npu_v0_tb_params.svh"
    `include "npu_v0_regs.svh"
    `include "soc_v0_addr.svh"

    localparam logic [31:0] NPU_WRAPPER_CTRL = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_CTRL;
    localparam logic [31:0] NPU_WRAPPER_STATUS = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_STATUS;
    localparam logic [31:0] NPU_WRAPPER_PERF_CTRL = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_CTRL;
    localparam logic [31:0] NPU_WRAPPER_PERF_STATUS = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_STATUS;
    localparam logic [31:0] NPU_WRAPPER_PERF_TOTAL_CYCLES =
        SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_TOTAL_CYCLES;
    localparam logic [31:0] NPU_WRAPPER_PERF_CORE_MATMUL_CYCLES =
        SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_CORE_MATMUL_CYCLES;
    localparam logic [31:0] NPU_WRAPPER_A_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_A_BASE;
    localparam logic [31:0] NPU_WRAPPER_B_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_B_BASE;
    localparam logic [31:0] NPU_WRAPPER_C_BASE = SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_C_BASE;
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
        check_and_clear_perf_snapshot();
        bus_write(SOC_TEST_STATUS_BASE, SOC_TEST_STATUS_PASS_VALUE);
        bus_read(SOC_TEST_STATUS_BASE, actual_status);

        if (actual_status !== SOC_TEST_STATUS_PASS_VALUE ||
            sim_status !== SOC_TEST_STATUS_PASS_VALUE) begin
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
            while (timeout > 0 && status[NPU_OPSCHED_STATUS_DONE_BIT] != 1'b1) begin
                bus_read(NPU_WRAPPER_STATUS, status);
                timeout = timeout - 1;
            end
            if (status[NPU_OPSCHED_STATUS_DONE_BIT] != 1'b1) begin
                $display("FAIL opsched timeout status=%h", status);
                $fatal(1);
            end
        end
    endtask

    task automatic check_and_clear_perf_snapshot;
        logic [31:0] perf_status;
        logic [31:0] perf_total_cycles;
        logic [31:0] perf_core_matmul_cycles;
        begin
            bus_read(NPU_WRAPPER_PERF_STATUS, perf_status);
            bus_read(NPU_WRAPPER_PERF_TOTAL_CYCLES, perf_total_cycles);
            bus_read(NPU_WRAPPER_PERF_CORE_MATMUL_CYCLES, perf_core_matmul_cycles);
            if ((perf_status &
                 (NPU_OPSCHED_PERF_STATUS_VALID_MASK |
                  NPU_OPSCHED_PERF_STATUS_RUNNING_MASK |
                  NPU_OPSCHED_PERF_STATUS_OVERFLOW_MASK)) !== NPU_OPSCHED_PERF_STATUS_VALID_MASK ||
                perf_total_cycles == 32'h0000_0000 ||
                perf_core_matmul_cycles == 32'h0000_0000) begin
                $display(
                    "FAIL perf CSR readback status=%h total=%0d matmul=%0d",
                    perf_status,
                    perf_total_cycles,
                    perf_core_matmul_cycles
                );
                $fatal(1);
            end
            bus_write(NPU_WRAPPER_PERF_CTRL, NPU_OPSCHED_PERF_CTRL_CLEAR_MASK);
            bus_read(NPU_WRAPPER_PERF_STATUS, perf_status);
            bus_read(NPU_WRAPPER_PERF_TOTAL_CYCLES, perf_total_cycles);
            if ((perf_status &
                 (NPU_OPSCHED_PERF_STATUS_VALID_MASK |
                  NPU_OPSCHED_PERF_STATUS_RUNNING_MASK |
                  NPU_OPSCHED_PERF_STATUS_OVERFLOW_MASK)) !== 32'h0000_0000 ||
                perf_total_cycles !== 32'h0000_0000) begin
                $display(
                    "FAIL perf CSR clear status=%h total=%0d",
                    perf_status,
                    perf_total_cycles
                );
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

            bus_write(NPU_WRAPPER_CTRL, NPU_OPSCHED_CTRL_START_MASK);
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

endmodule
