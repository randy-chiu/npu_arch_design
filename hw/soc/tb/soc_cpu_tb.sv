module soc_cpu_tb;
    localparam int CPU_SOC_TIMEOUT_CYCLES = 20000;

    logic clk;
    logic rst_n;
    logic [31:0] sim_status;
    logic cpu_trap;

    soc_cpu_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .sim_status(sim_status),
        .cpu_trap(cpu_trap)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        rst_n = 1'b0;
        repeat (8) @(posedge clk);
        rst_n = 1'b1;

        repeat (CPU_SOC_TIMEOUT_CYCLES) begin
            @(posedge clk);
            if (sim_status == 32'h0000_0001) begin
                $display("PASS PicoRV32 firmware-controlled SoC smoke test");
                $finish;
            end
            if (sim_status == 32'hffff_ffff || sim_status[31]) begin
                $display("FAIL firmware reported mismatch status=%h", sim_status);
                $fatal(1);
            end
            if (cpu_trap) begin
                $display("FAIL CPU trap");
                $fatal(1);
            end
        end

        $display("FAIL CPU firmware timeout status=%h", sim_status);
        $fatal(1);
    end
endmodule
