module soc_cpu_tb;
    `include "npu_v0_regs.svh"
    `include "soc_v0_addr.svh"

    localparam int CPU_SOC_TIMEOUT_CYCLES = 20000;
    localparam logic [3:0] DESC_READ_STATE = 4'd1;
    localparam logic [3:0] DESC_FETCH_PROGRAM_STATE = 4'd2;
    localparam logic [3:0] DESC_FETCH_INPUT0_STATE = 4'd3;
    localparam logic [3:0] DESC_FETCH_INPUT1_STATE = 4'd4;
    localparam logic [3:0] DESC_START_CORE_STATE = 4'd5;
    localparam logic [3:0] DESC_WAIT_CORE_STATE = 4'd6;
    localparam logic [3:0] DESC_WRITE_OUTPUT_STATE = 4'd7;
    localparam logic [3:0] DESC_DONE_STATE = 4'd8;
    localparam logic [1:0] CORE_FETCH_STATE = 2'd1;
    localparam logic [1:0] CORE_MATMUL_STATE = 2'd2;
    localparam logic [1:0] CORE_DONE_STATE = 2'd3;

    logic clk;
    logic rst_n;
    logic [31:0] sim_status;
    logic cpu_trap;
    integer perf_job_id;
    integer perf_active;
    integer perf_total_cycles;
    integer perf_desc_read_cycles;
    integer perf_fetch_program_cycles;
    integer perf_fetch_input0_cycles;
    integer perf_fetch_input1_cycles;
    integer perf_start_core_cycles;
    integer perf_wait_core_cycles;
    integer perf_write_output_cycles;
    integer perf_done_cycles;
    integer perf_core_cycles;
    integer perf_core_fetch_cycles;
    integer perf_core_matmul_cycles;
    integer perf_core_done_cycles;
    integer perf_sram_read_cycles;
    integer perf_sram_write_cycles;
    integer perf_core_host_write_cycles;
    integer perf_core_host_read_cycles;
    integer perf_desc_words;
    integer perf_program_words;
    integer perf_input0_words;
    integer perf_input1_words;
    integer perf_output_words;

    soc_cpu_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .sim_status(sim_status),
        .cpu_trap(cpu_trap)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            perf_job_id <= 0;
            perf_active <= 0;
            reset_perf_counters();
        end else begin
            if (!perf_active &&
                dut.npu_wrapper_req &&
                dut.npu_wrapper_we &&
                dut.npu_wrapper_addr == NPU_OPSCHED_CTRL &&
                dut.npu_wrapper_wdata[0]) begin
                perf_active <= 1;
                perf_job_id <= perf_job_id + 1;
                reset_perf_counters();
            end else if (perf_active) begin
                perf_total_cycles <= perf_total_cycles + 1;

                case (dut.u_npu_wrapper.desc_state)
                    DESC_READ_STATE: perf_desc_read_cycles <= perf_desc_read_cycles + 1;
                    DESC_FETCH_PROGRAM_STATE: perf_fetch_program_cycles <= perf_fetch_program_cycles + 1;
                    DESC_FETCH_INPUT0_STATE: perf_fetch_input0_cycles <= perf_fetch_input0_cycles + 1;
                    DESC_FETCH_INPUT1_STATE: perf_fetch_input1_cycles <= perf_fetch_input1_cycles + 1;
                    DESC_START_CORE_STATE: perf_start_core_cycles <= perf_start_core_cycles + 1;
                    DESC_WAIT_CORE_STATE: perf_wait_core_cycles <= perf_wait_core_cycles + 1;
                    DESC_WRITE_OUTPUT_STATE: perf_write_output_cycles <= perf_write_output_cycles + 1;
                    DESC_DONE_STATE: perf_done_cycles <= perf_done_cycles + 1;
                    default: begin
                    end
                endcase

                if (dut.u_npu_wrapper.desc_state == DESC_WAIT_CORE_STATE ||
                    dut.u_npu_wrapper.start_pulse ||
                    dut.u_npu_wrapper.u_npu.done) begin
                    perf_core_cycles <= perf_core_cycles + 1;
                    case (dut.u_npu_wrapper.u_npu.state)
                        CORE_FETCH_STATE: perf_core_fetch_cycles <= perf_core_fetch_cycles + 1;
                        CORE_MATMUL_STATE: perf_core_matmul_cycles <= perf_core_matmul_cycles + 1;
                        CORE_DONE_STATE: perf_core_done_cycles <= perf_core_done_cycles + 1;
                        default: begin
                        end
                    endcase
                end

                if (dut.u_npu_wrapper.desc_state == DESC_DONE_STATE) begin
                    print_perf_job();
                    perf_active <= 0;
                end

                if (dut.u_npu_wrapper.sram_req && !dut.u_npu_wrapper.sram_we) begin
                    perf_sram_read_cycles <= perf_sram_read_cycles + 1;
                    case (dut.u_npu_wrapper.desc_state)
                        DESC_READ_STATE: perf_desc_words <= perf_desc_words + 1;
                        DESC_FETCH_PROGRAM_STATE: perf_program_words <= perf_program_words + 1;
                        DESC_FETCH_INPUT0_STATE: perf_input0_words <= perf_input0_words + 1;
                        DESC_FETCH_INPUT1_STATE: perf_input1_words <= perf_input1_words + 1;
                        default: begin
                        end
                    endcase
                end

                if (dut.u_npu_wrapper.sram_req && dut.u_npu_wrapper.sram_we) begin
                    perf_sram_write_cycles <= perf_sram_write_cycles + 1;
                    if (dut.u_npu_wrapper.desc_state == DESC_WRITE_OUTPUT_STATE) begin
                        perf_output_words <= perf_output_words + 1;
                    end
                end

                if (dut.u_npu_wrapper.desc_host_we) begin
                    perf_core_host_write_cycles <= perf_core_host_write_cycles + 1;
                end

                if (dut.u_npu_wrapper.desc_state == DESC_WRITE_OUTPUT_STATE) begin
                    perf_core_host_read_cycles <= perf_core_host_read_cycles + 1;
                end
            end
        end
    end

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

    task automatic reset_perf_counters;
        begin
            perf_total_cycles <= 0;
            perf_desc_read_cycles <= 0;
            perf_fetch_program_cycles <= 0;
            perf_fetch_input0_cycles <= 0;
            perf_fetch_input1_cycles <= 0;
            perf_start_core_cycles <= 0;
            perf_wait_core_cycles <= 0;
            perf_write_output_cycles <= 0;
            perf_done_cycles <= 0;
            perf_core_cycles <= 0;
            perf_core_fetch_cycles <= 0;
            perf_core_matmul_cycles <= 0;
            perf_core_done_cycles <= 0;
            perf_sram_read_cycles <= 0;
            perf_sram_write_cycles <= 0;
            perf_core_host_write_cycles <= 0;
            perf_core_host_read_cycles <= 0;
            perf_desc_words <= 0;
            perf_program_words <= 0;
            perf_input0_words <= 0;
            perf_input1_words <= 0;
            perf_output_words <= 0;
        end
    endtask

    task automatic print_perf_job;
        begin
            if (dut.u_npu_wrapper.job_op_type == SOC_NPU_JOB_OP_MATMUL) begin
                $display("PERF_JOB {\"id\":%0d,\"name\":\"matmul\",\"total_cycles\":%0d,\"wrapper\":{\"desc_read\":%0d,\"fetch_program\":%0d,\"fetch_input0\":%0d,\"fetch_input1\":%0d,\"start_core\":%0d,\"wait_core\":%0d,\"write_output\":%0d,\"done\":%0d},\"core\":{\"total\":%0d,\"fetch\":%0d,\"matmul\":%0d,\"done\":%0d},\"movement\":{\"sram_read_cycles\":%0d,\"sram_write_cycles\":%0d,\"core_host_write_cycles\":%0d,\"core_host_read_cycles\":%0d,\"desc_words\":%0d,\"program_words\":%0d,\"input0_words\":%0d,\"input1_words\":%0d,\"output_words\":%0d}}",
                    perf_job_id,
                    perf_total_cycles,
                    perf_desc_read_cycles,
                    perf_fetch_program_cycles,
                    perf_fetch_input0_cycles,
                    perf_fetch_input1_cycles,
                    perf_start_core_cycles,
                    perf_wait_core_cycles,
                    perf_write_output_cycles,
                    perf_done_cycles,
                    perf_core_cycles,
                    perf_core_fetch_cycles,
                    perf_core_matmul_cycles,
                    perf_core_done_cycles,
                    perf_sram_read_cycles,
                    perf_sram_write_cycles,
                    perf_core_host_write_cycles,
                    perf_core_host_read_cycles,
                    perf_desc_words,
                    perf_program_words,
                    perf_input0_words,
                    perf_input1_words,
                    perf_output_words);
            end else if (dut.u_npu_wrapper.job_op_type == SOC_NPU_JOB_OP_SOFTMAX) begin
                $display("PERF_JOB {\"id\":%0d,\"name\":\"softmax\",\"total_cycles\":%0d,\"wrapper\":{\"desc_read\":%0d,\"fetch_program\":%0d,\"fetch_input0\":%0d,\"fetch_input1\":%0d,\"start_core\":%0d,\"wait_core\":%0d,\"write_output\":%0d,\"done\":%0d},\"core\":{\"total\":%0d,\"fetch\":%0d,\"matmul\":%0d,\"done\":%0d},\"movement\":{\"sram_read_cycles\":%0d,\"sram_write_cycles\":%0d,\"core_host_write_cycles\":%0d,\"core_host_read_cycles\":%0d,\"desc_words\":%0d,\"program_words\":%0d,\"input0_words\":%0d,\"input1_words\":%0d,\"output_words\":%0d}}",
                    perf_job_id,
                    perf_total_cycles,
                    perf_desc_read_cycles,
                    perf_fetch_program_cycles,
                    perf_fetch_input0_cycles,
                    perf_fetch_input1_cycles,
                    perf_start_core_cycles,
                    perf_wait_core_cycles,
                    perf_write_output_cycles,
                    perf_done_cycles,
                    perf_core_cycles,
                    perf_core_fetch_cycles,
                    perf_core_matmul_cycles,
                    perf_core_done_cycles,
                    perf_sram_read_cycles,
                    perf_sram_write_cycles,
                    perf_core_host_write_cycles,
                    perf_core_host_read_cycles,
                    perf_desc_words,
                    perf_program_words,
                    perf_input0_words,
                    perf_input1_words,
                    perf_output_words);
            end else begin
                $display("PERF_JOB {\"id\":%0d,\"name\":\"unknown\",\"total_cycles\":%0d,\"wrapper\":{\"desc_read\":%0d,\"fetch_program\":%0d,\"fetch_input0\":%0d,\"fetch_input1\":%0d,\"start_core\":%0d,\"wait_core\":%0d,\"write_output\":%0d,\"done\":%0d},\"core\":{\"total\":%0d,\"fetch\":%0d,\"matmul\":%0d,\"done\":%0d},\"movement\":{\"sram_read_cycles\":%0d,\"sram_write_cycles\":%0d,\"core_host_write_cycles\":%0d,\"core_host_read_cycles\":%0d,\"desc_words\":%0d,\"program_words\":%0d,\"input0_words\":%0d,\"input1_words\":%0d,\"output_words\":%0d}}",
                    perf_job_id,
                    perf_total_cycles,
                    perf_desc_read_cycles,
                    perf_fetch_program_cycles,
                    perf_fetch_input0_cycles,
                    perf_fetch_input1_cycles,
                    perf_start_core_cycles,
                    perf_wait_core_cycles,
                    perf_write_output_cycles,
                    perf_done_cycles,
                    perf_core_cycles,
                    perf_core_fetch_cycles,
                    perf_core_matmul_cycles,
                    perf_core_done_cycles,
                    perf_sram_read_cycles,
                    perf_sram_write_cycles,
                    perf_core_host_write_cycles,
                    perf_core_host_read_cycles,
                    perf_desc_words,
                    perf_program_words,
                    perf_input0_words,
                    perf_input1_words,
                    perf_output_words);
            end
        end
    endtask
endmodule
