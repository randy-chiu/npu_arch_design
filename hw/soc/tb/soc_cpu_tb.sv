module soc_cpu_tb;
    `include "npu_v0_regs.svh"
    `include "soc_v0_addr.svh"

    localparam int CPU_SOC_TIMEOUT_CYCLES = 20000000;

    logic clk;
    logic rst_n;
    logic [31:0] sim_status;
    logic cpu_trap;

    integer ref_job_id;
    integer ref_active;
    integer ref_total_cycles;
    integer ref_core_cycles;
    integer ref_core_matmul_cycles;
    integer ref_dm_active_cycles;
    integer ref_dm_setup_cycles;
    integer ref_dm_transfer_cycles;
    integer ref_dm_stall_cycles;
    integer ref_dm_words;
    integer ref_dm_read_words;
    integer ref_dm_write_words;
    integer ref_sram_read_words;
    integer ref_sram_write_words;
    integer ref_check_pending;
    integer expected_total_cycles;
    integer expected_core_cycles;
    integer expected_core_matmul_cycles;
    integer expected_dm_active_cycles;
    integer expected_dm_setup_cycles;
    integer expected_dm_transfer_cycles;
    integer expected_dm_stall_cycles;
    integer expected_dm_words;
    integer expected_dm_read_words;
    integer expected_dm_write_words;
    integer expected_sram_read_words;
    integer expected_sram_write_words;

    logic [31:0] csr_status;
    logic [31:0] csr_job_id;
    logic [31:0] csr_op_type;
    logic [31:0] csr_total_cycles;
    logic [31:0] csr_core_cycles;
    logic [31:0] csr_core_matmul_cycles;
    logic [31:0] csr_dm_active_cycles;
    logic [31:0] csr_dm_setup_cycles;
    logic [31:0] csr_dm_transfer_cycles;
    logic [31:0] csr_dm_stall_cycles;
    logic [31:0] csr_dm_words;
    logic [31:0] csr_dm_read_words;
    logic [31:0] csr_dm_write_words;
    logic [31:0] csr_sram_read_words;

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
            ref_job_id <= 0;
            ref_active <= 0;
            ref_check_pending <= 0;
            reset_reference_counters();
        end else begin
            if (ref_check_pending) begin
                check_perf_snapshot_reference();
                ref_check_pending <= 0;
            end

            if (!ref_active &&
                dut.npu_wrapper_req &&
                dut.npu_wrapper_we &&
                dut.npu_wrapper_addr == NPU_OPSCHED_CTRL &&
                dut.npu_wrapper_wdata[NPU_OPSCHED_CTRL_START_BIT]) begin
                ref_active <= 1;
                ref_job_id <= ref_job_id + 1;
                reset_reference_counters();
            end else if (ref_active) begin
                ref_total_cycles <= ref_total_cycles + 1;
                if (dut.u_npu_wrapper.core_perf_active) begin
                    ref_core_cycles <= ref_core_cycles + 1;
                end
                if (dut.u_npu_wrapper.core_perf_matmul_active) begin
                    ref_core_matmul_cycles <= ref_core_matmul_cycles + 1;
                end
                if (dut.u_npu_wrapper.mover_perf_active) begin
                    ref_dm_active_cycles <= ref_dm_active_cycles + 1;
                end
                if (dut.u_npu_wrapper.mover_perf_setup) begin
                    ref_dm_setup_cycles <= ref_dm_setup_cycles + 1;
                end
                if (dut.u_npu_wrapper.mover_perf_transfer) begin
                    ref_dm_transfer_cycles <= ref_dm_transfer_cycles + 1;
                    ref_dm_words <= ref_dm_words + dut.u_npu_wrapper.mover_perf_words;
                    if (dut.u_npu_wrapper.mover_store) begin
                        ref_dm_write_words <= ref_dm_write_words + dut.u_npu_wrapper.mover_perf_words;
                    end else begin
                        ref_dm_read_words <= ref_dm_read_words + dut.u_npu_wrapper.mover_perf_words;
                    end
                end
                if (dut.u_npu_wrapper.mover_perf_stall) begin
                    ref_dm_stall_cycles <= ref_dm_stall_cycles + 1;
                end
                if (dut.u_npu_wrapper.sram_req && dut.u_npu_wrapper.sram_we == '0) begin
                    if (dut.u_npu_wrapper.mover_perf_transfer) begin
                        ref_sram_read_words <= ref_sram_read_words +
                            dut.u_npu_wrapper.mover_perf_words;
                    end else begin
                        ref_sram_read_words <= ref_sram_read_words + 1;
                    end
                end
                if (dut.u_npu_wrapper.sram_req && dut.u_npu_wrapper.sram_we != '0) begin
                    ref_sram_write_words <= ref_sram_write_words +
                        count_lanes(dut.u_npu_wrapper.sram_we);
                end

                if (dut.u_npu_wrapper.perf_complete_event) begin
                    ref_active <= 0;
                    ref_check_pending <= 1;
                    expected_total_cycles <= ref_total_cycles;
                    expected_core_cycles <= ref_core_cycles;
                    expected_core_matmul_cycles <= ref_core_matmul_cycles;
                    expected_dm_active_cycles <= ref_dm_active_cycles;
                    expected_dm_setup_cycles <= ref_dm_setup_cycles;
                    expected_dm_transfer_cycles <= ref_dm_transfer_cycles;
                    expected_dm_stall_cycles <= ref_dm_stall_cycles;
                    expected_dm_words <= ref_dm_words;
                    expected_dm_read_words <= ref_dm_read_words;
                    expected_dm_write_words <= ref_dm_write_words;
                    expected_sram_read_words <= ref_sram_read_words;
                    expected_sram_write_words <= ref_sram_write_words;
                end
            end

            if (dut.npu_wrapper_req && !dut.npu_wrapper_we && dut.npu_wrapper_ready) begin
                case (dut.npu_wrapper_addr)
                    NPU_OPSCHED_PERF_STATUS: csr_status <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_JOB_ID: csr_job_id <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_OP_TYPE: csr_op_type <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_TOTAL_CYCLES: csr_total_cycles <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_CORE_ACTIVE_CYCLES: csr_core_cycles <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_CORE_MATMUL_CYCLES: csr_core_matmul_cycles <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_DATA_MOVER_ACTIVE_CYCLES: csr_dm_active_cycles <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_DATA_MOVER_SETUP_CYCLES: csr_dm_setup_cycles <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_DATA_MOVER_TRANSFER_CYCLES: csr_dm_transfer_cycles <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_DATA_MOVER_STALL_CYCLES: csr_dm_stall_cycles <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_DATA_MOVER_WORDS: csr_dm_words <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_DATA_MOVER_READ_WORDS: csr_dm_read_words <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_DATA_MOVER_WRITE_WORDS: csr_dm_write_words <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_SRAM_READ_WORDS: csr_sram_read_words <= dut.npu_wrapper_rdata;
                    NPU_OPSCHED_PERF_SRAM_WRITE_WORDS: emit_csr_perf_job(dut.npu_wrapper_rdata);
                    default: begin
                    end
                endcase
            end
        end
    end

    initial begin
        rst_n = 1'b0;
        repeat (8) @(posedge clk);
        rst_n = 1'b1;

        repeat (CPU_SOC_TIMEOUT_CYCLES) begin
            @(posedge clk);
            if (sim_status == SOC_TEST_STATUS_PASS_VALUE) begin
                $display("PASS PicoRV32 firmware-controlled SoC smoke test");
                $finish;
            end
            if (sim_status == SOC_TEST_STATUS_FAIL_VALUE ||
                (sim_status & SOC_TEST_STATUS_FAIL_CODE_FLAG_VALUE) != 32'h0000_0000) begin
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

    function automatic integer count_lanes(input logic [SOC_NPU_SRAM_LANES-1:0] lanes);
        integer lane;
        begin
            count_lanes = 0;
            for (lane = 0; lane < SOC_NPU_SRAM_LANES; lane = lane + 1) begin
                if (lanes[lane]) begin
                    count_lanes = count_lanes + 1;
                end
            end
        end
    endfunction

    task automatic reset_reference_counters;
        begin
            ref_total_cycles <= 0;
            ref_core_cycles <= 0;
            ref_core_matmul_cycles <= 0;
            ref_dm_active_cycles <= 0;
            ref_dm_setup_cycles <= 0;
            ref_dm_transfer_cycles <= 0;
            ref_dm_stall_cycles <= 0;
            ref_dm_words <= 0;
            ref_dm_read_words <= 0;
            ref_dm_write_words <= 0;
            ref_sram_read_words <= 0;
            ref_sram_write_words <= 0;
        end
    endtask

    task automatic check_perf_snapshot_reference;
        begin
            if (!dut.u_npu_wrapper.perf_snapshot_valid ||
                dut.u_npu_wrapper.perf_snapshot_overflow ||
                dut.u_npu_wrapper.perf_running ||
                dut.u_npu_wrapper.perf_snap_total_cycles !== expected_total_cycles ||
                dut.u_npu_wrapper.perf_snap_core_active_cycles !== expected_core_cycles ||
                dut.u_npu_wrapper.perf_snap_core_matmul_cycles !== expected_core_matmul_cycles ||
                dut.u_npu_wrapper.perf_snap_data_mover_active_cycles !== expected_dm_active_cycles ||
                dut.u_npu_wrapper.perf_snap_data_mover_setup_cycles !== expected_dm_setup_cycles ||
                dut.u_npu_wrapper.perf_snap_data_mover_transfer_cycles !== expected_dm_transfer_cycles ||
                dut.u_npu_wrapper.perf_snap_data_mover_stall_cycles !== expected_dm_stall_cycles ||
                dut.u_npu_wrapper.perf_snap_data_mover_words !== expected_dm_words ||
                dut.u_npu_wrapper.perf_snap_data_mover_read_words !== expected_dm_read_words ||
                dut.u_npu_wrapper.perf_snap_data_mover_write_words !== expected_dm_write_words ||
                dut.u_npu_wrapper.perf_snap_sram_read_words !== expected_sram_read_words ||
                dut.u_npu_wrapper.perf_snap_sram_write_words !== expected_sram_write_words) begin
                $display("FAIL perf CSR/reference mismatch job=%0d total=%0d/%0d mover=%0d/%0d",
                    ref_job_id, expected_total_cycles, dut.u_npu_wrapper.perf_snap_total_cycles,
                    expected_dm_words, dut.u_npu_wrapper.perf_snap_data_mover_words);
                $fatal(1);
            end
        end
    endtask

    task automatic emit_csr_perf_job(input logic [31:0] sram_write_words);
        begin
            if ((csr_status &
                 (NPU_OPSCHED_PERF_STATUS_VALID_MASK |
                  NPU_OPSCHED_PERF_STATUS_RUNNING_MASK |
                  NPU_OPSCHED_PERF_STATUS_OVERFLOW_MASK)) !==
                NPU_OPSCHED_PERF_STATUS_VALID_MASK) begin
                $display("FAIL invalid firmware-read perf CSR status job=%0d status=%h", csr_job_id, csr_status);
                $fatal(1);
            end
            if (csr_op_type == SOC_NPU_JOB_OP_MATMUL) begin
                print_csr_perf_job("matmul", sram_write_words);
            end else if (csr_op_type == SOC_NPU_JOB_OP_MATMUL_K_STREAM) begin
                print_csr_perf_job("matmul_k_stream", sram_write_words);
            end else if (csr_op_type == SOC_NPU_JOB_OP_SOFTMAX) begin
                print_csr_perf_job("softmax", sram_write_words);
            end else begin
                print_csr_perf_job("unknown", sram_write_words);
            end
        end
    endtask

    task automatic print_csr_perf_job(input string op_name, input logic [31:0] sram_write_words);
        begin
            $display("PERF_JOB {\"source\":\"architectural_perf_csr_snapshot\",\"job_id\":%0d,\"id\":%0d,\"name\":\"%s\",\"total_cycles\":%0d,\"core\":{\"total\":%0d,\"matmul\":%0d},\"data_mover\":{\"active_cycles\":%0d,\"setup_cycles\":%0d,\"transfer_cycles\":%0d,\"stall_cycles\":%0d,\"words\":%0d,\"read_words\":%0d,\"write_words\":%0d},\"sram\":{\"read_words\":%0d,\"write_words\":%0d}}",
                csr_job_id, csr_job_id, op_name, csr_total_cycles, csr_core_cycles,
                csr_core_matmul_cycles, csr_dm_active_cycles, csr_dm_setup_cycles,
                csr_dm_transfer_cycles, csr_dm_stall_cycles, csr_dm_words,
                csr_dm_read_words, csr_dm_write_words, csr_sram_read_words,
                sram_write_words);
        end
    endtask
endmodule
