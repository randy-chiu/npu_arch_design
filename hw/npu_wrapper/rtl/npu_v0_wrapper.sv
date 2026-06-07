module npu_v0_wrapper #(
    parameter int CORE_HOST_LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        bus_req,
    input  logic        bus_we,
    input  logic [11:0] bus_addr,
    input  logic [31:0] bus_wdata,
    output logic [31:0] bus_rdata,
    output logic        bus_ready,

    output logic        sram_req,
    output logic [CORE_HOST_LANES-1:0] sram_we,
    output logic [31:0] sram_addr,
    output logic [(CORE_HOST_LANES*32)-1:0] sram_wdata,
    input  logic [(CORE_HOST_LANES*32)-1:0] sram_rdata,
    input  logic        sram_ready
);
    `include "npu_v0_regs.svh"

    logic        cmd_valid;
    logic [31:0] cmd_desc_addr;
    logic        cmd_ready;
    logic        core_busy;
    logic        core_done;
    logic        done_latched;
    logic        irq_enable;
    logic        irq_status;
    logic [31:0] desc_addr;

    logic        legacy_window;
    logic [31:0] legacy_rdata;
    logic        legacy_ready;

    logic        core_perf_snapshot_valid;
    logic        core_perf_snapshot_overflow;
    logic [31:0] core_perf_snap_total_cycles;
    logic [31:0] core_perf_snap_core_active_cycles;
    logic [31:0] core_perf_snap_core_matmul_cycles;
    logic [31:0] core_perf_snap_data_mover_active_cycles;
    logic [31:0] core_perf_snap_data_mover_setup_cycles;
    logic [31:0] core_perf_snap_data_mover_transfer_cycles;
    logic [31:0] core_perf_snap_data_mover_stall_cycles;
    logic [31:0] core_perf_snap_data_mover_words;
    logic [31:0] core_perf_snap_data_mover_read_words;
    logic [31:0] core_perf_snap_data_mover_write_words;
    logic [31:0] core_perf_snap_sram_read_words;
    logic [31:0] core_perf_snap_sram_write_words;
    logic [31:0] core_perf_snap_job_id;
    logic [31:0] core_perf_snap_op_type;
    logic [31:0] core_perf_snap_cmd_active_cycles;
    logic [31:0] core_perf_snap_cmd_wait_cycles;
    logic [31:0] core_perf_snap_dm_compute_overlap_cycles;
    logic [31:0] core_perf_snap_uop_sched_active_cycles;
    logic [31:0] core_perf_snap_uop_sched_wait_cycles;
    logic [31:0] core_perf_snap_core_wait_data_cycles;
    logic [31:0] core_perf_snap_core_local_active_cycles;
    logic [31:0] core_perf_snap_dm_program_cycles;
    logic [31:0] core_perf_snap_dm_initial_input_cycles;
    logic [31:0] core_perf_snap_dm_prefetch_cycles;
    logic [31:0] core_perf_snap_dm_output_cycles;

    logic        perf_snapshot_valid;
    logic        perf_snapshot_overflow;
    logic [31:0] perf_snap_total_cycles;
    logic [31:0] perf_snap_core_active_cycles;
    logic [31:0] perf_snap_core_matmul_cycles;
    logic [31:0] perf_snap_data_mover_active_cycles;
    logic [31:0] perf_snap_data_mover_setup_cycles;
    logic [31:0] perf_snap_data_mover_transfer_cycles;
    logic [31:0] perf_snap_data_mover_stall_cycles;
    logic [31:0] perf_snap_data_mover_words;
    logic [31:0] perf_snap_data_mover_read_words;
    logic [31:0] perf_snap_data_mover_write_words;
    logic [31:0] perf_snap_sram_read_words;
    logic [31:0] perf_snap_sram_write_words;
    logic [31:0] perf_snap_job_id;
    logic [31:0] perf_snap_op_type;
    logic [31:0] perf_snap_cmd_active_cycles;
    logic [31:0] perf_snap_cmd_wait_cycles;
    logic [31:0] perf_snap_dm_compute_overlap_cycles;
    logic [31:0] perf_snap_uop_sched_active_cycles;
    logic [31:0] perf_snap_uop_sched_wait_cycles;
    logic [31:0] perf_snap_core_wait_data_cycles;
    logic [31:0] perf_snap_core_local_active_cycles;
    logic [31:0] perf_snap_dm_program_cycles;
    logic [31:0] perf_snap_dm_initial_input_cycles;
    logic [31:0] perf_snap_dm_prefetch_cycles;
    logic [31:0] perf_snap_dm_output_cycles;

    assign bus_ready = bus_req;
    assign cmd_desc_addr = desc_addr;
    assign cmd_valid =
        bus_req && bus_we && bus_addr == NPU_OPSCHED_CTRL &&
        bus_wdata[NPU_OPSCHED_CTRL_START_BIT] && cmd_ready;
    assign legacy_window =
        (bus_addr >= NPU_OPSCHED_A_BASE && bus_addr < NPU_OPSCHED_A_BASE + NPU_OPSCHED_A_BASE_SIZE_BYTES) ||
        (bus_addr >= NPU_OPSCHED_B_BASE && bus_addr < NPU_OPSCHED_B_BASE + NPU_OPSCHED_B_BASE_SIZE_BYTES) ||
        (bus_addr >= NPU_OPSCHED_C_BASE && bus_addr < NPU_OPSCHED_C_BASE + NPU_OPSCHED_C_BASE_SIZE_BYTES) ||
        (bus_addr >= NPU_OPSCHED_PROGRAM_BASE && bus_addr < NPU_OPSCHED_PROGRAM_BASE + NPU_OPSCHED_PROGRAM_BASE_SIZE_BYTES);

    npu_v0_core_system #(
        .CORE_HOST_LANES(CORE_HOST_LANES)
    ) u_core_system (
        .clk(clk),
        .rst_n(rst_n),
        .cmd_valid(cmd_valid),
        .cmd_desc_addr(cmd_desc_addr),
        .cmd_ready(cmd_ready),
        .core_busy(core_busy),
        .core_done(core_done),
        .legacy_req(bus_req && legacy_window),
        .legacy_we(bus_we),
        .legacy_addr(bus_addr),
        .legacy_wdata(bus_wdata),
        .legacy_rdata(legacy_rdata),
        .legacy_ready(legacy_ready),
        .sram_req(sram_req),
        .sram_we(sram_we),
        .sram_addr(sram_addr),
        .sram_wdata(sram_wdata),
        .sram_rdata(sram_rdata),
        .sram_ready(sram_ready),
        .perf_snapshot_valid(core_perf_snapshot_valid),
        .perf_snapshot_overflow(core_perf_snapshot_overflow),
        .perf_snap_total_cycles(core_perf_snap_total_cycles),
        .perf_snap_core_active_cycles(core_perf_snap_core_active_cycles),
        .perf_snap_core_matmul_cycles(core_perf_snap_core_matmul_cycles),
        .perf_snap_data_mover_active_cycles(core_perf_snap_data_mover_active_cycles),
        .perf_snap_data_mover_setup_cycles(core_perf_snap_data_mover_setup_cycles),
        .perf_snap_data_mover_transfer_cycles(core_perf_snap_data_mover_transfer_cycles),
        .perf_snap_data_mover_stall_cycles(core_perf_snap_data_mover_stall_cycles),
        .perf_snap_data_mover_words(core_perf_snap_data_mover_words),
        .perf_snap_data_mover_read_words(core_perf_snap_data_mover_read_words),
        .perf_snap_data_mover_write_words(core_perf_snap_data_mover_write_words),
        .perf_snap_sram_read_words(core_perf_snap_sram_read_words),
        .perf_snap_sram_write_words(core_perf_snap_sram_write_words),
        .perf_snap_job_id(core_perf_snap_job_id),
        .perf_snap_op_type(core_perf_snap_op_type),
        .perf_snap_cmd_active_cycles(core_perf_snap_cmd_active_cycles),
        .perf_snap_cmd_wait_cycles(core_perf_snap_cmd_wait_cycles),
        .perf_snap_dm_compute_overlap_cycles(core_perf_snap_dm_compute_overlap_cycles),
        .perf_snap_uop_sched_active_cycles(core_perf_snap_uop_sched_active_cycles),
        .perf_snap_uop_sched_wait_cycles(core_perf_snap_uop_sched_wait_cycles),
        .perf_snap_core_wait_data_cycles(core_perf_snap_core_wait_data_cycles),
        .perf_snap_core_local_active_cycles(core_perf_snap_core_local_active_cycles),
        .perf_snap_dm_program_cycles(core_perf_snap_dm_program_cycles),
        .perf_snap_dm_initial_input_cycles(core_perf_snap_dm_initial_input_cycles),
        .perf_snap_dm_prefetch_cycles(core_perf_snap_dm_prefetch_cycles),
        .perf_snap_dm_output_cycles(core_perf_snap_dm_output_cycles)
    );

    always_comb begin
        bus_rdata = 32'h0000_0000;
        case (bus_addr)
            NPU_OPSCHED_STATUS: begin
                bus_rdata[NPU_OPSCHED_STATUS_DONE_BIT] = done_latched;
                bus_rdata[NPU_OPSCHED_STATUS_BUSY_BIT] = core_busy;
                bus_rdata[NPU_OPSCHED_STATUS_IDLE_BIT] = !core_busy;
            end
            NPU_OPSCHED_VERSION: bus_rdata = 32'h0001_0000;
            NPU_OPSCHED_IRQ_ENABLE: bus_rdata[NPU_OPSCHED_IRQ_ENABLE_ENABLE_BIT] = irq_enable;
            NPU_OPSCHED_IRQ_STATUS: bus_rdata[NPU_OPSCHED_IRQ_STATUS_PENDING_BIT] = irq_status;
            NPU_OPSCHED_DESC_ADDR: bus_rdata = desc_addr;
            NPU_OPSCHED_PERF_STATUS: begin
                bus_rdata[NPU_OPSCHED_PERF_STATUS_VALID_BIT] = perf_snapshot_valid;
                bus_rdata[NPU_OPSCHED_PERF_STATUS_RUNNING_BIT] = core_busy;
                bus_rdata[NPU_OPSCHED_PERF_STATUS_OVERFLOW_BIT] = perf_snapshot_overflow;
            end
            NPU_OPSCHED_PERF_TOTAL_CYCLES: bus_rdata = perf_snap_total_cycles;
            NPU_OPSCHED_PERF_CORE_ACTIVE_CYCLES: bus_rdata = perf_snap_core_active_cycles;
            NPU_OPSCHED_PERF_CORE_MATMUL_CYCLES: bus_rdata = perf_snap_core_matmul_cycles;
            NPU_OPSCHED_PERF_DATA_MOVER_ACTIVE_CYCLES: bus_rdata = perf_snap_data_mover_active_cycles;
            NPU_OPSCHED_PERF_DATA_MOVER_SETUP_CYCLES: bus_rdata = perf_snap_data_mover_setup_cycles;
            NPU_OPSCHED_PERF_DATA_MOVER_TRANSFER_CYCLES: bus_rdata = perf_snap_data_mover_transfer_cycles;
            NPU_OPSCHED_PERF_DATA_MOVER_STALL_CYCLES: bus_rdata = perf_snap_data_mover_stall_cycles;
            NPU_OPSCHED_PERF_DATA_MOVER_WORDS: bus_rdata = perf_snap_data_mover_words;
            NPU_OPSCHED_PERF_SRAM_READ_WORDS: bus_rdata = perf_snap_sram_read_words;
            NPU_OPSCHED_PERF_SRAM_WRITE_WORDS: bus_rdata = perf_snap_sram_write_words;
            NPU_OPSCHED_PERF_JOB_ID: bus_rdata = perf_snap_job_id;
            NPU_OPSCHED_PERF_OP_TYPE: bus_rdata = perf_snap_op_type;
            NPU_OPSCHED_PERF_DATA_MOVER_READ_WORDS: bus_rdata = perf_snap_data_mover_read_words;
            NPU_OPSCHED_PERF_DATA_MOVER_WRITE_WORDS: bus_rdata = perf_snap_data_mover_write_words;
            NPU_OPSCHED_PERF_CMD_ACTIVE_CYCLES: bus_rdata = perf_snap_cmd_active_cycles;
            NPU_OPSCHED_PERF_CMD_WAIT_CYCLES: bus_rdata = perf_snap_cmd_wait_cycles;
            NPU_OPSCHED_PERF_DM_COMPUTE_OVERLAP_CYCLES: bus_rdata = perf_snap_dm_compute_overlap_cycles;
            NPU_OPSCHED_PERF_UOP_SCHED_ACTIVE_CYCLES: bus_rdata = perf_snap_uop_sched_active_cycles;
            NPU_OPSCHED_PERF_UOP_SCHED_WAIT_CYCLES: bus_rdata = perf_snap_uop_sched_wait_cycles;
            NPU_OPSCHED_PERF_CORE_WAIT_DATA_CYCLES: bus_rdata = perf_snap_core_wait_data_cycles;
            NPU_OPSCHED_PERF_CORE_LOCAL_ACTIVE_CYCLES: bus_rdata = perf_snap_core_local_active_cycles;
            NPU_OPSCHED_PERF_DM_PROGRAM_CYCLES: bus_rdata = perf_snap_dm_program_cycles;
            NPU_OPSCHED_PERF_DM_INITIAL_INPUT_CYCLES: bus_rdata = perf_snap_dm_initial_input_cycles;
            NPU_OPSCHED_PERF_DM_PREFETCH_CYCLES: bus_rdata = perf_snap_dm_prefetch_cycles;
            NPU_OPSCHED_PERF_DM_OUTPUT_CYCLES: bus_rdata = perf_snap_dm_output_cycles;
            default: if (legacy_window && !bus_we) bus_rdata = legacy_rdata;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            desc_addr <= 32'h0000_0000;
            done_latched <= 1'b0;
            irq_enable <= 1'b0;
            irq_status <= 1'b0;
            perf_snapshot_valid <= 1'b0;
            perf_snapshot_overflow <= 1'b0;
            perf_snap_total_cycles <= 32'h0000_0000;
            perf_snap_core_active_cycles <= 32'h0000_0000;
            perf_snap_core_matmul_cycles <= 32'h0000_0000;
            perf_snap_data_mover_active_cycles <= 32'h0000_0000;
            perf_snap_data_mover_setup_cycles <= 32'h0000_0000;
            perf_snap_data_mover_transfer_cycles <= 32'h0000_0000;
            perf_snap_data_mover_stall_cycles <= 32'h0000_0000;
            perf_snap_data_mover_words <= 32'h0000_0000;
            perf_snap_data_mover_read_words <= 32'h0000_0000;
            perf_snap_data_mover_write_words <= 32'h0000_0000;
            perf_snap_sram_read_words <= 32'h0000_0000;
            perf_snap_sram_write_words <= 32'h0000_0000;
            perf_snap_job_id <= 32'h0000_0000;
            perf_snap_op_type <= 32'h0000_0000;
            perf_snap_cmd_active_cycles <= 32'h0000_0000;
            perf_snap_cmd_wait_cycles <= 32'h0000_0000;
            perf_snap_dm_compute_overlap_cycles <= 32'h0000_0000;
            perf_snap_uop_sched_active_cycles <= 32'h0000_0000;
            perf_snap_uop_sched_wait_cycles <= 32'h0000_0000;
            perf_snap_core_wait_data_cycles <= 32'h0000_0000;
            perf_snap_core_local_active_cycles <= 32'h0000_0000;
            perf_snap_dm_program_cycles <= 32'h0000_0000;
            perf_snap_dm_initial_input_cycles <= 32'h0000_0000;
            perf_snap_dm_prefetch_cycles <= 32'h0000_0000;
            perf_snap_dm_output_cycles <= 32'h0000_0000;
        end else begin
            if (core_done) begin
                done_latched <= 1'b1;
                irq_status <= irq_enable;
                perf_snapshot_valid <= core_perf_snapshot_valid;
                perf_snapshot_overflow <= core_perf_snapshot_overflow;
                perf_snap_total_cycles <= core_perf_snap_total_cycles;
                perf_snap_core_active_cycles <= core_perf_snap_core_active_cycles;
                perf_snap_core_matmul_cycles <= core_perf_snap_core_matmul_cycles;
                perf_snap_data_mover_active_cycles <= core_perf_snap_data_mover_active_cycles;
                perf_snap_data_mover_setup_cycles <= core_perf_snap_data_mover_setup_cycles;
                perf_snap_data_mover_transfer_cycles <= core_perf_snap_data_mover_transfer_cycles;
                perf_snap_data_mover_stall_cycles <= core_perf_snap_data_mover_stall_cycles;
                perf_snap_data_mover_words <= core_perf_snap_data_mover_words;
                perf_snap_data_mover_read_words <= core_perf_snap_data_mover_read_words;
                perf_snap_data_mover_write_words <= core_perf_snap_data_mover_write_words;
                perf_snap_sram_read_words <= core_perf_snap_sram_read_words;
                perf_snap_sram_write_words <= core_perf_snap_sram_write_words;
                perf_snap_job_id <= core_perf_snap_job_id;
                perf_snap_op_type <= core_perf_snap_op_type;
                perf_snap_cmd_active_cycles <= core_perf_snap_cmd_active_cycles;
                perf_snap_cmd_wait_cycles <= core_perf_snap_cmd_wait_cycles;
                perf_snap_dm_compute_overlap_cycles <= core_perf_snap_dm_compute_overlap_cycles;
                perf_snap_uop_sched_active_cycles <= core_perf_snap_uop_sched_active_cycles;
                perf_snap_uop_sched_wait_cycles <= core_perf_snap_uop_sched_wait_cycles;
                perf_snap_core_wait_data_cycles <= core_perf_snap_core_wait_data_cycles;
                perf_snap_core_local_active_cycles <= core_perf_snap_core_local_active_cycles;
                perf_snap_dm_program_cycles <= core_perf_snap_dm_program_cycles;
                perf_snap_dm_initial_input_cycles <= core_perf_snap_dm_initial_input_cycles;
                perf_snap_dm_prefetch_cycles <= core_perf_snap_dm_prefetch_cycles;
                perf_snap_dm_output_cycles <= core_perf_snap_dm_output_cycles;
            end
            if (bus_req && bus_we) begin
                case (bus_addr)
                    NPU_OPSCHED_CTRL: begin
                        if (bus_wdata[NPU_OPSCHED_CTRL_START_BIT] && cmd_ready) begin
                            done_latched <= 1'b0;
                            irq_status <= 1'b0;
                        end
                    end
                    NPU_OPSCHED_DESC_ADDR: desc_addr <= bus_wdata;
                    NPU_OPSCHED_IRQ_ENABLE: irq_enable <= bus_wdata[NPU_OPSCHED_IRQ_ENABLE_ENABLE_BIT];
                    NPU_OPSCHED_IRQ_STATUS: if (bus_wdata[NPU_OPSCHED_IRQ_STATUS_PENDING_BIT]) irq_status <= 1'b0;
                    NPU_OPSCHED_PERF_CTRL: begin
                        if (bus_wdata[NPU_OPSCHED_PERF_CTRL_CLEAR_BIT] && !core_busy) begin
                            perf_snapshot_valid <= 1'b0;
                            perf_snapshot_overflow <= 1'b0;
                            perf_snap_total_cycles <= 32'h0000_0000;
                            perf_snap_core_active_cycles <= 32'h0000_0000;
                            perf_snap_core_matmul_cycles <= 32'h0000_0000;
                            perf_snap_data_mover_active_cycles <= 32'h0000_0000;
                            perf_snap_data_mover_setup_cycles <= 32'h0000_0000;
                            perf_snap_data_mover_transfer_cycles <= 32'h0000_0000;
                            perf_snap_data_mover_stall_cycles <= 32'h0000_0000;
                            perf_snap_data_mover_words <= 32'h0000_0000;
                            perf_snap_data_mover_read_words <= 32'h0000_0000;
                            perf_snap_data_mover_write_words <= 32'h0000_0000;
                            perf_snap_sram_read_words <= 32'h0000_0000;
                            perf_snap_sram_write_words <= 32'h0000_0000;
                            perf_snap_job_id <= 32'h0000_0000;
                            perf_snap_op_type <= 32'h0000_0000;
                            perf_snap_cmd_active_cycles <= 32'h0000_0000;
                            perf_snap_cmd_wait_cycles <= 32'h0000_0000;
                            perf_snap_dm_compute_overlap_cycles <= 32'h0000_0000;
                            perf_snap_uop_sched_active_cycles <= 32'h0000_0000;
                            perf_snap_uop_sched_wait_cycles <= 32'h0000_0000;
                            perf_snap_core_wait_data_cycles <= 32'h0000_0000;
                            perf_snap_core_local_active_cycles <= 32'h0000_0000;
                            perf_snap_dm_program_cycles <= 32'h0000_0000;
                            perf_snap_dm_initial_input_cycles <= 32'h0000_0000;
                            perf_snap_dm_prefetch_cycles <= 32'h0000_0000;
                            perf_snap_dm_output_cycles <= 32'h0000_0000;
                        end
                    end
                    default: begin
                    end
                endcase
            end
        end
    end
endmodule
