module npu_v0_opsched #(
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
    `include "soc_v0_addr.svh"

    typedef enum logic [3:0] {
        DESC_IDLE,
        DESC_READ,
        DESC_FETCH_PROGRAM,
        DESC_FETCH_INPUT0,
        DESC_FETCH_INPUT1,
        DESC_START_CORE,
        DESC_WAIT_CORE,
        DESC_WRITE_OUTPUT,
        DESC_DONE,
        DESC_CONFIG_ACC,
        DESC_DISABLE_ACC,
        DESC_CONFIG_NEXT_BANK
    } desc_state_t;

    logic        start_pulse;
    logic        npu_done;
    logic        busy;
    logic        done_latched;
    logic        irq_enable;
    logic        irq_status;
    logic [31:0] desc_addr;

    desc_state_t desc_state;
    logic [3:0]  desc_idx;
    logic [7:0]  transfer_idx;
    logic [15:0] stream_chunk_idx;
    logic [1:0]  prefetch_phase;
    logic        core_done_seen;

    logic [31:0] job_op_type;
    logic [31:0] job_program_addr;
    logic [31:0] job_program_words;
    logic [31:0] job_input0_addr;
    logic [31:0] job_input0_words;
    logic [31:0] job_input1_addr;
    logic [31:0] job_input1_words;
    logic [31:0] job_output_addr;
    logic [31:0] job_output_words;
    logic [31:0] job_k_chunks;

    logic [CORE_HOST_LANES-1:0] npu_host_we;
    logic [11:0] npu_host_addr;
    logic [(CORE_HOST_LANES*32)-1:0] npu_host_wdata;
    logic [(CORE_HOST_LANES*32)-1:0] npu_host_rdata;
    logic [31:0] npu_host_rdata_lane0;

    logic        legacy_host_we;
    logic [11:0] legacy_host_addr;
    logic        desc_host_we;
    logic [11:0] desc_host_addr;
    logic [31:0] desc_host_wdata;
    logic        mover_start;
    logic        mover_store;
    logic [31:0] mover_sram_base;
    logic [11:0] mover_host_base;
    logic [7:0]  mover_words;
    logic        mover_busy;
    logic        mover_complete;
    logic [7:0]  mover_index;
    logic        mover_sram_req;
    logic [CORE_HOST_LANES-1:0] mover_sram_we;
    logic [31:0] mover_sram_addr;
    logic [(CORE_HOST_LANES*32)-1:0] mover_sram_wdata;
    logic [CORE_HOST_LANES-1:0] mover_host_we;
    logic [11:0] mover_host_addr;
    logic [(CORE_HOST_LANES*32)-1:0] mover_host_wdata;
    logic        mover_perf_active;
    logic        mover_perf_setup;
    logic        mover_perf_transfer;
    logic        mover_perf_stall;
    logic [31:0] mover_perf_words;
    logic        job_is_k_stream;
    logic        k_stream_has_next;
    logic        k_stream_next_bank;

    localparam int DATA_MOVER_WORDS_PER_CYCLE = SOC_NPU_DATA_MOVER_WORDS_PER_CYCLE;
    localparam int DATA_MOVER_SETUP_CYCLES = SOC_NPU_DATA_MOVER_SETUP_CYCLES;

    assign job_is_k_stream = (job_op_type == SOC_NPU_JOB_OP_MATMUL_K_STREAM);
    assign k_stream_has_next =
        job_is_k_stream && (stream_chunk_idx + 16'h0001 < job_k_chunks[15:0]);
    assign k_stream_next_bank = ~stream_chunk_idx[0];

    assign bus_ready = bus_req;
    assign npu_host_addr = (desc_state == DESC_IDLE) ? legacy_host_addr : desc_host_addr;
    assign npu_host_we = (desc_state == DESC_IDLE) ?
        {{(CORE_HOST_LANES-1){1'b0}}, legacy_host_we} :
        (mover_host_we | {{(CORE_HOST_LANES-1){1'b0}}, desc_host_we});
    assign npu_host_wdata = (desc_state == DESC_IDLE) ?
        {{((CORE_HOST_LANES-1)*32){1'b0}}, bus_wdata} :
        (mover_host_wdata | {{((CORE_HOST_LANES-1)*32){1'b0}}, desc_host_wdata});
    assign npu_host_rdata_lane0 = npu_host_rdata[31:0];

    npu_v0_top #(
        .CORE_HOST_LANES(CORE_HOST_LANES)
    ) u_npu (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_pulse),
        .op(1'b0),
        .done(npu_done),
        .host_we(npu_host_we),
        .host_addr(npu_host_addr),
        .host_wdata(npu_host_wdata),
        .host_rdata(npu_host_rdata)
    );

    npu_v0_data_mover #(
        .WORDS_PER_CYCLE(DATA_MOVER_WORDS_PER_CYCLE),
        .SETUP_CYCLES(DATA_MOVER_SETUP_CYCLES),
        .LANES(CORE_HOST_LANES)
    ) u_data_mover (
        .clk(clk),
        .rst_n(rst_n),
        .start(mover_start),
        .direction_store(mover_store),
        .sram_base_addr(mover_sram_base),
        .host_base_addr(mover_host_base),
        .words(mover_words),
        .busy(mover_busy),
        .complete(mover_complete),
        .index(mover_index),
        .sram_req(mover_sram_req),
        .sram_we(mover_sram_we),
        .sram_addr(mover_sram_addr),
        .sram_wdata(mover_sram_wdata),
        .sram_rdata(sram_rdata),
        .host_we(mover_host_we),
        .host_addr(mover_host_addr),
        .host_wdata(mover_host_wdata),
        .host_rdata(npu_host_rdata),
        .perf_active(mover_perf_active),
        .perf_setup(mover_perf_setup),
        .perf_transfer(mover_perf_transfer),
        .perf_stall(mover_perf_stall),
        .perf_words(mover_perf_words)
    );

    always @* begin
        legacy_host_addr = 12'h000;
        if (bus_addr >= NPU_OPSCHED_A_BASE && bus_addr < NPU_OPSCHED_A_BASE + 12'h100) begin
            legacy_host_addr = 12'h000 + ((bus_addr - NPU_OPSCHED_A_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_B_BASE && bus_addr < NPU_OPSCHED_B_BASE + 12'h100) begin
            legacy_host_addr = 12'h100 + ((bus_addr - NPU_OPSCHED_B_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_C_BASE && bus_addr < NPU_OPSCHED_C_BASE + 12'h100) begin
            legacy_host_addr = 12'h200 + ((bus_addr - NPU_OPSCHED_C_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_X_BASE && bus_addr < NPU_OPSCHED_X_BASE + 12'h080) begin
            legacy_host_addr = 12'h300 + ((bus_addr - NPU_OPSCHED_X_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_Y_BASE && bus_addr < NPU_OPSCHED_Y_BASE + 12'h080) begin
            legacy_host_addr = 12'h380 + ((bus_addr - NPU_OPSCHED_Y_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_PROGRAM_BASE && bus_addr < NPU_OPSCHED_PROGRAM_BASE + 12'h100) begin
            legacy_host_addr = 12'h400 + ((bus_addr - NPU_OPSCHED_PROGRAM_BASE) >> 2);
        end
    end

    always_comb begin
        legacy_host_we = 1'b0;
        if (bus_req && bus_we) begin
            legacy_host_we =
                (bus_addr >= NPU_OPSCHED_A_BASE && bus_addr < NPU_OPSCHED_A_BASE + 12'h100) ||
                (bus_addr >= NPU_OPSCHED_B_BASE && bus_addr < NPU_OPSCHED_B_BASE + 12'h100) ||
                (bus_addr >= NPU_OPSCHED_X_BASE && bus_addr < NPU_OPSCHED_X_BASE + 12'h080) ||
                (bus_addr >= NPU_OPSCHED_PROGRAM_BASE && bus_addr < NPU_OPSCHED_PROGRAM_BASE + 12'h100);
        end
    end

    always @* begin
        sram_req = 1'b0;
        sram_we = '0;
        sram_addr = 32'h0000_0000;
        sram_wdata = '0;
        desc_host_we = 1'b0;
        desc_host_addr = 12'h000;
        desc_host_wdata = 32'h0000_0000;
        mover_start = 1'b0;
        mover_store = 1'b0;
        mover_sram_base = 32'h0000_0000;
        mover_host_base = 12'h000;
        mover_words = 8'h00;

        case (desc_state)
            DESC_READ: begin
                sram_req = 1'b1;
                sram_addr = desc_addr + {26'h0, desc_idx, 2'b00};
            end
            DESC_FETCH_PROGRAM: begin
                mover_start = !mover_busy;
                mover_sram_base = job_program_addr;
                mover_host_base = 12'h400;
                mover_words = job_program_words[7:0];
                sram_req = mover_sram_req;
                sram_we = mover_sram_we;
                sram_addr = mover_sram_addr;
                sram_wdata = mover_sram_wdata;
                desc_host_we = |mover_host_we;
                desc_host_addr = mover_host_addr;
                desc_host_wdata = 32'h0000_0000;
            end
            DESC_FETCH_INPUT0: begin
                mover_start = !mover_busy;
                if (job_is_k_stream) begin
                    mover_sram_base = job_input0_addr + ((stream_chunk_idx * job_input0_words[15:0]) << 2);
                end else begin
                    mover_sram_base = job_input0_addr;
                end
                mover_words = job_input0_words[7:0];
                if (job_op_type == SOC_NPU_JOB_OP_SOFTMAX) begin
                    mover_host_base = 12'h300;
                end else begin
                    mover_host_base = 12'h000;
                end
                sram_req = mover_sram_req;
                sram_we = mover_sram_we;
                sram_addr = mover_sram_addr;
                sram_wdata = mover_sram_wdata;
                desc_host_we = |mover_host_we;
                desc_host_addr = mover_host_addr;
                desc_host_wdata = 32'h0000_0000;
            end
            DESC_FETCH_INPUT1: begin
                mover_start = !mover_busy;
                if (job_is_k_stream) begin
                    mover_sram_base = job_input1_addr + ((stream_chunk_idx * job_input1_words[15:0]) << 2);
                end else begin
                    mover_sram_base = job_input1_addr;
                end
                mover_host_base = 12'h100;
                mover_words = job_input1_words[7:0];
                sram_req = mover_sram_req;
                sram_we = mover_sram_we;
                sram_addr = mover_sram_addr;
                sram_wdata = mover_sram_wdata;
                desc_host_we = |mover_host_we;
                desc_host_addr = mover_host_addr;
                desc_host_wdata = 32'h0000_0000;
            end
            DESC_WRITE_OUTPUT: begin
                mover_start = !mover_busy;
                mover_store = 1'b1;
                mover_sram_base = job_output_addr;
                mover_words = job_output_words[7:0];
                if (job_op_type == SOC_NPU_JOB_OP_SOFTMAX) begin
                    mover_host_base = 12'h380;
                end else begin
                    mover_host_base = 12'h200;
                end
                sram_req = mover_sram_req;
                sram_we = mover_sram_we;
                sram_addr = mover_sram_addr;
                sram_wdata = mover_sram_wdata;
                desc_host_we = |mover_host_we;
                desc_host_addr = mover_host_addr;
                desc_host_wdata = 32'h0000_0000;
            end
            DESC_CONFIG_ACC: begin
                desc_host_we = 1'b1;
                desc_host_addr = 12'h500;
                desc_host_wdata = 32'h0000_0003;
            end
            DESC_CONFIG_NEXT_BANK: begin
                desc_host_we = 1'b1;
                desc_host_addr = 12'h500;
                desc_host_wdata = 32'h0000_0001 |
                    (k_stream_next_bank ? 32'h0000_000c : 32'h0000_0000);
            end
            DESC_DISABLE_ACC: begin
                desc_host_we = 1'b1;
                desc_host_addr = 12'h500;
                desc_host_wdata = 32'h0000_0000;
            end
            DESC_WAIT_CORE: begin
                if (k_stream_has_next && prefetch_phase != 2'd2) begin
                    mover_start = !mover_busy;
                    mover_words = (prefetch_phase == 2'd0) ?
                        job_input0_words[7:0] : job_input1_words[7:0];
                    if (prefetch_phase == 2'd0) begin
                        mover_sram_base = job_input0_addr +
                            (((stream_chunk_idx + 16'h0001) * job_input0_words[15:0]) << 2);
                        mover_host_base = 12'h000;
                    end else begin
                        mover_sram_base = job_input1_addr +
                            (((stream_chunk_idx + 16'h0001) * job_input1_words[15:0]) << 2);
                        mover_host_base = 12'h100;
                    end
                    sram_req = mover_sram_req;
                    sram_we = mover_sram_we;
                    sram_addr = mover_sram_addr;
                    sram_wdata = mover_sram_wdata;
                    desc_host_we = |mover_host_we;
                    desc_host_addr = mover_host_addr;
                    desc_host_wdata = 32'h0000_0000;
                end
            end
            default: begin
            end
        endcase
    end

    always_comb begin
        bus_rdata = 32'h0000_0000;
        case (bus_addr)
            NPU_OPSCHED_CTRL: begin
                bus_rdata = 32'h0000_0000;
            end
            NPU_OPSCHED_STATUS: begin
                bus_rdata = {29'h0, !busy, busy, done_latched};
            end
            NPU_OPSCHED_VERSION: begin
                bus_rdata = 32'h0001_0000;
            end
            NPU_OPSCHED_IRQ_ENABLE: begin
                bus_rdata = {31'h0, irq_enable};
            end
            NPU_OPSCHED_IRQ_STATUS: begin
                bus_rdata = {31'h0, irq_status};
            end
            NPU_OPSCHED_DESC_ADDR: begin
                bus_rdata = desc_addr;
            end
            default: begin
                if (!bus_we &&
                    ((bus_addr >= NPU_OPSCHED_C_BASE && bus_addr < NPU_OPSCHED_C_BASE + 12'h100) ||
                     (bus_addr >= NPU_OPSCHED_Y_BASE && bus_addr < NPU_OPSCHED_Y_BASE + 12'h080))) begin
                    bus_rdata = npu_host_rdata_lane0;
                end
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_pulse <= 1'b0;
            busy <= 1'b0;
            done_latched <= 1'b0;
            irq_enable <= 1'b0;
            irq_status <= 1'b0;
            desc_addr <= 32'h0000_0000;
            desc_state <= DESC_IDLE;
            desc_idx <= 4'h0;
            transfer_idx <= 8'h0;
            stream_chunk_idx <= 16'h0000;
            prefetch_phase <= 2'd0;
            core_done_seen <= 1'b0;
            job_op_type <= 32'h0000_0000;
            job_program_addr <= 32'h0000_0000;
            job_program_words <= 32'h0000_0000;
            job_input0_addr <= 32'h0000_0000;
            job_input0_words <= 32'h0000_0000;
            job_input1_addr <= 32'h0000_0000;
            job_input1_words <= 32'h0000_0000;
            job_output_addr <= 32'h0000_0000;
            job_output_words <= 32'h0000_0000;
            job_k_chunks <= 32'h0000_0000;
        end else begin
            start_pulse <= 1'b0;
            if (npu_done && desc_state == DESC_IDLE) begin
                busy <= 1'b0;
                done_latched <= 1'b1;
                irq_status <= irq_enable;
            end

            case (desc_state)
                DESC_IDLE: begin
                end
                DESC_READ: begin
                    case (desc_idx)
                        SOC_NPU_JOB_DESC_OP_TYPE_WORD: job_op_type <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_PROGRAM_ADDR_WORD: job_program_addr <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_PROGRAM_WORDS_WORD: job_program_words <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_INPUT0_ADDR_WORD: job_input0_addr <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_INPUT0_WORDS_WORD: job_input0_words <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_INPUT1_ADDR_WORD: job_input1_addr <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_INPUT1_WORDS_WORD: job_input1_words <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_OUTPUT_ADDR_WORD: job_output_addr <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_OUTPUT_WORDS_WORD: job_output_words <= sram_rdata[31:0];
                        SOC_NPU_JOB_DESC_K_CHUNKS_WORD: job_k_chunks <= sram_rdata[31:0];
                        default: begin
                        end
                    endcase
                    if (desc_idx == SOC_NPU_JOB_DESC_WORDS - 1) begin
                        desc_idx <= 4'h0;
                        transfer_idx <= 8'h0;
                        stream_chunk_idx <= 16'h0000;
                        prefetch_phase <= 2'd0;
                        core_done_seen <= 1'b0;
                        desc_state <= DESC_FETCH_PROGRAM;
                    end else begin
                        desc_idx <= desc_idx + 1'b1;
                    end
                end
                DESC_FETCH_PROGRAM: begin
                    if (mover_complete) begin
                        transfer_idx <= 8'h0;
                        stream_chunk_idx <= 16'h0000;
                        if (job_is_k_stream) begin
                            desc_state <= DESC_CONFIG_ACC;
                        end else begin
                            desc_state <= DESC_FETCH_INPUT0;
                        end
                    end
                end
                DESC_CONFIG_ACC: begin
                    desc_state <= DESC_FETCH_INPUT0;
                end
                DESC_CONFIG_NEXT_BANK: begin
                    desc_state <= DESC_WAIT_CORE;
                end
                DESC_DISABLE_ACC: begin
                    desc_state <= DESC_DONE;
                end
                DESC_FETCH_INPUT0: begin
                    if (mover_complete) begin
                        transfer_idx <= 8'h0;
                        if (job_op_type == SOC_NPU_JOB_OP_MATMUL || job_is_k_stream) begin
                            desc_state <= DESC_FETCH_INPUT1;
                        end else begin
                            desc_state <= DESC_START_CORE;
                        end
                    end
                end
                DESC_FETCH_INPUT1: begin
                    if (mover_complete) begin
                        transfer_idx <= 8'h0;
                        desc_state <= DESC_START_CORE;
                    end
                end
                DESC_START_CORE: begin
                    start_pulse <= 1'b1;
                    core_done_seen <= 1'b0;
                    prefetch_phase <= 2'd0;
                    if (k_stream_has_next) begin
                        desc_state <= DESC_CONFIG_NEXT_BANK;
                    end else begin
                        desc_state <= DESC_WAIT_CORE;
                    end
                end
                DESC_WAIT_CORE: begin
                    if (npu_done) begin
                        core_done_seen <= 1'b1;
                    end
                    if (k_stream_has_next && mover_complete) begin
                        if (prefetch_phase == 2'd0) begin
                            prefetch_phase <= 2'd1;
                        end else if (prefetch_phase == 2'd1) begin
                            prefetch_phase <= 2'd2;
                        end
                    end
                    if ((core_done_seen || npu_done) &&
                        (!k_stream_has_next ||
                         prefetch_phase == 2'd2 ||
                         (prefetch_phase == 2'd1 && mover_complete))) begin
                        transfer_idx <= 8'h0;
                        core_done_seen <= 1'b0;
                        prefetch_phase <= 2'd0;
                        if (k_stream_has_next) begin
                            stream_chunk_idx <= stream_chunk_idx + 16'h0001;
                            desc_state <= DESC_START_CORE;
                        end else begin
                            desc_state <= DESC_WRITE_OUTPUT;
                        end
                    end
                end
                DESC_WRITE_OUTPUT: begin
                    if (mover_complete) begin
                        transfer_idx <= 8'h0;
                        if (job_is_k_stream) begin
                            desc_state <= DESC_DISABLE_ACC;
                        end else begin
                            desc_state <= DESC_DONE;
                        end
                    end
                end
                DESC_DONE: begin
                    busy <= 1'b0;
                    done_latched <= 1'b1;
                    irq_status <= irq_enable;
                    desc_state <= DESC_IDLE;
                end
                default: begin
                    desc_state <= DESC_IDLE;
                    busy <= 1'b0;
                    done_latched <= 1'b1;
                end
            endcase

            if (bus_req && bus_we) begin
                case (bus_addr)
                    NPU_OPSCHED_CTRL: begin
                        if (bus_wdata[0]) begin
                            busy <= 1'b1;
                            done_latched <= 1'b0;
                            irq_status <= 1'b0;
                            if (desc_addr != 32'h0000_0000) begin
                                desc_idx <= 4'h0;
                                transfer_idx <= 8'h0;
                                stream_chunk_idx <= 16'h0000;
                                prefetch_phase <= 2'd0;
                                core_done_seen <= 1'b0;
                                desc_state <= DESC_READ;
                            end else begin
                                start_pulse <= 1'b1;
                            end
                        end
                    end
                    NPU_OPSCHED_IRQ_ENABLE: begin
                        irq_enable <= bus_wdata[0];
                    end
                    NPU_OPSCHED_IRQ_STATUS: begin
                        if (bus_wdata[0]) irq_status <= 1'b0;
                    end
                    NPU_OPSCHED_DESC_ADDR: begin
                        desc_addr <= bus_wdata;
                    end
                    default: begin
                    end
                endcase
            end
        end
    end
endmodule
