module npu_v0_opsched (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        bus_req,
    input  logic        bus_we,
    input  logic [11:0] bus_addr,
    input  logic [31:0] bus_wdata,
    output logic [31:0] bus_rdata,
    output logic        bus_ready,

    output logic        sram_req,
    output logic        sram_we,
    output logic [31:0] sram_addr,
    output logic [31:0] sram_wdata,
    input  logic [31:0] sram_rdata,
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
        DESC_DONE
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

    logic [31:0] job_op_type;
    logic [31:0] job_program_addr;
    logic [31:0] job_program_words;
    logic [31:0] job_input0_addr;
    logic [31:0] job_input0_words;
    logic [31:0] job_input1_addr;
    logic [31:0] job_input1_words;
    logic [31:0] job_output_addr;
    logic [31:0] job_output_words;

    logic        npu_host_we;
    logic [11:0] npu_host_addr;
    logic [31:0] npu_host_wdata;
    logic [31:0] npu_host_rdata;

    logic        legacy_host_we;
    logic [11:0] legacy_host_addr;
    logic        desc_host_we;
    logic [11:0] desc_host_addr;
    logic [31:0] desc_host_wdata;

    assign bus_ready = bus_req;
    assign npu_host_addr = (desc_state == DESC_IDLE) ? legacy_host_addr : desc_host_addr;
    assign npu_host_we = (desc_state == DESC_IDLE) ? legacy_host_we : desc_host_we;
    assign npu_host_wdata = (desc_state == DESC_IDLE) ? bus_wdata : desc_host_wdata;

    npu_v0_top u_npu (
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
        sram_we = 1'b0;
        sram_addr = 32'h0000_0000;
        sram_wdata = 32'h0000_0000;
        desc_host_we = 1'b0;
        desc_host_addr = 12'h000;
        desc_host_wdata = 32'h0000_0000;

        case (desc_state)
            DESC_READ: begin
                sram_req = 1'b1;
                sram_addr = desc_addr + {26'h0, desc_idx, 2'b00};
            end
            DESC_FETCH_PROGRAM: begin
                sram_req = 1'b1;
                sram_addr = job_program_addr + {22'h0, transfer_idx, 2'b00};
                desc_host_we = 1'b1;
                desc_host_addr = 12'h400 + transfer_idx[3:0];
                desc_host_wdata = sram_rdata;
            end
            DESC_FETCH_INPUT0: begin
                sram_req = 1'b1;
                sram_addr = job_input0_addr + {22'h0, transfer_idx, 2'b00};
                desc_host_we = 1'b1;
                if (job_op_type == SOC_NPU_JOB_OP_SOFTMAX) begin
                    desc_host_addr = 12'h300 + transfer_idx[2:0];
                end else begin
                    desc_host_addr = 12'h000 + transfer_idx[5:0];
                end
                desc_host_wdata = sram_rdata;
            end
            DESC_FETCH_INPUT1: begin
                sram_req = 1'b1;
                sram_addr = job_input1_addr + {22'h0, transfer_idx, 2'b00};
                desc_host_we = 1'b1;
                desc_host_addr = 12'h100 + transfer_idx[5:0];
                desc_host_wdata = sram_rdata;
            end
            DESC_WRITE_OUTPUT: begin
                sram_req = 1'b1;
                sram_we = 1'b1;
                sram_addr = job_output_addr + {22'h0, transfer_idx, 2'b00};
                if (job_op_type == SOC_NPU_JOB_OP_SOFTMAX) begin
                    desc_host_addr = 12'h380 + transfer_idx[2:0];
                    sram_wdata = npu_host_rdata;
                end else begin
                    desc_host_addr = 12'h200 + transfer_idx[5:0];
                    sram_wdata = npu_host_rdata;
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
                    bus_rdata = npu_host_rdata;
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
            job_op_type <= 32'h0000_0000;
            job_program_addr <= 32'h0000_0000;
            job_program_words <= 32'h0000_0000;
            job_input0_addr <= 32'h0000_0000;
            job_input0_words <= 32'h0000_0000;
            job_input1_addr <= 32'h0000_0000;
            job_input1_words <= 32'h0000_0000;
            job_output_addr <= 32'h0000_0000;
            job_output_words <= 32'h0000_0000;
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
                        SOC_NPU_JOB_DESC_OP_TYPE_WORD: job_op_type <= sram_rdata;
                        SOC_NPU_JOB_DESC_PROGRAM_ADDR_WORD: job_program_addr <= sram_rdata;
                        SOC_NPU_JOB_DESC_PROGRAM_WORDS_WORD: job_program_words <= sram_rdata;
                        SOC_NPU_JOB_DESC_INPUT0_ADDR_WORD: job_input0_addr <= sram_rdata;
                        SOC_NPU_JOB_DESC_INPUT0_WORDS_WORD: job_input0_words <= sram_rdata;
                        SOC_NPU_JOB_DESC_INPUT1_ADDR_WORD: job_input1_addr <= sram_rdata;
                        SOC_NPU_JOB_DESC_INPUT1_WORDS_WORD: job_input1_words <= sram_rdata;
                        SOC_NPU_JOB_DESC_OUTPUT_ADDR_WORD: job_output_addr <= sram_rdata;
                        SOC_NPU_JOB_DESC_OUTPUT_WORDS_WORD: job_output_words <= sram_rdata;
                        default: begin
                        end
                    endcase
                    if (desc_idx == SOC_NPU_JOB_DESC_WORDS - 1) begin
                        desc_idx <= 4'h0;
                        transfer_idx <= 8'h0;
                        desc_state <= DESC_FETCH_PROGRAM;
                    end else begin
                        desc_idx <= desc_idx + 1'b1;
                    end
                end
                DESC_FETCH_PROGRAM: begin
                    if (transfer_idx + 1'b1 >= job_program_words[7:0]) begin
                        transfer_idx <= 8'h0;
                        desc_state <= DESC_FETCH_INPUT0;
                    end else begin
                        transfer_idx <= transfer_idx + 1'b1;
                    end
                end
                DESC_FETCH_INPUT0: begin
                    if (transfer_idx + 1'b1 >= job_input0_words[7:0]) begin
                        transfer_idx <= 8'h0;
                        if (job_op_type == SOC_NPU_JOB_OP_MATMUL) begin
                            desc_state <= DESC_FETCH_INPUT1;
                        end else begin
                            desc_state <= DESC_START_CORE;
                        end
                    end else begin
                        transfer_idx <= transfer_idx + 1'b1;
                    end
                end
                DESC_FETCH_INPUT1: begin
                    if (transfer_idx + 1'b1 >= job_input1_words[7:0]) begin
                        transfer_idx <= 8'h0;
                        desc_state <= DESC_START_CORE;
                    end else begin
                        transfer_idx <= transfer_idx + 1'b1;
                    end
                end
                DESC_START_CORE: begin
                    start_pulse <= 1'b1;
                    desc_state <= DESC_WAIT_CORE;
                end
                DESC_WAIT_CORE: begin
                    if (npu_done) begin
                        transfer_idx <= 8'h0;
                        desc_state <= DESC_WRITE_OUTPUT;
                    end
                end
                DESC_WRITE_OUTPUT: begin
                    if (transfer_idx + 1'b1 >= job_output_words[7:0]) begin
                        transfer_idx <= 8'h0;
                        desc_state <= DESC_DONE;
                    end else begin
                        transfer_idx <= transfer_idx + 1'b1;
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
