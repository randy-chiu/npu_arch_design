module npu_v0_opsched (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        bus_req,
    input  logic        bus_we,
    input  logic [11:0] bus_addr,
    input  logic [31:0] bus_wdata,
    output logic [31:0] bus_rdata,
    output logic        bus_ready
);
    `include "npu_v0_regs.svh"

    logic        start_pulse;
    logic        npu_done;
    logic        busy;
    logic        done_latched;
    logic        irq_enable;
    logic        irq_status;

    logic        npu_host_we;
    logic [11:0] npu_host_addr;
    logic [31:0] npu_host_wdata;
    logic [31:0] npu_host_rdata;

    assign bus_ready = bus_req;
    assign npu_host_wdata = bus_wdata;

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

    always_comb begin
        npu_host_addr = 12'h000;
        if (bus_addr >= NPU_OPSCHED_A_BASE && bus_addr < NPU_OPSCHED_A_BASE + 12'h100) begin
            npu_host_addr = 12'h000 + ((bus_addr - NPU_OPSCHED_A_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_B_BASE && bus_addr < NPU_OPSCHED_B_BASE + 12'h100) begin
            npu_host_addr = 12'h100 + ((bus_addr - NPU_OPSCHED_B_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_C_BASE && bus_addr < NPU_OPSCHED_C_BASE + 12'h100) begin
            npu_host_addr = 12'h200 + ((bus_addr - NPU_OPSCHED_C_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_X_BASE && bus_addr < NPU_OPSCHED_X_BASE + 12'h080) begin
            npu_host_addr = 12'h300 + ((bus_addr - NPU_OPSCHED_X_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_Y_BASE && bus_addr < NPU_OPSCHED_Y_BASE + 12'h080) begin
            npu_host_addr = 12'h380 + ((bus_addr - NPU_OPSCHED_Y_BASE) >> 2);
        end else if (bus_addr >= NPU_OPSCHED_PROGRAM_BASE && bus_addr < NPU_OPSCHED_PROGRAM_BASE + 12'h100) begin
            npu_host_addr = 12'h400 + ((bus_addr - NPU_OPSCHED_PROGRAM_BASE) >> 2);
        end
    end

    always_comb begin
        npu_host_we = 1'b0;
        if (bus_req && bus_we) begin
            npu_host_we =
                (bus_addr >= NPU_OPSCHED_A_BASE && bus_addr < NPU_OPSCHED_A_BASE + 12'h100) ||
                (bus_addr >= NPU_OPSCHED_B_BASE && bus_addr < NPU_OPSCHED_B_BASE + 12'h100) ||
                (bus_addr >= NPU_OPSCHED_X_BASE && bus_addr < NPU_OPSCHED_X_BASE + 12'h080) ||
                (bus_addr >= NPU_OPSCHED_PROGRAM_BASE && bus_addr < NPU_OPSCHED_PROGRAM_BASE + 12'h100);
        end
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
        end else begin
            start_pulse <= 1'b0;
            if (npu_done) begin
                busy <= 1'b0;
                done_latched <= 1'b1;
                irq_status <= irq_enable;
            end

            if (bus_req && bus_we) begin
                case (bus_addr)
                    NPU_OPSCHED_CTRL: begin
                        if (bus_wdata[0]) begin
                            start_pulse <= 1'b1;
                            busy <= 1'b1;
                            done_latched <= 1'b0;
                            irq_status <= 1'b0;
                        end
                    end
                    NPU_OPSCHED_IRQ_ENABLE: begin
                        irq_enable <= bus_wdata[0];
                    end
                    NPU_OPSCHED_IRQ_STATUS: begin
                        if (bus_wdata[0]) irq_status <= 1'b0;
                    end
                    default: begin
                    end
                endcase
            end
        end
    end
endmodule
