module soc_dma #(
    parameter int LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        bus_req,
    input  logic        bus_we,
    input  logic [11:0] bus_addr,
    input  logic [31:0] bus_wdata,
    output logic [31:0] bus_rdata,
    output logic        bus_ready,

    output logic        rom_req,
    output logic [31:0] rom_addr,
    input  logic [(LANES*32)-1:0] rom_rdata,
    input  logic        rom_ready,

    output logic        sram_req,
    output logic [LANES-1:0] sram_we,
    output logic [31:0] sram_addr,
    output logic [(LANES*32)-1:0] sram_wdata,
    input  logic        sram_ready
);
    `include "soc_v0_addr.svh"

    logic        busy;
    logic        done;
    logic [31:0] src_addr;
    logic [31:0] dst_addr;
    logic [31:0] words;
    logic [31:0] index;
    logic [31:0] dst_local_addr;
    logic [31:0] dst_lane;
    logic [31:0] active_words;
    integer lane;

    assign bus_ready = bus_req;
    assign rom_req = busy && (words != 32'h0000_0000);
    assign rom_addr = (src_addr - SOC_BOOT_ROM_BASE) + (index << 2);
    assign dst_local_addr = (dst_addr - SOC_SRAM_BASE) + (index << 2);
    assign dst_lane = (dst_local_addr >> 2) % LANES;
    assign sram_req = busy && (words != 32'h0000_0000);
    assign sram_addr = dst_local_addr - (dst_lane << 2);

    always_comb begin
        bus_rdata = 32'h0000_0000;
        case (bus_addr)
            SOC_DMA_STATUS_OFFSET[11:0]: begin
                bus_rdata[SOC_DMA_STATUS_DONE_BIT] = done;
                bus_rdata[SOC_DMA_STATUS_BUSY_BIT] = busy;
                bus_rdata[SOC_DMA_STATUS_IDLE_BIT] = !busy;
            end
            SOC_DMA_SRC_OFFSET[11:0]: bus_rdata = src_addr;
            SOC_DMA_DST_OFFSET[11:0]: bus_rdata = dst_addr;
            SOC_DMA_WORDS_OFFSET[11:0]: bus_rdata = words;
            default: bus_rdata = 32'h0000_0000;
        endcase
    end

    always_comb begin
        active_words = 32'h0000_0000;
        sram_we = '0;
        sram_wdata = '0;
        if (busy && rom_ready && sram_ready && words != 32'h0000_0000) begin
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                if (lane[31:0] >= dst_lane &&
                    index + (lane[31:0] - dst_lane) < words) begin
                    sram_we[lane] = 1'b1;
                    sram_wdata[(lane * 32) +: 32] = rom_rdata[((lane[31:0] - dst_lane) * 32) +: 32];
                    active_words = active_words + 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b1;
            src_addr <= 32'h0000_0000;
            dst_addr <= 32'h0000_0000;
            words <= 32'h0000_0000;
            index <= 32'h0000_0000;
        end else begin
            if (bus_req && bus_we && !busy) begin
                case (bus_addr)
                    SOC_DMA_CTRL_OFFSET[11:0]: begin
                        if (bus_wdata[SOC_DMA_CTRL_START_BIT]) begin
                            busy <= (words != 32'h0000_0000);
                            done <= (words == 32'h0000_0000);
                            index <= 32'h0000_0000;
                        end
                    end
                    SOC_DMA_SRC_OFFSET[11:0]: src_addr <= bus_wdata;
                    SOC_DMA_DST_OFFSET[11:0]: dst_addr <= bus_wdata;
                    SOC_DMA_WORDS_OFFSET[11:0]: words <= bus_wdata;
                    default: begin
                    end
                endcase
            end

            if (busy && rom_ready && sram_ready) begin
                if (index + active_words >= words) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    index <= 32'h0000_0000;
                end else begin
                    index <= index + active_words;
                    done <= 1'b0;
                end
            end
        end
    end
endmodule
