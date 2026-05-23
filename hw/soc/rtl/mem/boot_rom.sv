module boot_rom #(
    parameter int WORDS = 8192,
    parameter string INIT_HEX = "",
    parameter int DMA_LANES = 4
) (
    input  logic        clk,
    input  logic        req,
    input  logic        we,
    input  logic [31:0] addr,
    input  logic [31:0] wdata,
    output logic [31:0] rdata,
    output logic        ready,

    input  logic        dma_req,
    input  logic [31:0] dma_addr,
    output logic [(DMA_LANES*32)-1:0] dma_rdata,
    output logic        dma_ready
);
    // Current bring-up simplification:
    // INIT_HEX points at the full smoke firmware image, not just a tiny boot
    // stub. That image includes start.S, the CPU-side NPU driver, main(), and
    // generated test data. Watch ROM size as software grows; a later SoC model
    // should add a flash/loader flow instead of forcing all code into ROM.
    logic [31:0] mem [0:WORDS-1];
    logic [$clog2(WORDS)-1:0] word_addr;
    logic [$clog2(WORDS)-1:0] dma_word_addr;
    genvar dma_lane;

    assign word_addr = addr[$clog2(WORDS)+1:2];
    assign dma_word_addr = dma_addr[$clog2(WORDS)+1:2];
    assign ready = req;
    assign dma_ready = dma_req;
    assign rdata = (req && !we) ? mem[word_addr] : 32'h0000_0000;

    generate
        for (dma_lane = 0; dma_lane < DMA_LANES; dma_lane = dma_lane + 1) begin : gen_dma_read_lane
            assign dma_rdata[(dma_lane * 32) +: 32] =
                dma_req ? mem[dma_word_addr + dma_lane] : 32'h0000_0000;
        end
    endgenerate

    initial begin
        if (INIT_HEX != "") begin
            $readmemh(INIT_HEX, mem);
        end
    end

endmodule
