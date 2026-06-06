module npu_v0_data_mover #(
    parameter int WORDS_PER_CYCLE = 1,
    parameter int SETUP_CYCLES = 0,
    parameter int LANES = 4
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        start,
    input  logic        direction_store,
    input  logic [31:0] sram_base_addr,
    input  logic [11:0] host_base_addr,
    input  logic [7:0]  words,

    output logic        busy,
    output logic        complete,
    output logic [7:0]  index,

    output logic        sram_req,
    output logic [LANES-1:0] sram_we,
    output logic [31:0] sram_addr,
    output logic [(LANES*32)-1:0] sram_wdata,
    input  logic [(LANES*32)-1:0] sram_rdata,

    output logic [LANES-1:0] host_we,
    output logic [11:0] host_addr,
    output logic [(LANES*32)-1:0] host_wdata,
    input  logic [(LANES*32)-1:0] host_rdata,

    output logic        perf_active,
    output logic        perf_setup,
    output logic        perf_transfer,
    output logic        perf_stall,
    output logic [31:0] perf_words
);
    initial begin
        if (WORDS_PER_CYCLE <= 0 || WORDS_PER_CYCLE > LANES) begin
            $error("npu_v0_data_mover WORDS_PER_CYCLE must be in 1..LANES");
        end
        if (SETUP_CYCLES < 0) begin
            $error("npu_v0_data_mover SETUP_CYCLES must be non-negative");
        end
    end

    localparam int SETUP_COUNT_WIDTH = (SETUP_CYCLES > 0) ? $clog2(SETUP_CYCLES + 1) : 1;
    localparam logic [7:0] WORDS_PER_CYCLE_U8 = WORDS_PER_CYCLE[7:0];

    logic active;
    logic setup_active;
    logic [7:0] current_index;
    logic [SETUP_COUNT_WIDTH-1:0] setup_count;
    logic [LANES-1:0] active_lane_mask;
    integer lane_idx;
    integer perf_lane_idx;

    assign setup_active = (SETUP_CYCLES > 0) && (start || setup_count != '0);
    assign active = (busy || start) && !setup_active;
    assign current_index = busy ? index : 8'h00;
    assign complete = active && (words == 8'h00 || current_index + WORDS_PER_CYCLE_U8 >= words);

    always_comb begin
        active_lane_mask = '0;
        for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
            if (lane_idx < WORDS_PER_CYCLE && current_index + lane_idx < words) begin
                active_lane_mask = active_lane_mask | ({{(LANES-1){1'b0}}, 1'b1} << lane_idx);
            end
        end
    end

    always_comb begin
        sram_req = active && (words != 8'h00);
        sram_we = (active && direction_store && (words != 8'h00)) ? active_lane_mask : '0;
        sram_addr = sram_base_addr + {22'h0, current_index, 2'b00};
        sram_wdata = host_rdata;
        host_we = (active && !direction_store && (words != 8'h00)) ? active_lane_mask : '0;
        host_addr = host_base_addr + {4'h0, current_index};
        host_wdata = sram_rdata;
    end

    always_comb begin
        perf_words = 32'h0000_0000;
        for (perf_lane_idx = 0; perf_lane_idx < LANES; perf_lane_idx = perf_lane_idx + 1) begin
            if (active_lane_mask[perf_lane_idx]) begin
                perf_words = perf_words + 1'b1;
            end
        end
    end

    assign perf_active = busy || start;
    assign perf_setup = setup_active;
    assign perf_transfer = active && (words != 8'h00);
    assign perf_stall = 1'b0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            index <= 8'h00;
            setup_count <= '0;
        end else if (SETUP_CYCLES > 0 && start && !busy && setup_count == '0) begin
            setup_count <= SETUP_CYCLES;
            busy <= 1'b1;
            index <= 8'h00;
        end else if (SETUP_CYCLES > 0 && setup_count != '0) begin
            setup_count <= setup_count - 1'b1;
            busy <= 1'b1;
            index <= 8'h00;
        end else if (active) begin
            if (complete) begin
                busy <= 1'b0;
                index <= 8'h00;
                setup_count <= '0;
            end else begin
                busy <= 1'b1;
                index <= current_index + WORDS_PER_CYCLE_U8;
                setup_count <= '0;
            end
        end else begin
            busy <= 1'b0;
            index <= 8'h00;
            setup_count <= '0;
        end
    end
endmodule
