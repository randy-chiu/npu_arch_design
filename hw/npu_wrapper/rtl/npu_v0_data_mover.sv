module npu_v0_data_mover (
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
    output logic        sram_we,
    output logic [31:0] sram_addr,
    output logic [31:0] sram_wdata,
    input  logic [31:0] sram_rdata,

    output logic        host_we,
    output logic [11:0] host_addr,
    output logic [31:0] host_wdata,
    input  logic [31:0] host_rdata
);
    logic active;
    logic [7:0] current_index;

    assign active = busy || start;
    assign current_index = busy ? index : 8'h00;
    assign complete = active && (words == 8'h00 || current_index + 8'h01 >= words);

    always_comb begin
        sram_req = active && (words != 8'h00);
        sram_we = active && direction_store && (words != 8'h00);
        sram_addr = sram_base_addr + {22'h0, current_index, 2'b00};
        sram_wdata = host_rdata;
        host_we = active && !direction_store && (words != 8'h00);
        host_addr = host_base_addr + {4'h0, current_index};
        host_wdata = sram_rdata;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            index <= 8'h00;
        end else if (active) begin
            if (complete) begin
                busy <= 1'b0;
                index <= 8'h00;
            end else begin
                busy <= 1'b1;
                index <= current_index + 8'h01;
            end
        end else begin
            busy <= 1'b0;
            index <= 8'h00;
        end
    end
endmodule
