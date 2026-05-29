module accumulator_file #(
    parameter int TILE_ELEMS = 64,
    parameter int ACC_WIDTH = 32,
    parameter int BANKS = 2,
    parameter int COUNTER_WIDTH = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic [$clog2(BANKS)-1:0] bank_select,
    input  logic clear,
    input  logic read_enable,
    input  logic write_enable,
    input  logic accumulate_enable,
    input  logic [(TILE_ELEMS*ACC_WIDTH)-1:0] write_data_flat,
    output logic [(TILE_ELEMS*ACC_WIDTH)-1:0] read_data_flat,

    output logic [COUNTER_WIDTH-1:0] acc_read_count,
    output logic [COUNTER_WIDTH-1:0] acc_write_count,
    output logic [COUNTER_WIDTH-1:0] acc_clear_count,
    output logic [COUNTER_WIDTH-1:0] acc_residency_cycles,
    output logic [COUNTER_WIDTH-1:0] acc_spill_count
);
    logic signed [ACC_WIDTH-1:0] banks [0:BANKS-1][0:TILE_ELEMS-1];

    integer bank_idx;
    integer elem_idx;
    integer clear_idx;
    integer write_idx;

    always_comb begin
        for (elem_idx = 0; elem_idx < TILE_ELEMS; elem_idx = elem_idx + 1) begin
            read_data_flat[(elem_idx * ACC_WIDTH) +: ACC_WIDTH] = banks[bank_select][elem_idx];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (bank_idx = 0; bank_idx < BANKS; bank_idx = bank_idx + 1) begin
                for (elem_idx = 0; elem_idx < TILE_ELEMS; elem_idx = elem_idx + 1) begin
                    banks[bank_idx][elem_idx] <= '0;
                end
            end
            acc_read_count <= '0;
            acc_write_count <= '0;
            acc_clear_count <= '0;
            acc_residency_cycles <= '0;
            acc_spill_count <= '0;
        end else begin
            if (read_enable) begin
                acc_read_count <= acc_read_count + 1'b1;
            end
            if (clear) begin
                for (clear_idx = 0; clear_idx < TILE_ELEMS; clear_idx = clear_idx + 1) begin
                    banks[bank_select][clear_idx] <= '0;
                end
                acc_clear_count <= acc_clear_count + 1'b1;
                acc_residency_cycles <= '0;
            end else if (write_enable) begin
                for (write_idx = 0; write_idx < TILE_ELEMS; write_idx = write_idx + 1) begin
                    if (accumulate_enable) begin
                        banks[bank_select][write_idx] <=
                            banks[bank_select][write_idx] +
                            $signed(write_data_flat[(write_idx * ACC_WIDTH) +: ACC_WIDTH]);
                    end else begin
                        banks[bank_select][write_idx] <=
                            $signed(write_data_flat[(write_idx * ACC_WIDTH) +: ACC_WIDTH]);
                    end
                end
                acc_write_count <= acc_write_count + 1'b1;
                acc_residency_cycles <= acc_residency_cycles + 1'b1;
            end else if (acc_residency_cycles != {COUNTER_WIDTH{1'b1}}) begin
                acc_residency_cycles <= acc_residency_cycles + 1'b1;
            end
        end
    end
endmodule
