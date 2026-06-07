module npu_v0_uop_scheduler (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,

    output logic [$clog2(RTL_HOST_PROGRAM_WORDS)-1:0] program_addr,
    input  logic [31:0] program_rdata,

    output logic        local_load_valid,
    output logic        local_store_valid,
    output logic [3:0]  local_opcode,
    output logic [3:0]  local_tensor,
    output logic [3:0]  local_buffer,
    output logic        primitive_cmd_valid,
    input  logic        primitive_cmd_ready,
    output logic [3:0]  primitive_cmd_opcode,
    output logic [3:0]  primitive_cmd_row,
    output logic [3:0]  primitive_cmd_lane,
    input  logic        primitive_rsp_valid,
    output logic        primitive_rsp_ready,
    output logic        matrix_start,
    input  logic        matrix_done,

    output logic        perf_active,
    output logic        perf_wait,
    output logic [3:0]  perf_wait_reason
);
    `include "npu_v0_spec.svh"

    typedef enum logic [2:0] {
        SCHED_IDLE,
        SCHED_FETCH,
        SCHED_WAIT_MATRIX,
        SCHED_WAIT_LOCAL,
        SCHED_DONE
    } sched_state_t;

    sched_state_t state;
    logic [$clog2(RTL_HOST_PROGRAM_WORDS)-1:0] pc;
    logic [31:0] current_instr;

    assign program_addr = pc;
    assign current_instr = program_rdata;
    assign primitive_cmd_row = current_instr[UOP_ARG0_MSB:UOP_ARG0_LSB];
    assign primitive_cmd_lane = current_instr[UOP_ARG1_MSB:UOP_ARG1_LSB];
    assign primitive_cmd_opcode = current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB];
    assign local_tensor = primitive_cmd_row;
    assign local_buffer = primitive_cmd_lane;
    assign local_opcode = primitive_cmd_opcode;
    assign local_load_valid =
        state == SCHED_FETCH &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] == UOP_LOAD;
    assign local_store_valid =
        state == SCHED_FETCH &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] == UOP_STORE;
    assign primitive_cmd_valid =
        state == SCHED_FETCH &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] != UOP_LOAD &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] != UOP_STORE &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] != UOP_MATMUL &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] != UOP_HALT;
    assign primitive_rsp_ready = state == SCHED_WAIT_LOCAL;
    assign matrix_start =
        state == SCHED_FETCH &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] == UOP_MATMUL;
    assign done = state == SCHED_DONE;
    always_comb begin
        perf_wait_reason = TRACE_SCHED_WAIT_NONE;
        if (state == SCHED_WAIT_MATRIX && !matrix_done) begin
            perf_wait_reason = TRACE_SCHED_WAIT_MATRIX_RESPONSE;
        end else if (state == SCHED_FETCH && primitive_cmd_valid && !primitive_cmd_ready) begin
            perf_wait_reason = TRACE_SCHED_WAIT_PRIMITIVE_ACCEPT;
        end else if (state == SCHED_WAIT_LOCAL && !primitive_rsp_valid) begin
            perf_wait_reason = TRACE_SCHED_WAIT_PRIMITIVE_RESPONSE;
        end
    end
    assign perf_wait = perf_wait_reason != TRACE_SCHED_WAIT_NONE;
    assign perf_active =
        (state == SCHED_FETCH && !perf_wait) ||
        (state == SCHED_WAIT_MATRIX && matrix_done) ||
        (state == SCHED_WAIT_LOCAL && primitive_rsp_valid) ||
        state == SCHED_DONE;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= SCHED_IDLE;
            pc <= '0;
        end else begin
            case (state)
                SCHED_IDLE: begin
                    if (start) begin
                        pc <= '0;
                        state <= SCHED_FETCH;
                    end
                end
                SCHED_FETCH: begin
                    case (current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB])
                        UOP_MATMUL: begin
                            pc <= pc + 1'b1;
                            state <= SCHED_WAIT_MATRIX;
                        end
                        UOP_HALT: begin
                            pc <= pc + 1'b1;
                            state <= SCHED_DONE;
                        end
                        default: begin
                            if (primitive_cmd_valid) begin
                                if (primitive_cmd_ready) begin
                                    pc <= pc + 1'b1;
                                    state <= SCHED_WAIT_LOCAL;
                                end
                            end else begin
                                pc <= pc + 1'b1;
                                state <= SCHED_FETCH;
                            end
                        end
                    endcase
                end
                SCHED_WAIT_MATRIX: begin
                    if (matrix_done) begin
                        state <= SCHED_FETCH;
                    end
                end
                SCHED_WAIT_LOCAL: begin
                    if (primitive_rsp_valid) begin
                        state <= SCHED_FETCH;
                    end
                end
                SCHED_DONE: begin
                    if (!start) begin
                        state <= SCHED_IDLE;
                    end
                end
                default: state <= SCHED_IDLE;
            endcase
        end
    end
endmodule
