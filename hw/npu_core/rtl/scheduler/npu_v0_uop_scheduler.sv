module npu_v0_uop_scheduler (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,

    output logic [$clog2(RTL_HOST_PROGRAM_WORDS)-1:0] program_addr,
    input  logic [31:0] program_rdata,

    output logic        local_load_valid,
    output logic        local_store_valid,
    output logic        local_exec_valid,
    output logic [3:0]  local_opcode,
    output logic [3:0]  local_tensor,
    output logic [3:0]  local_buffer,
    input  logic        local_exec_blocking,
    input  logic        local_exec_done,
    output logic        matrix_start,
    input  logic        matrix_done,

    output logic        perf_active,
    output logic        perf_wait
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
    assign local_tensor = current_instr[UOP_ARG0_MSB:UOP_ARG0_LSB];
    assign local_buffer = current_instr[UOP_ARG1_MSB:UOP_ARG1_LSB];
    assign local_opcode = current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB];
    assign local_load_valid =
        state == SCHED_FETCH &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] == UOP_LOAD;
    assign local_store_valid =
        state == SCHED_FETCH &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] == UOP_STORE;
    assign local_exec_valid =
        state == SCHED_FETCH &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] != UOP_LOAD &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] != UOP_STORE &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] != UOP_MATMUL &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] != UOP_HALT;
    assign matrix_start =
        state == SCHED_FETCH &&
        current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB] == UOP_MATMUL;
    assign done = state == SCHED_DONE;
    assign perf_active = state == SCHED_FETCH || state == SCHED_DONE;
    assign perf_wait = state == SCHED_WAIT_MATRIX || state == SCHED_WAIT_LOCAL;

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
                    pc <= pc + 1'b1;
                    case (current_instr[UOP_OPCODE_MSB:UOP_OPCODE_LSB])
                        UOP_MATMUL: state <= SCHED_WAIT_MATRIX;
                        UOP_HALT: state <= SCHED_DONE;
                        default: begin
                            if (local_exec_valid && local_exec_blocking) begin
                                state <= SCHED_WAIT_LOCAL;
                            end else begin
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
                    if (local_exec_done) begin
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
