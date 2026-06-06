#include "npu_driver.h"

#include "npu_v0_regs.h"
#include "soc_v0_addr.h"

static inline volatile uint32_t *mmio32(uint32_t addr)
{
    return (volatile uint32_t *)addr;
}

void npu_write_words(uint32_t offset, const uint32_t *values, size_t len)
{
    volatile uint32_t *base = mmio32(SOC_NPU_WRAPPER_BASE + offset);
    for (size_t i = 0; i < len; ++i) {
        base[i] = values[i];
    }
}

void npu_read_words(uint32_t offset, uint32_t *values, size_t len)
{
    volatile uint32_t *base = mmio32(SOC_NPU_WRAPPER_BASE + offset);
    for (size_t i = 0; i < len; ++i) {
        values[i] = base[i];
    }
}

void npu_set_desc_addr(uint32_t addr)
{
    *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_DESC_ADDR) = addr;
}

void npu_start(void)
{
    *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_CTRL) = NPU_OPSCHED_CTRL_START_MASK;
}

uint32_t npu_status(void)
{
    return *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_STATUS);
}

void npu_wait_done(void)
{
    while ((npu_status() & NPU_OPSCHED_STATUS_DONE_MASK) == 0u) {
    }
}

void npu_read_perf_snapshot(npu_perf_snapshot_t *snapshot)
{
    snapshot->status = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_STATUS);
    snapshot->job_id = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_JOB_ID);
    snapshot->op_type = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_OP_TYPE);
    snapshot->total_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_TOTAL_CYCLES);
    snapshot->core_active_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_CORE_ACTIVE_CYCLES);
    snapshot->core_matmul_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_CORE_MATMUL_CYCLES);
    snapshot->data_mover_active_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DATA_MOVER_ACTIVE_CYCLES);
    snapshot->data_mover_setup_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DATA_MOVER_SETUP_CYCLES);
    snapshot->data_mover_transfer_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DATA_MOVER_TRANSFER_CYCLES);
    snapshot->data_mover_stall_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DATA_MOVER_STALL_CYCLES);
    snapshot->data_mover_words = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DATA_MOVER_WORDS);
    snapshot->data_mover_read_words = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DATA_MOVER_READ_WORDS);
    snapshot->data_mover_write_words = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DATA_MOVER_WRITE_WORDS);
    snapshot->sram_read_words = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_SRAM_READ_WORDS);
    snapshot->command_active_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_CMD_ACTIVE_CYCLES);
    snapshot->command_wait_cycles = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_CMD_WAIT_CYCLES);
    snapshot->data_mover_compute_overlap_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DM_COMPUTE_OVERLAP_CYCLES);
    snapshot->uop_scheduler_active_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_UOP_SCHED_ACTIVE_CYCLES);
    snapshot->uop_scheduler_wait_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_UOP_SCHED_WAIT_CYCLES);
    snapshot->core_wait_data_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_CORE_WAIT_DATA_CYCLES);
    snapshot->core_local_active_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_CORE_LOCAL_ACTIVE_CYCLES);
    snapshot->data_mover_program_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DM_PROGRAM_CYCLES);
    snapshot->data_mover_initial_input_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DM_INITIAL_INPUT_CYCLES);
    snapshot->data_mover_prefetch_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DM_PREFETCH_CYCLES);
    snapshot->data_mover_output_cycles =
        *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_DM_OUTPUT_CYCLES);
    snapshot->sram_write_words = *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_PERF_SRAM_WRITE_WORDS);
}

void dma_copy_words(uint32_t *dst, const uint32_t *src, uint32_t len)
{
    *mmio32(SOC_DMA_BASE + SOC_DMA_SRC_OFFSET) = (uint32_t)(uintptr_t)src;
    *mmio32(SOC_DMA_BASE + SOC_DMA_DST_OFFSET) = (uint32_t)(uintptr_t)dst;
    *mmio32(SOC_DMA_BASE + SOC_DMA_WORDS_OFFSET) = len;
    *mmio32(SOC_DMA_BASE + SOC_DMA_CTRL_OFFSET) = SOC_DMA_CTRL_START_MASK;
    while ((*mmio32(SOC_DMA_BASE + SOC_DMA_STATUS_OFFSET) & SOC_DMA_STATUS_DONE_MASK) == 0u) {
    }
}

void test_status_pass(void)
{
    *mmio32(SOC_TEST_STATUS_BASE) = SOC_TEST_STATUS_PASS_VALUE;
}

void test_status_fail(void)
{
    *mmio32(SOC_TEST_STATUS_BASE) = SOC_TEST_STATUS_FAIL_VALUE;
}

void test_status_fail_code(uint32_t code)
{
    *mmio32(SOC_TEST_STATUS_BASE) = SOC_TEST_STATUS_FAIL_CODE_FLAG_VALUE | code;
}
