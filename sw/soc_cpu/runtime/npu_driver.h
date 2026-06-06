#ifndef NPU_DRIVER_H
#define NPU_DRIVER_H

#include <stddef.h>
#include <stdint.h>

void npu_write_words(uint32_t offset, const uint32_t *values, size_t len);
void npu_read_words(uint32_t offset, uint32_t *values, size_t len);
void npu_set_desc_addr(uint32_t addr);
void npu_start(void);
void npu_wait_done(void);
uint32_t npu_status(void);

typedef struct {
    uint32_t status;
    uint32_t job_id;
    uint32_t op_type;
    uint32_t total_cycles;
    uint32_t core_active_cycles;
    uint32_t core_matmul_cycles;
    uint32_t data_mover_active_cycles;
    uint32_t data_mover_setup_cycles;
    uint32_t data_mover_transfer_cycles;
    uint32_t data_mover_stall_cycles;
    uint32_t data_mover_words;
    uint32_t data_mover_read_words;
    uint32_t data_mover_write_words;
    uint32_t sram_read_words;
    uint32_t sram_write_words;
    uint32_t command_active_cycles;
    uint32_t command_wait_cycles;
    uint32_t data_mover_compute_overlap_cycles;
    uint32_t uop_scheduler_active_cycles;
    uint32_t uop_scheduler_wait_cycles;
    uint32_t core_wait_data_cycles;
    uint32_t core_local_active_cycles;
    uint32_t data_mover_program_cycles;
    uint32_t data_mover_initial_input_cycles;
    uint32_t data_mover_prefetch_cycles;
    uint32_t data_mover_output_cycles;
} npu_perf_snapshot_t;

void npu_read_perf_snapshot(npu_perf_snapshot_t *snapshot);

void dma_copy_words(uint32_t *dst, const uint32_t *src, uint32_t len);

void test_status_pass(void);
void test_status_fail(void);
void test_status_fail_code(uint32_t code);

#endif
