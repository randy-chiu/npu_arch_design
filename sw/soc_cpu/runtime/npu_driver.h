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

void dma_copy_words(uint32_t *dst, const uint32_t *src, uint32_t len);

void test_status_pass(void);
void test_status_fail(void);
void test_status_fail_code(uint32_t code);

#endif
