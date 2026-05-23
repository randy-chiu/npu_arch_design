#include "npu_driver.h"

#include "npu_v0_regs.h"
#include "soc_v0_addr.h"

static inline volatile uint32_t *mmio32(uint32_t addr)
{
    return (volatile uint32_t *)addr;
}

enum {
    DMA_CTRL_OFFSET = 0x000u,
    DMA_STATUS_OFFSET = 0x004u,
    DMA_SRC_OFFSET = 0x008u,
    DMA_DST_OFFSET = 0x00cu,
    DMA_WORDS_OFFSET = 0x010u,
};

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
    *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_CTRL) = 1u;
}

uint32_t npu_status(void)
{
    return *mmio32(SOC_NPU_WRAPPER_BASE + NPU_OPSCHED_STATUS);
}

void npu_wait_done(void)
{
    while ((npu_status() & 1u) == 0u) {
    }
}

void dma_copy_words(uint32_t *dst, const uint32_t *src, uint32_t len)
{
    *mmio32(SOC_DMA_BASE + DMA_SRC_OFFSET) = (uint32_t)(uintptr_t)src;
    *mmio32(SOC_DMA_BASE + DMA_DST_OFFSET) = (uint32_t)(uintptr_t)dst;
    *mmio32(SOC_DMA_BASE + DMA_WORDS_OFFSET) = len;
    *mmio32(SOC_DMA_BASE + DMA_CTRL_OFFSET) = 1u;
    while ((*mmio32(SOC_DMA_BASE + DMA_STATUS_OFFSET) & 1u) == 0u) {
    }
}

void test_status_pass(void)
{
    *mmio32(SOC_TEST_STATUS_BASE) = 1u;
}

void test_status_fail(void)
{
    *mmio32(SOC_TEST_STATUS_BASE) = 0xffffffffu;
}

void test_status_fail_code(uint32_t code)
{
    *mmio32(SOC_TEST_STATUS_BASE) = 0x80000000u | code;
}
