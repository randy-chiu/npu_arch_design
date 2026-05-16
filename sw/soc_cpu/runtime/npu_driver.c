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
