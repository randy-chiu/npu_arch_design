#include <stdint.h>

#include "npu_driver.h"
#include "npu_v0_regs.h"
#include "soc_cpu_smoke_data.h"

static int check_words(uint32_t offset, const uint32_t *expected, uint32_t len, uint32_t fail_base)
{
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t actual;
        npu_read_words(offset + i * 4u, &actual, 1u);
        if (actual != expected[i]) {
            test_status_fail_code(fail_base | ((actual & 0xffu) << 8) | (expected[i] & 0xffu));
            return 0;
        }
    }
    return 1;
}

static int check_low_bytes(uint32_t offset, const uint32_t *expected, uint32_t len, uint32_t fail_base)
{
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t actual;
        npu_read_words(offset + i * 4u, &actual, 1u);
        if ((actual & 0xffu) != (expected[i] & 0xffu)) {
            test_status_fail_code(fail_base | ((actual & 0xffu) << 8) | (expected[i] & 0xffu));
            return 0;
        }
    }
    return 1;
}

int main(void)
{
    npu_write_words(NPU_OPSCHED_A_BASE, matmul_a, MATMUL_A_LEN);
    npu_write_words(NPU_OPSCHED_B_BASE, matmul_b, MATMUL_B_LEN);
    npu_write_words(NPU_OPSCHED_PROGRAM_BASE, matmul_program, MATMUL_PROGRAM_LEN);
    npu_start();
    npu_wait_done();

    if (!check_words(NPU_OPSCHED_C_BASE, matmul_expected_c, MATMUL_EXPECTED_C_LEN, 0x100u)) {
        return 1;
    }

    npu_write_words(NPU_OPSCHED_X_BASE, softmax_x, SOFTMAX_X_LEN);
    npu_write_words(NPU_OPSCHED_PROGRAM_BASE, softmax_program, SOFTMAX_PROGRAM_LEN);
    npu_start();
    npu_wait_done();

    if (!check_low_bytes(NPU_OPSCHED_Y_BASE, softmax_expected_y, SOFTMAX_EXPECTED_Y_LEN, 0x200u)) {
        return 1;
    }

    test_status_pass();
    return 0;
}
