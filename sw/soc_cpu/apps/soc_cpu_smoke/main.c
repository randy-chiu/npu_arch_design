#include <stdint.h>

#include "npu_driver.h"
#include "npu_v0_regs.h"
#include "soc_v0_addr.h"
#include "soc_cpu_smoke_data.h"

static uint32_t matmul_a_sram[MATMUL_A_LEN];
static uint32_t matmul_b_sram[MATMUL_B_LEN];
static uint32_t matmul_program_sram[MATMUL_PROGRAM_LEN];
static uint32_t matmul_c_sram[MATMUL_EXPECTED_C_LEN];

static uint32_t softmax_x_sram[SOFTMAX_X_LEN];
static uint32_t softmax_program_sram[SOFTMAX_PROGRAM_LEN];
static uint32_t softmax_y_sram[SOFTMAX_EXPECTED_Y_LEN];

static soc_npu_job_desc_t job_desc;

static uint32_t ptr32(const void *ptr)
{
    return (uint32_t)(uintptr_t)ptr;
}

static void copy_words(uint32_t *dst, const uint32_t *src, uint32_t len)
{
    for (uint32_t i = 0; i < len; ++i) {
        dst[i] = src[i];
    }
}

static int check_words(const uint32_t *actual, const uint32_t *expected, uint32_t len, uint32_t fail_base)
{
    for (uint32_t i = 0; i < len; ++i) {
        if (actual[i] != expected[i]) {
            test_status_fail_code(fail_base | ((actual[i] & 0xffu) << 8) | (expected[i] & 0xffu));
            return 0;
        }
    }
    return 1;
}

static int check_low_bytes(const uint32_t *actual, const uint32_t *expected, uint32_t len, uint32_t fail_base)
{
    for (uint32_t i = 0; i < len; ++i) {
        if ((actual[i] & 0xffu) != (expected[i] & 0xffu)) {
            test_status_fail_code(fail_base | ((actual[i] & 0xffu) << 8) | (expected[i] & 0xffu));
            return 0;
        }
    }
    return 1;
}

static void run_job(void)
{
    npu_set_desc_addr(ptr32(&job_desc));
    npu_start();
    npu_wait_done();
}

int main(void)
{
    copy_words(matmul_a_sram, matmul_a, MATMUL_A_LEN);
    copy_words(matmul_b_sram, matmul_b, MATMUL_B_LEN);
    copy_words(matmul_program_sram, matmul_program, MATMUL_PROGRAM_LEN);

    job_desc.op_type = SOC_NPU_JOB_OP_MATMUL;
    job_desc.program_addr = ptr32(matmul_program_sram);
    job_desc.program_words = MATMUL_PROGRAM_LEN;
    job_desc.input0_addr = ptr32(matmul_a_sram);
    job_desc.input0_words = MATMUL_A_LEN;
    job_desc.input1_addr = ptr32(matmul_b_sram);
    job_desc.input1_words = MATMUL_B_LEN;
    job_desc.output_addr = ptr32(matmul_c_sram);
    job_desc.output_words = MATMUL_EXPECTED_C_LEN;
    run_job();

    if (!check_words(matmul_c_sram, matmul_expected_c, MATMUL_EXPECTED_C_LEN, 0x100u)) {
        return 1;
    }

    copy_words(softmax_x_sram, softmax_x, SOFTMAX_X_LEN);
    copy_words(softmax_program_sram, softmax_program, SOFTMAX_PROGRAM_LEN);

    job_desc.op_type = SOC_NPU_JOB_OP_SOFTMAX;
    job_desc.program_addr = ptr32(softmax_program_sram);
    job_desc.program_words = SOFTMAX_PROGRAM_LEN;
    job_desc.input0_addr = ptr32(softmax_x_sram);
    job_desc.input0_words = SOFTMAX_X_LEN;
    job_desc.input1_addr = 0u;
    job_desc.input1_words = 0u;
    job_desc.output_addr = ptr32(softmax_y_sram);
    job_desc.output_words = SOFTMAX_EXPECTED_Y_LEN;
    run_job();

    if (!check_low_bytes(softmax_y_sram, softmax_expected_y, SOFTMAX_EXPECTED_Y_LEN, 0x200u)) {
        return 1;
    }

    test_status_pass();
    return 0;
}
