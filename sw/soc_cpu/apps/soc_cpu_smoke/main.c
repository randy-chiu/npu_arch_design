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

static uint32_t digits_a_tile_sram[DIGITS_TILE_WORDS];
static uint32_t digits_b_tile_sram[DIGITS_TILE_WORDS];
static uint32_t digits_c_tile_sram[DIGITS_TILE_WORDS];
static int32_t digits_logits_sram[DIGITS_LOGITS_WORDS];

#if REAL_MNIST_CNN_FC1_TILE_ENABLED
static uint32_t real_mnist_cnn_fc1_tile_a_sram[REAL_MNIST_CNN_FC1_TILE_WORDS];
static uint32_t real_mnist_cnn_fc1_tile_b_sram[REAL_MNIST_CNN_FC1_TILE_WORDS];
static uint32_t real_mnist_cnn_fc1_tile_c_sram[REAL_MNIST_CNN_FC1_TILE_WORDS];
#endif

#if REAL_MNIST_CNN_FC1_K_STREAM_ENABLED
static uint32_t real_mnist_cnn_fc1_k_stream_a_sram[REAL_MNIST_CNN_FC1_K_STREAM_CHUNKS][REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS];
static uint32_t real_mnist_cnn_fc1_k_stream_b_sram[REAL_MNIST_CNN_FC1_K_STREAM_CHUNKS][REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS];
static uint32_t real_mnist_cnn_fc1_k_stream_c_sram[REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS];
#endif

#if REAL_MNIST_CNN_FC1_FULL_K_STREAM_ENABLED
static uint32_t real_mnist_cnn_fc1_full_k_stream_a_sram[REAL_MNIST_CNN_FC1_FULL_K_STREAM_CHUNKS][REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS];
static uint32_t real_mnist_cnn_fc1_full_k_stream_b_sram[REAL_MNIST_CNN_FC1_FULL_K_STREAM_CHUNKS][REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS];
static uint32_t real_mnist_cnn_fc1_full_k_stream_c_sram[REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_COUNT][REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS];
#endif

#if REAL_MNIST_CNN_FC2_ENABLED
static uint32_t real_mnist_cnn_fc2_a_tile_sram[REAL_MNIST_CNN_FC2_TILE_WORDS];
static uint32_t real_mnist_cnn_fc2_b_tile_sram[REAL_MNIST_CNN_FC2_TILE_WORDS];
static uint32_t real_mnist_cnn_fc2_c_tile_sram[REAL_MNIST_CNN_FC2_TILE_WORDS];
static int32_t real_mnist_cnn_fc2_logits_sram[REAL_MNIST_CNN_FC2_LOGITS_WORDS];
#endif

static soc_npu_job_desc_t job_desc;

static uint32_t ptr32(const void *ptr)
{
    return (uint32_t)(uintptr_t)ptr;
}

static void copy_words(uint32_t *dst, const uint32_t *src, uint32_t len)
{
    dma_copy_words(dst, src, len);
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

static void clear_digits_logits(void)
{
    for (uint32_t i = 0; i < DIGITS_LOGITS_WORDS; ++i) {
        digits_logits_sram[i] = 0;
    }
}

static void run_digits_tile(uint32_t tile)
{
    copy_words(digits_a_tile_sram, digits_tile_a[tile], DIGITS_TILE_WORDS);
    copy_words(digits_b_tile_sram, digits_tile_b[tile], DIGITS_TILE_WORDS);

    job_desc.op_type = SOC_NPU_JOB_OP_MATMUL;
    job_desc.program_addr = ptr32(matmul_program_sram);
    job_desc.program_words = MATMUL_PROGRAM_LEN;
    job_desc.input0_addr = ptr32(digits_a_tile_sram);
    job_desc.input0_words = DIGITS_TILE_WORDS;
    job_desc.input1_addr = ptr32(digits_b_tile_sram);
    job_desc.input1_words = DIGITS_TILE_WORDS;
    job_desc.output_addr = ptr32(digits_c_tile_sram);
    job_desc.output_words = DIGITS_TILE_WORDS;
    run_job();
}

static void accumulate_digits_tile(uint32_t tile)
{
    uint32_t n_offset = digits_tile_n_offsets[0][tile];
    for (uint32_t row = 0; row < 8u; ++row) {
        for (uint32_t col = 0; col < 8u; ++col) {
            uint32_t tile_idx = row * 8u + col;
            uint32_t logits_idx = row * DIGITS_CLASS_COLUMNS + n_offset + col;
            digits_logits_sram[logits_idx] += (int32_t)digits_c_tile_sram[tile_idx];
        }
    }
}

static int check_digits_logits(void)
{
    for (uint32_t i = 0; i < DIGITS_LOGITS_WORDS; ++i) {
        if (digits_logits_sram[i] != (int32_t)digits_expected_logits[i]) {
            test_status_fail_code(0x300u | ((uint32_t)i & 0xffu));
            return 0;
        }
    }
    return 1;
}

static uint32_t predict_digits_label(void)
{
    uint32_t best = 0u;
    int32_t best_value = digits_logits_sram[0];
    for (uint32_t cls = 1u; cls < DIGITS_CLASS_COUNT; ++cls) {
        if (digits_logits_sram[cls] > best_value) {
            best = cls;
            best_value = digits_logits_sram[cls];
        }
    }
    return best;
}

static int run_digits_classifier(void)
{
    clear_digits_logits();
    for (uint32_t tile = 0; tile < DIGITS_TILE_COUNT; ++tile) {
        run_digits_tile(tile);
        accumulate_digits_tile(tile);
    }
    if (!check_digits_logits()) {
        return 0;
    }
    if (predict_digits_label() != DIGITS_EXPECTED_LABEL) {
        test_status_fail_code(0x400u | (predict_digits_label() & 0xffu));
        return 0;
    }
    return 1;
}

#if REAL_MNIST_CNN_FC1_TILE_ENABLED
static int run_real_mnist_cnn_fc1_tile(void)
{
    copy_words(real_mnist_cnn_fc1_tile_a_sram, real_mnist_cnn_fc1_tile_a, REAL_MNIST_CNN_FC1_TILE_WORDS);
    copy_words(real_mnist_cnn_fc1_tile_b_sram, real_mnist_cnn_fc1_tile_b, REAL_MNIST_CNN_FC1_TILE_WORDS);

    job_desc.op_type = SOC_NPU_JOB_OP_MATMUL;
    job_desc.program_addr = ptr32(matmul_program_sram);
    job_desc.program_words = MATMUL_PROGRAM_LEN;
    job_desc.input0_addr = ptr32(real_mnist_cnn_fc1_tile_a_sram);
    job_desc.input0_words = REAL_MNIST_CNN_FC1_TILE_WORDS;
    job_desc.input1_addr = ptr32(real_mnist_cnn_fc1_tile_b_sram);
    job_desc.input1_words = REAL_MNIST_CNN_FC1_TILE_WORDS;
    job_desc.output_addr = ptr32(real_mnist_cnn_fc1_tile_c_sram);
    job_desc.output_words = REAL_MNIST_CNN_FC1_TILE_WORDS;
    run_job();

    if (!check_words(real_mnist_cnn_fc1_tile_c_sram, real_mnist_cnn_fc1_tile_expected_c,
                     REAL_MNIST_CNN_FC1_TILE_WORDS, 0x700u)) {
        return 0;
    }
    return 1;
}
#endif

#if REAL_MNIST_CNN_FC1_K_STREAM_ENABLED
static int run_real_mnist_cnn_fc1_k_stream(void)
{
    for (uint32_t chunk = 0; chunk < REAL_MNIST_CNN_FC1_K_STREAM_CHUNKS; ++chunk) {
        copy_words(real_mnist_cnn_fc1_k_stream_a_sram[chunk], real_mnist_cnn_fc1_k_stream_a[chunk],
                   REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS);
        copy_words(real_mnist_cnn_fc1_k_stream_b_sram[chunk], real_mnist_cnn_fc1_k_stream_b[chunk],
                   REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS);
    }

    job_desc.op_type = SOC_NPU_JOB_OP_MATMUL_K_STREAM;
    job_desc.program_addr = ptr32(matmul_program_sram);
    job_desc.program_words = MATMUL_PROGRAM_LEN;
    job_desc.input0_addr = ptr32(real_mnist_cnn_fc1_k_stream_a_sram);
    job_desc.input0_words = REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS;
    job_desc.input1_addr = ptr32(real_mnist_cnn_fc1_k_stream_b_sram);
    job_desc.input1_words = REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS;
    job_desc.output_addr = ptr32(real_mnist_cnn_fc1_k_stream_c_sram);
    job_desc.output_words = REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS;
    job_desc.k_chunks = REAL_MNIST_CNN_FC1_K_STREAM_CHUNKS;
    run_job();

    if (!check_words(real_mnist_cnn_fc1_k_stream_c_sram, real_mnist_cnn_fc1_k_stream_expected_c,
                     REAL_MNIST_CNN_FC1_K_STREAM_TILE_WORDS, 0x800u)) {
        return 0;
    }
    return 1;
}
#endif

#if REAL_MNIST_CNN_FC1_FULL_K_STREAM_ENABLED
static int run_real_mnist_cnn_fc1_full_k_stream(void)
{
    for (uint32_t chunk = 0; chunk < REAL_MNIST_CNN_FC1_FULL_K_STREAM_CHUNKS; ++chunk) {
        copy_words(real_mnist_cnn_fc1_full_k_stream_a_sram[chunk], real_mnist_cnn_fc1_full_k_stream_a[chunk],
                   REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS);
    }

    for (uint32_t tile = 0; tile < REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_COUNT; ++tile) {
        for (uint32_t chunk = 0; chunk < REAL_MNIST_CNN_FC1_FULL_K_STREAM_CHUNKS; ++chunk) {
            copy_words(real_mnist_cnn_fc1_full_k_stream_b_sram[chunk],
                       real_mnist_cnn_fc1_full_k_stream_b[tile][chunk],
                       REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS);
        }

        job_desc.op_type = SOC_NPU_JOB_OP_MATMUL_K_STREAM;
        job_desc.program_addr = ptr32(matmul_program_sram);
        job_desc.program_words = MATMUL_PROGRAM_LEN;
        job_desc.input0_addr = ptr32(real_mnist_cnn_fc1_full_k_stream_a_sram);
        job_desc.input0_words = REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS;
        job_desc.input1_addr = ptr32(real_mnist_cnn_fc1_full_k_stream_b_sram);
        job_desc.input1_words = REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS;
        job_desc.output_addr = ptr32(real_mnist_cnn_fc1_full_k_stream_c_sram[tile]);
        job_desc.output_words = REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS;
        job_desc.k_chunks = REAL_MNIST_CNN_FC1_FULL_K_STREAM_CHUNKS;
        run_job();

        if (!check_words(real_mnist_cnn_fc1_full_k_stream_c_sram[tile],
                         real_mnist_cnn_fc1_full_k_stream_expected_c[tile],
                         REAL_MNIST_CNN_FC1_FULL_K_STREAM_TILE_WORDS,
                         0x900u | ((tile & 0xfu) << 4))) {
            return 0;
        }
    }
    return 1;
}
#endif

#if REAL_MNIST_CNN_FC2_ENABLED
static void clear_real_mnist_cnn_fc2_logits(void)
{
    for (uint32_t i = 0; i < REAL_MNIST_CNN_FC2_LOGITS_WORDS; ++i) {
        real_mnist_cnn_fc2_logits_sram[i] = 0;
    }
}

static void run_real_mnist_cnn_fc2_tile(uint32_t tile)
{
    copy_words(real_mnist_cnn_fc2_a_tile_sram, real_mnist_cnn_fc2_tile_a[tile], REAL_MNIST_CNN_FC2_TILE_WORDS);
    copy_words(real_mnist_cnn_fc2_b_tile_sram, real_mnist_cnn_fc2_tile_b[tile], REAL_MNIST_CNN_FC2_TILE_WORDS);

    job_desc.op_type = SOC_NPU_JOB_OP_MATMUL;
    job_desc.program_addr = ptr32(matmul_program_sram);
    job_desc.program_words = MATMUL_PROGRAM_LEN;
    job_desc.input0_addr = ptr32(real_mnist_cnn_fc2_a_tile_sram);
    job_desc.input0_words = REAL_MNIST_CNN_FC2_TILE_WORDS;
    job_desc.input1_addr = ptr32(real_mnist_cnn_fc2_b_tile_sram);
    job_desc.input1_words = REAL_MNIST_CNN_FC2_TILE_WORDS;
    job_desc.output_addr = ptr32(real_mnist_cnn_fc2_c_tile_sram);
    job_desc.output_words = REAL_MNIST_CNN_FC2_TILE_WORDS;
    run_job();
}

static void accumulate_real_mnist_cnn_fc2_tile(uint32_t tile)
{
    uint32_t n_offset = real_mnist_cnn_fc2_tile_n_offsets[0][tile];
    for (uint32_t row = 0; row < 8u; ++row) {
        for (uint32_t col = 0; col < 8u; ++col) {
            uint32_t tile_idx = row * 8u + col;
            uint32_t logits_idx = row * REAL_MNIST_CNN_FC2_CLASS_COLUMNS + n_offset + col;
            real_mnist_cnn_fc2_logits_sram[logits_idx] += (int32_t)real_mnist_cnn_fc2_c_tile_sram[tile_idx];
        }
    }
}

static int check_real_mnist_cnn_fc2_scaled_logits(void)
{
    for (uint32_t cls = 0; cls < REAL_MNIST_CNN_FC2_CLASS_COUNT; ++cls) {
        int32_t actual = real_mnist_cnn_fc2_logits_sram[cls] + (int32_t)real_mnist_cnn_fc2_bias_scaled[cls];
        if (actual != (int32_t)real_mnist_cnn_fc2_expected_scaled_logits[cls]) {
            test_status_fail_code(0x500u | ((uint32_t)cls & 0xffu));
            return 0;
        }
    }
    return 1;
}

static uint32_t predict_real_mnist_cnn_fc2_label(void)
{
    uint32_t best = 0u;
    int32_t best_value = real_mnist_cnn_fc2_logits_sram[0] + (int32_t)real_mnist_cnn_fc2_bias_scaled[0];
    for (uint32_t cls = 1u; cls < REAL_MNIST_CNN_FC2_CLASS_COUNT; ++cls) {
        int32_t value = real_mnist_cnn_fc2_logits_sram[cls] + (int32_t)real_mnist_cnn_fc2_bias_scaled[cls];
        if (value > best_value) {
            best = cls;
            best_value = value;
        }
    }
    return best;
}

static int run_real_mnist_cnn_fc2(void)
{
    clear_real_mnist_cnn_fc2_logits();
    for (uint32_t tile = 0; tile < REAL_MNIST_CNN_FC2_TILE_COUNT; ++tile) {
        run_real_mnist_cnn_fc2_tile(tile);
        accumulate_real_mnist_cnn_fc2_tile(tile);
    }
    if (!check_real_mnist_cnn_fc2_scaled_logits()) {
        return 0;
    }
    if (predict_real_mnist_cnn_fc2_label() != REAL_MNIST_CNN_FC2_EXPECTED_LABEL) {
        test_status_fail_code(0x600u | (predict_real_mnist_cnn_fc2_label() & 0xffu));
        return 0;
    }
    return 1;
}
#endif

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

    if (!run_digits_classifier()) {
        return 1;
    }

#if REAL_MNIST_CNN_FC2_ENABLED
    #if REAL_MNIST_CNN_FC1_TILE_ENABLED
    if (!run_real_mnist_cnn_fc1_tile()) {
        return 1;
    }
    #endif

    #if REAL_MNIST_CNN_FC1_K_STREAM_ENABLED
    if (!run_real_mnist_cnn_fc1_k_stream()) {
        return 1;
    }
    #endif

    #if REAL_MNIST_CNN_FC1_FULL_K_STREAM_ENABLED
    if (!run_real_mnist_cnn_fc1_full_k_stream()) {
        return 1;
    }
    #endif

    if (!run_real_mnist_cnn_fc2()) {
        return 1;
    }
#endif

    test_status_pass();
    return 0;
}
