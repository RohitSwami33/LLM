// Block 3: vec_dot_q4_0_q8_0 — ggml-compatible layouts.
//   block_q4_0 { fp16 d; uint8 qs[16]; }  low nibble=val j (j&1==0), high=j odd
//   block_q8_0 { fp16 d; int8  qs[32]; }
// dot = dA*dB * sum_i(a_i-8)*b_i   (integer-exact per block)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdint>
#include <arm_neon.h>

#ifdef __ARM_FEATURE_DOTPROD
static const char* kPath = "neon dotprod (vdotq_s32)";
#else
static const char* kPath = "neon widening (vmull/vmlal)";
#endif

struct block_q4_0 { uint16_t d; uint8_t qs[16]; };
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(block_q4_0) == 18, "layout");
static_assert(sizeof(block_q8_0) == 34, "layout");

static inline float fp16_to_f(uint16_t h) {
    union { uint16_t u; __fp16 f; } v; v.u = h; return (float)v.f;
}
static inline uint16_t f_to_fp16(float f) {
    union { uint16_t u; __fp16 f; } v; v.f = (__fp16)f; return v.u;
}

// ---- quantization helpers (self-consistent; d = amax/8 vs amax/127) ----
static void quant_row_f32_q4_0(const float* x, int k, block_q4_0* out) {
    const int nb = k / 32;
    for (int b = 0; b < nb; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[(size_t)b * 32 + j]));
        const float d = amax / 8.f;
        out[b].d = f_to_fp16(d);
        const float id = d ? 1.f / d : 0.f;
        for (int j = 0; j < 32; ++j) {
            int q = (int)lrintf(x[(size_t)b * 32 + j] * id);
            q = q < -8 ? -8 : (q > 7 ? 7 : q);
            if (j < 16) out[b].qs[j]      = (uint8_t)((q + 8) & 0xF);                    // low nibble: elem j
            else        out[b].qs[j - 16] |= (uint8_t)(((q + 8) & 0xF) << 4);             // high nibble: elem j+16
        }
    }
}
static void quant_col_f32_q8_0(const float* x, int k, block_q8_0* out) {
    for (int b = 0; b < k / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[(size_t)b * 32 + j]));
        const float d = amax / 127.f;
        out[b].d = f_to_fp16(d);
        const float id = d ? 1.f / d : 0.f;
        for (int j = 0; j < 32; ++j)
            out[b].qs[j] = (int8_t)lrintf(x[(size_t)b * 32 + j] * id);
    }
}

// ---- reference: exact int64 math per block, fp16 scale combo ----
static float vec_dot_q4_0_q8_0_ref(const block_q4_0* a, const block_q8_0* q8b, int k) {
    float accf = 0.f;
    for (int bl = 0; bl < k / 32; ++bl) {
        const uint8_t* q4 = a[bl].qs;
        const int8_t* q8 = q8b[bl].qs;
        int64_t s = 0;
        for (int j = 0; j < 16; ++j) {
            const int lo = (int)(q4[j] & 0xF) - 8;
            const int hi = (int)(q4[j] >> 4) - 8;
            s += (int64_t)lo * q8[j] + (int64_t)hi * q8[j + 16];
        }
        accf += (float)s * fp16_to_f(a[bl].d) * fp16_to_f(q8b[bl].d);
    }
    return accf;
}

// ---- NEON kernel ----
#ifdef __ARM_FEATURE_DOTPROD
static inline int32x4_t dot32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    return vdotq_s32(acc, a, b);
}
#else
static inline int32x4_t dot32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    int16x8_t pl = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t ph = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(
            vaddl_s16(vget_low_s16(pl), vget_high_s16(pl)),
            vaddl_s16(vget_low_s16(ph), vget_high_s16(ph))));
}
#endif

static inline int8x16_t q4_nibbles(uint8x16_t u, bool high) {
    return high ? vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), vdupq_n_s8(8))
                : vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
}

static float vec_dot_q4_0_q8_0_neon(const block_q4_0* a, const block_q8_0* b, int k) {
    float acc = 0.f;
    const int nb = k / 32;
    for (int i = 0; i < nb; ++i) {
        uint8x16_t u = vld1q_u8(a[i].qs);
        int8x16_t a_lo = q4_nibbles(u, false);
        int8x16_t a_hi = q4_nibbles(u, true);
        int8x16_t b0 = vld1q_s8(b[i].qs);      // vals 0..15
        int8x16_t b1 = vld1q_s8(b[i].qs + 16); // vals 16..31
        int32x4_t s = vdupq_n_s32(0);
        s = dot32(s, a_lo, b0);
        s = dot32(s, a_hi, b1);
        acc += (float)vaddvq_s32(s) * fp16_to_f(a[i].d) * fp16_to_f(b[i].d);
    }
    return acc;
}

static bool test_correctness() {
    const int k = 256;
    std::vector<float> a(k), b(k);
    for (int j = 0; j < k; ++j) { a[j] = sinf(j * 0.7f) * 0.5f; b[j] = cosf(j * 0.3f) * 0.25f; }
    std::vector<block_q4_0> qa(k / 32);
    std::vector<block_q8_0> qb(k / 32);
    quant_row_f32_q4_0(a.data(), k, qa.data());
    quant_col_f32_q8_0(b.data(), k, qb.data());

    // double-precision truth over the quantized values
    double truth = 0.0;
    for (int bl = 0; bl < k / 32; ++bl) {
        double sdbl = 0.0;
        for (int j = 0; j < 32; ++j) {
            const int nib = (j < 16) ? (int)(qa[bl].qs[j] & 0xF) : (int)(qa[bl].qs[j - 16] >> 4);
            sdbl += (double)(nib - 8) * qb[bl].qs[j];
        }
        truth += sdbl * (double)fp16_to_f(qa[bl].d) * (double)fp16_to_f(qb[bl].d);
    }
    const float r1 = vec_dot_q4_0_q8_0_ref(qa.data(), qb.data(), k);
    const float r2 = vec_dot_q4_0_q8_0_neon(qa.data(), qb.data(), k);
    printf("[test] truth=%.6f ref=%.6f neon=%.6f\n", truth, r1, r2);
    if (fabs(r1 - (float)truth) > 1e-3f * fabs((float)truth) + 1e-5f) { printf("FAIL ref\n"); return false; }
    if (fabs(r2 - (float)truth) > 1e-3f * fabs((float)truth) + 1e-5f) { printf("FAIL neon\n"); return false; }
    printf("[test] q4_0 x q8_0 correctness OK (%s)\n", kPath);
    return true;
}

int main() {
    if (!test_correctness()) return 1;
    const int n = 2560, d = 2560, reps = 40; // 80 blocks/row
    std::vector<float> A((size_t)n * d), B(d);
    std::vector<block_q4_0> Aq((size_t)n * d / 32);
    std::vector<block_q8_0> Bq(d / 32);
    for (size_t i = 0; i < A.size(); ++i) A[i] = sinf((i % 97) * 0.13f) * 0.4f;
    for (int j = 0; j < d; ++j) B[j] = cosf((j % 53) * 0.11f) * 0.3f;
    for (int i = 0; i < n; ++i) quant_row_f32_q4_0(A.data() + (size_t)i * d, d, Aq.data() + (size_t)i * d / 32);
    quant_col_f32_q8_0(B.data(), d, Bq.data());

    auto t0 = std::chrono::steady_clock::now();
    float chk = 0.f;
    for (int r = 0; r < reps; ++r)
        for (int i = 0; i < n; ++i)
            chk += vec_dot_q4_0_q8_0_neon(Aq.data() + (size_t)i * d / 32, Bq.data(), d);
    auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count() / reps;
    printf("[bench] q4_0 x q8_0 matvec %dx%d: %.3f ms  %.2f GFLOP/s  (%s, chk=%.3f)\n",
           n, d, sec * 1e3, 2.0 * n * d / sec / 1e9, kPath, chk);
    return 0;
}
