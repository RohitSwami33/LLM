// Block 4: race ggml's own NEON vec_dot_q4_0_q8_0 (2 blocks/iter) vs mine (Block 3, 1 block/iter).
// ggml path extracted verbatim from ggml-cpu/arch/arm/quants.c + ggml-cpu-impl.h, macros trimmed.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdint>
#include <arm_neon.h>

#define UNUSED(x) (void)(x)
#define GGML_RESTRICT __restrict__

struct block_q4_0 { uint16_t d; uint8_t qs[16]; };
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
struct block_q8_0s { uint16_t d; int8_t qs[32]; int32_t sum; }; // precomputed sum(qs) for the -8 fold
static_assert(sizeof(block_q4_0) == 18, "layout");
static_assert(sizeof(block_q8_0) == 34, "layout");

static inline float fp16_to_f(uint16_t h) {
    union { uint16_t u; __fp16 f; } v; v.u = h; return (float)v.f;
}
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_f(x)

static inline int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#endif
}

// ---- quantizers (ggml layout: low nibble byte j = elem j, high = elem j+16) ----
static void quant_row_f32_q4_0(const float* x, int k, block_q4_0* out) {
    for (int b = 0; b < k / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[(size_t)b * 32 + j]));
        const float d = amax / 8.f;
        union { uint16_t u; __fp16 f; } cv; cv.f = (__fp16)d; out[b].d = cv.u;
        const float id = d ? 1.f / d : 0.f;
        for (int j = 0; j < 32; ++j) {
            int q = (int)lrintf(x[(size_t)b * 32 + j] * id);
            q = q < -8 ? -8 : (q > 7 ? 7 : q);
            if (j < 16) out[b].qs[j]      = (uint8_t)((q + 8) & 0xF);
            else        out[b].qs[j - 16] |= (uint8_t)(((q + 8) & 0xF) << 4);
        }
    }
}
static void quant_col_f32_q8_0(const float* x, int k, block_q8_0* out) {
    for (int b = 0; b < k / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[(size_t)b * 32 + j]));
        const float d = amax / 127.f;
        union { uint16_t u; __fp16 f; } cv; cv.f = (__fp16)d; out[b].d = cv.u;
        const float id = d ? 1.f / d : 0.f;
        for (int j = 0; j < 32; ++j) out[b].qs[j] = (int8_t)lrintf(x[(size_t)b * 32 + j] * id);
    }
}

// ---- mine: Block 3 kernel (1 block/iter) ----
#ifdef __ARM_FEATURE_DOTPROD
static inline int32x4_t dot32(int32x4_t acc, int8x16_t a, int8x16_t b) { return vdotq_s32(acc, a, b); }
#else
static inline int32x4_t dot32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    int16x8_t pl = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t ph = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(
            vaddl_s16(vget_low_s16(pl), vget_high_s16(pl)),
            vaddl_s16(vget_low_s16(ph), vget_high_s16(ph))));
}
#endif
static inline int8x16_t q4_nib(uint8x16_t u, bool high) {
    return high ? vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(u, 4)), vdupq_n_s8(8))
                : vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
}
static float vm1(const block_q4_0* a, const block_q8_0* b, int k) {
    float acc = 0.f;
    for (int i = 0; i < k / 32; ++i) {
        uint8x16_t u = vld1q_u8(a[i].qs);
        int8x16_t a_lo = q4_nib(u, false), a_hi = q4_nib(u, true);
        int8x16_t b0 = vld1q_s8(b[i].qs), b1 = vld1q_s8(b[i].qs + 16);
        int32x4_t s = vdupq_n_s32(0);
        s = dot32(s, a_lo, b0); s = dot32(s, a_hi, b1);
        acc += (float)vaddvq_s32(s) * fp16_to_f(a[i].d) * fp16_to_f(b[i].d);
    }
    return acc;
}

static void quant_col_f32_q8_0s(const float* x, int k, block_q8_0s* out) {
    for (int b = 0; b < k / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[(size_t)b * 32 + j]));
        const float d = amax / 127.f;
        union { uint16_t u; __fp16 f; } cv; cv.f = (__fp16)d; out[b].d = cv.u;
        const float id = d ? 1.f / d : 0.f;
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            out[b].qs[j] = (int8_t)lrintf(x[(size_t)b * 32 + j] * id);
            sum += out[b].qs[j];
        }
        out[b].sum = sum;
    }
}

// ---- mine, v2 (Block 5): 4 blocks/iter, 4 independent fp32 accumulators ----
static float vm4(const block_q4_0* a, const block_q8_0* b, int k) {
    const int nb = k / 32;
    float32x4_t sv0 = vdupq_n_f32(0.f), sv1 = vdupq_n_f32(0.f),
                sv2 = vdupq_n_f32(0.f), sv3 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 4 <= nb; i += 4) {
        float scale[4];
        int32x4_t p[4];
        uint8x16_t u[4];
        for (int j = 0; j < 4; ++j) {
            const block_q4_0* xj = &a[i + j];
            const block_q8_0* yj = &b[i + j];
            u[j] = vld1q_u8(xj->qs);
            int8x16_t al = q4_nib(u[j], false), ah = q4_nib(u[j], true);
            int8x16_t b0 = vld1q_s8(yj->qs), b1 = vld1q_s8(yj->qs + 16);
            int32x4_t s = vdupq_n_s32(0);
            s = dot32(s, al, b0); s = dot32(s, ah, b1);
            p[j] = s;
            scale[j] = fp16_to_f(xj->d) * fp16_to_f(yj->d);
        }
        sv0 = vmlaq_n_f32(sv0, vcvtq_f32_s32(p[0]), scale[0]);
        sv1 = vmlaq_n_f32(sv1, vcvtq_f32_s32(p[1]), scale[1]);
        sv2 = vmlaq_n_f32(sv2, vcvtq_f32_s32(p[2]), scale[2]);
        sv3 = vmlaq_n_f32(sv3, vcvtq_f32_s32(p[3]), scale[3]);
    }
    float acc = vaddvq_f32(sv0) + vaddvq_f32(sv1) + vaddvq_f32(sv2) + vaddvq_f32(sv3);
    for (; i < nb; ++i) {
        uint8x16_t u = vld1q_u8(a[i].qs);
        int8x16_t al = q4_nib(u, false), ah = q4_nib(u, true);
        int8x16_t b0 = vld1q_s8(b[i].qs), b1 = vld1q_s8(b[i].qs + 16);
        int32x4_t s = vdupq_n_s32(0);
        s = dot32(s, al, b0); s = dot32(s, ah, b1);
        acc += (float)vaddvq_s32(s) * fp16_to_f(a[i].d) * fp16_to_f(b[i].d);
    }
    return acc;
}

// ---- mine, v3 (Block 6): unsigned-nibble SDOT, -8 folded via precomputed sum(qs) ----
// identity: dot(a-8, b) == dot(nibbles, b) - 8*sum(b)   (a-8 == nibble-8, values 0..15 int8-positive)
static float vm5(const block_q4_0* a, const block_q8_0s* b, int k) {
    const int nb = k / 32;
    float32x4_t sv0 = vdupq_n_f32(0.f), sv1 = vdupq_n_f32(0.f),
                sv2 = vdupq_n_f32(0.f), sv3 = vdupq_n_f32(0.f);
    float corr = 0.f;
    int i = 0;
    for (; i + 4 <= nb; i += 4) {
        float scale[4], c8[4];
        int32x4_t p[4];
        for (int j = 0; j < 4; ++j) {
            const block_q4_0* xj = &a[i + j];
            const block_q8_0s* yj = &b[i + j];
            uint8x16_t u = vld1q_u8(xj->qs);
            int8x16_t al = vreinterpretq_s8_u8(vandq_u8(u, vdupq_n_u8(0x0F)));    // nibble = a+8, int8 >= 0
            int8x16_t ah = vreinterpretq_s8_u8(vshrq_n_u8(u, 4));
            int8x16_t b0 = vld1q_s8(yj->qs), b1 = vld1q_s8(yj->qs + 16);
            int32x4_t s = vdupq_n_s32(0);
            s = dot32(s, al, b0); s = dot32(s, ah, b1);
            p[j] = s;
            const float s_j = fp16_to_f(xj->d) * fp16_to_f(yj->d);
            scale[j] = s_j;
            c8[j] = 8.f * (float)yj->sum * s_j;
        }
        sv0 = vmlaq_n_f32(sv0, vcvtq_f32_s32(p[0]), scale[0]);
        sv1 = vmlaq_n_f32(sv1, vcvtq_f32_s32(p[1]), scale[1]);
        sv2 = vmlaq_n_f32(sv2, vcvtq_f32_s32(p[2]), scale[2]);
        sv3 = vmlaq_n_f32(sv3, vcvtq_f32_s32(p[3]), scale[3]);
        corr += c8[0] + c8[1] + c8[2] + c8[3];
    }
    float acc = vaddvq_f32(sv0) + vaddvq_f32(sv1) + vaddvq_f32(sv2) + vaddvq_f32(sv3) - corr;
    for (; i < nb; ++i) {
        uint8x16_t u = vld1q_u8(a[i].qs);
        int8x16_t al = q4_nib(u, false), ah = q4_nib(u, true);
        int8x16_t b0 = vld1q_s8(b[i].qs), b1 = vld1q_s8(b[i].qs + 16);
        int32x4_t s = vdupq_n_s32(0);
        s = dot32(s, al, b0); s = dot32(s, ah, b1);
        acc += ((float)vaddvq_s32(s) - 8.f * (float)b[i].sum) * fp16_to_f(a[i].d) * fp16_to_f(b[i].d);
    }
    return acc;
}

// ---- mine v4 (Block 7): dual-row i8mm vmmla, 2 A-rows share B (GQA-style) ----
#if defined(__ARM_FEATURE_MATMUL_INT8)
static void vm6(const block_q4_0* a0, const block_q4_0* a1, const block_q8_0* b, int k, float* out2) {
    const int nb = k / 32;
    float32x4_t sumv = vdupq_n_f32(0.f);
    for (int i = 0; i < nb; i++) {
        const block_q4_0* b_x0 = &a0[i];
        const block_q4_0* b_x1 = &a1[i];
        const block_q8_0* b_y0 = &b[i];
        const block_q8_0* b_y1 = &b[i];

        const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, vdupq_n_u8(0x0F)));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, vdupq_n_u8(0x0F)));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));
        const int8x16_t x0_l = vsubq_s8(v0_0l, vdupq_n_s8(8));
        const int8x16_t x0_h = vsubq_s8(v0_0h, vdupq_n_s8(8));
        const int8x16_t x1_l = vsubq_s8(v0_1l, vdupq_n_s8(8));
        const int8x16_t x1_h = vsubq_s8(v0_1h, vdupq_n_s8(8));

        const int8x16_t y0_l = vld1q_s8(b_y0->qs);
        const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
        const int8x16_t y1_l = vld1q_s8(b_y1->qs);
        const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

        float32_t _scale[4] = {
            fp16_to_f(b_x0->d)*fp16_to_f(b_y0->d),
            fp16_to_f(b_x0->d)*fp16_to_f(b_y1->d),
            fp16_to_f(b_x1->d)*fp16_to_f(b_y0->d),
            fp16_to_f(b_x1->d)*fp16_to_f(b_y1->d)
        };
        float32x4_t scale = vld1q_f32(_scale);

        int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
        int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
        int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
        int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

        int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
        int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
        int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
        int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

        int32x4_t acc = vdupq_n_s32(0);
        acc = vmmlaq_s32(acc, l0, r0);
        acc = vmmlaq_s32(acc, l1, r1);
        acc = vmmlaq_s32(acc, l2, r2);
        acc = vmmlaq_s32(acc, l3, r3);
        sumv = vmlaq_f32(sumv, vcvtq_f32_s32(acc), scale);
    }

    float32x4_t sumv1 = vextq_f32(sumv, sumv, 2);
    float32x4_t sumv2 = vzip1q_f32(sumv, sumv1);
    vst1_f32(out2,       vget_low_f32(sumv2));
    vst1_f32(out2 + 1,   vget_high_f32(sumv2));
    (void)out2;
}
#endif

// ---- ggml upstream NEON single-row path (extracted verbatim) ----
static float vm_ggml(const block_q4_0* x, const block_q8_0* y, int n) {
    const int qk = 32; // QK8_0
    const int nb = n / qk;
    int ib = 0;
    float sumf = 0;
    {
        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t sumv1 = vdupq_n_f32(0.0f);

        for (; ib + 1 < nb; ib += 2) {
            const block_q4_0 * GGML_RESTRICT x0 = &x[ib + 0];
            const block_q4_0 * GGML_RESTRICT x1 = &x[ib + 1];
            const block_q8_0 * GGML_RESTRICT y0 = &y[ib + 0];
            const block_q8_0 * GGML_RESTRICT y1 = &y[ib + 1];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(x1->qs);
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));
            const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
            const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
            const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
            const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

            const int8x16_t v1_0l = vld1q_s8(y0->qs);
            const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
            const int8x16_t v1_1l = vld1q_s8(y1->qs);
            const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

            const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
            const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_CPU_FP16_TO_FP32(x0->d)*GGML_CPU_FP16_TO_FP32(y0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_CPU_FP16_TO_FP32(x1->d)*GGML_CPU_FP16_TO_FP32(y1->d));
        }

        sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    }
    for (; ib < nb; ++ib) {
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;
            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }
        sumf += (sumi0 + sumi1) * GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
    }
    return sumf;
}

static bool test_correctness() {
    const int k = 256;
    std::vector<float> a(k), b(k);
    for (int j = 0; j < k; ++j) { a[j] = sinf(j * 0.7f) * 0.5f; b[j] = cosf(j * 0.3f) * 0.25f; }
    std::vector<block_q4_0> qa(k / 32);
    std::vector<block_q8_0> qb(k / 32);
    std::vector<block_q8_0s> qbs(k / 32);
    quant_row_f32_q4_0(a.data(), k, qa.data());
    quant_col_f32_q8_0(b.data(), k, qb.data());
    quant_col_f32_q8_0s(b.data(), k, qbs.data());

    double truth = 0.0;
    for (int bl = 0; bl < k / 32; ++bl) {
        double s = 0.0;
        for (int j = 0; j < 32; ++j) {
            const int nib = (j < 16) ? (int)(qa[bl].qs[j] & 0xF) : (int)(qa[bl].qs[j - 16] >> 4);
            s += (double)(nib - 8) * qb[bl].qs[j];
        }
        truth += s * (double)fp16_to_f(qa[bl].d) * (double)fp16_to_f(qb[bl].d);
    }
    const float r1 = vm1(qa.data(), qb.data(), k);
    const float r2 = vm_ggml(qa.data(), qb.data(), k);
    const float r3 = vm4(qa.data(), qb.data(), k);
    const float r5 = vm5(qa.data(), qbs.data(), k);
    std::vector<float> a2(k);
    for (int j = 0; j < k; ++j) a2[j] = sinf(j * 0.9f + 1.0f) * 0.4f;
    std::vector<block_q4_0> qa2(k / 32);
    quant_row_f32_q4_0(a2.data(), k, qa2.data());
    double truth2 = 0.0;
    for (int bl = 0; bl < k / 32; ++bl) {
        double s = 0.0;
        for (int j = 0; j < 32; ++j) {
            const int nib = (j < 16) ? (int)(qa2[bl].qs[j] & 0xF) : (int)(qa2[bl].qs[j - 16] >> 4);
            s += (double)(nib - 8) * qb[bl].qs[j];
        }
        truth2 += s * (double)fp16_to_f(qa2[bl].d) * (double)fp16_to_f(qb[bl].d);
    }
    float r6[2] = { 0.f, 0.f };
#if defined(__ARM_FEATURE_MATMUL_INT8)
    vm6(qa.data(), qa2.data(), qb.data(), k, r6);
    printf("[test] vm6 rows: %.6f/%.6f (truth 1: %.6f, 2: %.6f)\n", r6[0], r6[1], truth, truth2);
#endif
    printf("[test] truth=%.6f mine=%.6f ggml=%.6f mine4=%.6f mine5=%.6f\n", truth, r1, r2, r3, r5);
    const float tol = 1e-3f * (float)fabs(truth) + 1e-5f;
    if (fabs(r1 - truth) > tol || fabs(r2 - truth) > tol || fabs(r3 - truth) > tol || fabs(r5 - truth) > tol) { printf("FAIL\n"); return false; }
#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (fabs(r6[0] - truth) > tol || fabs(r6[1] - truth2) > tol) { printf("FAIL vm6 row mismatch\n"); return false; }
#endif
    printf("[test] q4_0 x q8_0 race-correctness OK\n");
    return true;
}

int main() {
    if (!test_correctness()) return 1;
    const int n = 2560, d = 2560, reps = 40;
    std::vector<float> A((size_t)n * d), B(d);
    std::vector<block_q4_0> Aq((size_t)n * d / 32);
    std::vector<block_q8_0> Bq(d / 32);
    for (size_t i = 0; i < A.size(); ++i) A[i] = sinf((i % 97) * 0.13f) * 0.4f;
    for (int j = 0; j < d; ++j) B[j] = cosf((j % 53) * 0.11f) * 0.3f;
    std::vector<block_q8_0s> Bqs(d / 32);
    for (int i = 0; i < n; ++i) quant_row_f32_q4_0(A.data() + (size_t)i * d, d, Aq.data() + (size_t)i * d / 32);
    quant_col_f32_q8_0(B.data(), d, Bq.data());
    quant_col_f32_q8_0s(B.data(), d, Bqs.data());

    float c, chk0 = 0.f, chk1 = 0.f, chk3 = 0.f, chk5 = 0.f, chk6 = 0.f;
    auto run = [&](int which, double& sec, float& chk) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
        float r6[2];
#endif
        for (int i = 0; i < 3; ++i) c = which == 2 ? vm4(Aq.data(), Bq.data(), d)
                                                   : which == 3 ? vm5(Aq.data(), Bqs.data(), d)
#if defined(__ARM_FEATURE_MATMUL_INT8)
                                                   : which == 4 ? (vm6(Aq.data(), Aq.data() + d / 32, Bq.data(), d, r6), r6[0])
#endif
                                                   : which ? vm_ggml(Aq.data(), Bq.data(), d)
                                                           : vm1(Aq.data(), Bq.data(), d);
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            if (which == 4) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
                for (int i = 0; i < n; i += 2) {
                    vm6(Aq.data() + (size_t)i * d / 32, Aq.data() + (size_t)(i + 1) * d / 32, Bq.data(), d, r6);
                    chk += r6[0] + r6[1]; // same total MACs as single-row kernels
                }
#endif
            } else {
                for (int i = 0; i < n; ++i) {
                    c = which == 2 ? vm4(Aq.data() + (size_t)i * d / 32, Bq.data(), d)
                                   : which == 3 ? vm5(Aq.data() + (size_t)i * d / 32, Bqs.data(), d)
                                                : which ? vm_ggml(Aq.data() + (size_t)i * d / 32, Bq.data(), d)
                                                        : vm1(Aq.data() + (size_t)i * d / 32, Bq.data(), d);
                    chk += c; // loop-carried: every rep is observable, defeats redundancy elimination
                }
            }
        auto t1 = std::chrono::steady_clock::now();
        sec = std::chrono::duration<double>(t1 - t0).count() / reps;
    };
    double s1, s2, s3, s5, s6;
    run(0, s1, chk0); run(1, s2, chk1); run(2, s3, chk3); run(3, s5, chk5); run(4, s6, chk6); // sweep 1
    run(0, s1, chk0); run(1, s2, chk1); run(2, s3, chk3); run(3, s5, chk5); run(4, s6, chk6); // sweep 2
    double gf = 2.0 * n * d / 1e9;
    printf("[bench] mine1 : %8.3f ms  %7.2f GFLOP/s (chk=%.3f)\n", s1*1e3, gf/s1, chk0);
    printf("[bench] ggml2 : %8.3f ms  %7.2f GFLOP/s (chk=%.3f)\n", s2*1e3, gf/s2, chk1);
    printf("[bench] mine4 : %8.3f ms  %7.2f GFLOP/s (chk=%.3f)\n", s3*1e3, gf/s3, chk3);
    printf("[bench] mine5 : %8.3f ms  %7.2f GFLOP/s (chk=%.3f)\n", s5*1e3, gf/s5, chk5);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    printf("[bench] vm6-i8: %8.3f ms  %7.2f GFLOP/s (chk=%.3f)\n", s6*1e3, gf/s6, chk6);
#endif
    return 0;
}
