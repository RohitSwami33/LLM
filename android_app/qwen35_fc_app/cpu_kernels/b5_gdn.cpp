// b5_gdn.cpp - Block 8: Gated DeltaNet single-token step, fp32, 1 thread
// kernel A mirrors ggml-cpu/ops.cpp one_chunk (kda=0, K=1): memcpy-in + scale pass
//   + delta dot pass + rank-1 mad pass + attn dot pass  (~8 state-traffic units)
// kernel B is the fused row pass: per row read once for delta, then update+attn-dot
//   in registers, write once  (~2 state-traffic units)
// shapes from model metadata: S=128, H=32 (posttrained qwen35 4B gguf)

#include <arm_neon.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static inline float dot16(const float* a, const float* b, int n) {
    float32x4_t s0 = vdupq_n_f32(0.f), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t x0 = vld1q_f32(a + i), y0 = vld1q_f32(b + i);
        float32x4_t x1 = vld1q_f32(a + i + 4), y1 = vld1q_f32(b + i + 4);
        float32x4_t x2 = vld1q_f32(a + i + 8), y2 = vld1q_f32(b + i + 8);
        float32x4_t x3 = vld1q_f32(a + i + 12), y3 = vld1q_f32(b + i + 12);
        s0 = vfmaq_f32(s0, x0, y0);
        s1 = vfmaq_f32(s1, x1, y1);
        s2 = vfmaq_f32(s2, x2, y2);
        s3 = vfmaq_f32(s3, x3, y3);
    }
    float t = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; ++i) t += a[i] * b[i];
    return t;
}

static void gdn_tree(float* s_out, const float* s_in, const float* k, const float* q,
                     const float* v, float g, float beta, float* attn,
                     float* delta, int S) {
    memcpy(s_out, s_in, (size_t)S * S * sizeof(float));
    const float alpha = expf(g);
    for (int i = 0; i < S * S; ++i) s_out[i] *= alpha;
    for (int j = 0; j < S; ++j)
        delta[j] = (v[j] - dot16(s_out + (size_t)j * S, k, S)) * beta;
    for (int j = 0; j < S; ++j) {
        float* row = s_out + (size_t)j * S;
        const float d = delta[j];
        const float32x4_t dv = vdupq_n_f32(d);
        int c = 0;
        for (; c + 4 <= S; c += 4)
            vst1q_f32(row + c, vfmaq_f32(vld1q_f32(row + c), vld1q_f32(k + c), dv));
        for (; c < S; ++c) row[c] += d * k[c];
    }
    const float scale = 1.0f / sqrtf((float)S);
    for (int j = 0; j < S; ++j)
        attn[j] = dot16(s_out + (size_t)j * S, q, S) * scale;
}

static void gdn_fused(float* s_out, const float* s_in, const float* k, const float* q,
                      const float* v, float g, float beta, float* attn, int S) {
    const float alpha = expf(g);
    const float scale = 1.0f / sqrtf((float)S);
    for (int j = 0; j < S; ++j) {
        const float* rin = s_in + (size_t)j * S;
        float* rout = s_out + (size_t)j * S;
        const float d = (v[j] - alpha * dot16(rin, k, S)) * beta;
        const float32x4_t av = vdupq_n_f32(alpha), dv = vdupq_n_f32(d);
        float32x4_t oq0 = vdupq_n_f32(0.f), oq1 = oq0;
        int c = 0;
        for (; c + 8 <= S; c += 8) {
            float32x4_t s0 = vld1q_f32(rin + c), k0 = vld1q_f32(k + c), q0 = vld1q_f32(q + c);
            float32x4_t s1 = vld1q_f32(rin + c + 4), k1 = vld1q_f32(k + c + 4), q1 = vld1q_f32(q + c + 4);
            float32x4_t n0 = vfmaq_f32(vmulq_f32(av, s0), dv, k0);
            float32x4_t n1 = vfmaq_f32(vmulq_f32(av, s1), dv, k1);
            vst1q_f32(rout + c, n0);
            vst1q_f32(rout + c + 4, n1);
            oq0 = vfmaq_f32(oq0, n0, q0);
            oq1 = vfmaq_f32(oq1, n1, q1);
        }
        float ot = 0.f;
        for (; c < S; ++c) {
            rout[c] = alpha * rin[c] + d * k[c];
            ot += rout[c] * q[c];
        }
        attn[j] = (vaddvq_f32(oq0) + vaddvq_f32(oq1) + ot) * scale;
    }
}

static bool test_case(int S, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> kin(S), qin(S), vin(S), attnA(S), attnB(S), delta(S), s0((size_t)S * S);
    for (auto& x : kin) x = U(rng);
    for (auto& x : qin) x = U(rng);
    for (auto& x : vin) x = U(rng);
    for (auto& x : s0)  x = U(rng);
    const float g = -0.35f, beta = 0.05f;

    std::vector<float> SA(s0), SB(s0);
    gdn_tree(SA.data(), s0.data(), kin.data(), qin.data(), vin.data(), g, beta, attnA.data(), delta.data(), S);
    gdn_fused(SB.data(), s0.data(), kin.data(), qin.data(), vin.data(), g, beta, attnB.data(), S);

    std::vector<double> sd(s0.begin(), s0.end()), ad(S, 0.0);
    const double alpha = exp((double)g), scale = 1.0 / sqrt((double)S);
    for (int j = 0; j < S; ++j) {
        double u = 0.0;
        for (int c = 0; c < S; ++c) u += sd[(size_t)j * S + c] * kin[c];
        const double dd = (vin[j] - alpha * u) * beta;
        for (int c = 0; c < S; ++c) sd[(size_t)j * S + c] = alpha * sd[(size_t)j * S + c] + dd * kin[c];
        for (int c = 0; c < S; ++c) ad[j] += sd[(size_t)j * S + c] * qin[c];
        ad[j] *= scale;
    }

    auto check = [&](const std::vector<float>& got, const std::vector<double>& want, const char* tag) {
        double num = 0.0, den = 0.0, maxabs = 0.0;
        for (size_t i = 0; i < want.size(); ++i) {
            const double d = got[i] - want[i];
            num += d * d;
            den += want[i] * want[i];
            if (std::fabs(d) > maxabs) maxabs = std::fabs(d);
        }
        const double rms_rel = std::sqrt(num / (den + 1e-20));
        if (rms_rel > 1e-5 || maxabs > 2e-4) {
            printf("[test] FAIL %s: rms_rel=%.3g max_abs=%.3g\n", tag, rms_rel, maxabs);
            return false;
        }
        return true;
    };
    if (!check(attnA, ad, "attn_tree")) return false;
    if (!check(attnB, ad, "attn_fused")) return false;
    if (!check(SA, sd, "state_tree")) return false;
    if (!check(SB, sd, "state_fused")) return false;
    printf("[test] S=%d: tree and fused match double truth (state + attn)\n", S);
    return true;
}

static void bench(int S, int reps) {
    const int H = 32;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> k((size_t)H * S), q((size_t)H * S), v((size_t)H * S), gH(H), bH(H);
    for (auto& x : k) x = U(rng);
    for (auto& x : q) x = U(rng);
    for (auto& x : v) x = U(rng);
    for (int h = 0; h < H; ++h) { gH[h] = -0.6f + U(rng) * 0.65f; bH[h] = 0.02f + U(rng) * 0.07f; }
    std::vector<float> st_in((size_t)H * S * S), st_tree(st_in.size()), st_fus(st_in.size());
    for (auto& x : st_in) x = U(rng);
    std::vector<float> attn(S), delta(S);

    auto run = [&](int which, double& sec, float& chk) {
        for (int w = 0; w < 3; ++w)
            for (int h = 0; h < H; ++h) {
                const size_t o = (size_t)h * S * S;
                if (which) gdn_fused(st_fus.data() + o, st_in.data() + o, k.data() + (size_t)h * S,
                                     q.data() + (size_t)h * S, v.data() + (size_t)h * S, gH[h], bH[h],
                                     attn.data(), S);
                else gdn_tree(st_tree.data() + o, st_in.data() + o, k.data() + (size_t)h * S,
                              q.data() + (size_t)h * S, v.data() + (size_t)h * S, gH[h], bH[h],
                              attn.data(), delta.data(), S);
            }
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int h = 0; h < H; ++h) {
                const size_t o = (size_t)h * S * S;
                if (which) gdn_fused(st_fus.data() + o, st_in.data() + o, k.data() + (size_t)h * S,
                                     q.data() + (size_t)h * S, v.data() + (size_t)h * S, gH[h], bH[h],
                                     attn.data(), S);
                else gdn_tree(st_tree.data() + o, st_in.data() + o, k.data() + (size_t)h * S,
                              q.data() + (size_t)h * S, v.data() + (size_t)h * S, gH[h], bH[h],
                              attn.data(), delta.data(), S);
                for (int j = 0; j < S; ++j) chk += attn[j]; // loop-carried, defeats DCE
            }
        auto t1 = std::chrono::steady_clock::now();
        sec = std::chrono::duration<double>(t1 - t0).count() / reps;
    };

    double sA, sB;
    float ca = 0.f, cb = 0.f;
    run(0, sA, ca); run(1, sB, cb);
    run(0, sA, ca); run(1, sB, cb); // second sweep

    const double flops = 4.0 * S * S * H;                       // scale + dot + mad + dot, counted as 2 flops each
    const double tree_bytes = 8.0 * S * S * 4;                  // 8 traffic units per head-token
    const double fus_bytes = 2.0 * S * S * 4;                   // DRAM-ideal: read once, write once
    printf("[bench S=%d H=%d] tree : %8.3f us/layer-step  %7.2f GFLOP/s  ~%7.1f GB/s state\n",
           S, H, sA * 1e6, flops / sA / 1e9, tree_bytes * H / sA / 1e9);
    printf("[bench S=%d H=%d] fused: %8.3f us/layer-step  %7.2f GFLOP/s  ~%7.1f GB/s state\n",
           S, H, sB * 1e6, flops / sB / 1e9, fus_bytes * H / sB / 1e9);
    printf("[bench] chk tree=%.3f fused=%.3f\n", ca, cb);
}

int main() {
    if (!test_case(128, 1)) return 1;
    if (!test_case(67, 2)) return 1; // odd size exercises tail paths
    bench(128, 300);
    return 0;
}
