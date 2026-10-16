// Block 2: NEON f32 GEMV (vfmaccq_f32, 4 accumulators) vs scalar baseline.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <arm_neon.h>

static void matvec_scalar(const float* W, const float* x, float* y, int n, int d) {
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int j = 0; j < d; ++j) s += W[(size_t)i * d + j] * x[j];
        y[i] = s;
    }
}

static void matvec_neon(const float* W, const float* x, float* y, int n, int d) {
    const int d4 = d & ~3;
    for (int i = 0; i < n; ++i) {
        const float* w = W + (size_t)i * d;
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f),
                    a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
        int j = 0;
        for (; j + 16 <= d4; j += 16) {
            a0 = vfmaq_f32(a0, vld1q_f32(w + j +  0), vld1q_f32(x + j +  0));
            a1 = vfmaq_f32(a1, vld1q_f32(w + j +  4), vld1q_f32(x + j +  4));
            a2 = vfmaq_f32(a2, vld1q_f32(w + j +  8), vld1q_f32(x + j +  8));
            a3 = vfmaq_f32(a3, vld1q_f32(w + j + 12), vld1q_f32(x + j + 12));
        }
        for (; j < d4; j += 4) a0 = vfmaq_f32(a0, vld1q_f32(w + j), vld1q_f32(x + j));
        float s = vaddvq_f32(a0) + vaddvq_f32(a1) + vaddvq_f32(a2) + vaddvq_f32(a3);
        for (; j < d; ++j) s += w[j] * x[j]; // tail (d=2560 has none)
        y[i] = s;
    }
}

static bool test_correctness() {
    const int n = 8, d = 16;
    std::vector<float> W((size_t)n * d), x(d), y1(n), y2(n);
    for (size_t i = 0; i < W.size(); ++i) W[i] = (float)((int)(i % 13) - 6) * 0.25f;
    for (int j = 0; j < d; ++j) x[j] = (float)(j % 5) - 2.0f;
    matvec_scalar(W.data(), x.data(), y1.data(), n, d);
    matvec_neon(W.data(), x.data(), y2.data(), n, d);
    for (int i = 0; i < n; ++i) {
        double ref = 0.0;
        for (int j = 0; j < d; ++j) ref += (double)W[(size_t)i * d + j] * x[j];
        if (fabs(y2[i] - (float)ref) > 1e-4 * fabs(ref) + 1e-5) {
            printf("FAIL NEON row %d: got %g want %g\n", i, y2[i], ref);
            return false;
        }
    }
    printf("[test] NEON correctness OK\n");
    return true;
}

static double bench_clock(void (*fn)(const float*, const float*, float*, int, int),
                          std::vector<float>& W, std::vector<float>& x,
                          std::vector<float>& y, int n, int d,
                          int reps, double* chk) {
    for (int i = 0; i < 3; ++i) fn(W.data(), x.data(), y.data(), n, d);
    double c = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) { fn(W.data(), x.data(), y.data(), n, d); }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i += 512) c += y[i];
    *chk = c;
    return std::chrono::duration<double>(t1 - t0).count() / reps;
}

int main() {
    if (!test_correctness()) return 1;
    const int n = 2560, d = 2560, reps = 30;
    std::vector<float> W((size_t)n * d, 1.0f / d), x(d, 1.0f), y(n);

    double c1, c2;
    double s1 = bench_clock(matvec_scalar, W, x, y, n, d, reps, &c1);
    double s2 = bench_clock(matvec_neon,    W, x, y, n, d, reps, &c2);
    double gf = 2.0 * n * d / 1e9;
    printf("[bench] scalar : %8.3f ms  %7.2f GFLOP/s (chk=%.3f)\n", s1*1e3, gf/s1, c1);
    printf("[bench] neon4  : %8.3f ms  %7.2f GFLOP/s (chk=%.3f)\n", s2*1e3, gf/s2, c2);
    return 0;
}
