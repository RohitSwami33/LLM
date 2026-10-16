// Block 1: baseline — scalar f32 GEMV at -O3. Qwen3.5-4B hidden size (2560).
// Self-test: compare against double-precision reference on a small case.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

static void matvec_scalar(const float* W, const float* x, float* y, int n, int d) {
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int j = 0; j < d; ++j) s += W[(size_t)i * d + j] * x[j];
        y[i] = s;
    }
}

static int test_correctness() {
    const int n = 8, d = 16;
    std::vector<float> W((size_t)n * d), x(d), y(n);
    for (size_t i = 0; i < W.size(); ++i) W[i] = (float)((int)(i % 13) - 6) * 0.25f;
    for (int j = 0; j < d; ++j) x[j] = (float)(j % 5) - 2.0f;
    matvec_scalar(W.data(), x.data(), y.data(), n, d);
    for (int i = 0; i < n; ++i) {
        double ref = 0.0;
        for (int j = 0; j < d; ++j) ref += (double)W[(size_t)i * d + j] * x[j];
        if (fabs(y[i] - (float)ref) > 1e-3 * fabs(ref) + 1e-5) {
            printf("FAIL row %d: got %g want %g\n", i, y[i], ref);
            return 1;
        }
    }
    printf("[test] correctness OK\n");
    return 0;
}

int main() {
    if (test_correctness()) return 1;

    const int n = 2560, d = 2560; // Qwen3.5-4B hidden x hidden
    std::vector<float> W((size_t)n * d, 1.0f / d), x(d, 1.0f), y(n);

    double chk = 0.0;
    for (int i = 0; i < 3; ++i) matvec_scalar(W.data(), x.data(), y.data(), n, d); // warmup
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 30;
    for (int r = 0; r < reps; ++r) matvec_scalar(W.data(), x.data(), y.data(), n, d);
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i += 512) chk += y[i]; // consume result, defeats DCE

    double sec = std::chrono::duration<double>(t1 - t0).count() / reps;
    double gflops = 2.0 * n * d / sec / 1e9;
    printf("[bench] %dx%d matvec: %.3f ms/call  %.2f GFLOP/s (1 thread, chk=%.3f)\n",
           n, d, sec * 1e3, gflops, chk);
    return 0;
}
