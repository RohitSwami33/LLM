// b6_threads.cpp - Block 10: big-core selection logic (host unit tests)
// destined verbatim for ai_chat.cpp: probe cpuinfo_max_freq, select cores above
// the slowest core, cap the count, fall back safely when sysfs is unreadable.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

static std::vector<int> cpu_max_freqs(int ncpu) {
    std::vector<int> freqs(ncpu, 0);
    for (int i = 0; i < ncpu; i++) {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE* f = fopen(path, "r");
        int khz = 0;
        if (f) {
            if (fscanf(f, "%d", &khz) == 1) { freqs[i] = khz; }
            fclose(f);
        }
    }
    return freqs;
}

static int select_big_cores(const std::vector<int>& freqs, std::vector<bool>& mask, int cap) {
    const int ncpu = (int) freqs.size();
    int min_f = -1, max_f = -1;
    for (int f : freqs) {
        if (f > 0) {
            if (min_f < 0 || f < min_f) { min_f = f; }
            if (max_f < 0 || f > max_f) { max_f = f; }
        }
    }
    for (int i = 0; i < ncpu; i++) { mask[i] = false; }
    int n = 0;
    if (min_f > 0 && max_f > min_f) {
        // select every core faster than the slowest detected core
        for (int i = 0; i < ncpu; i++) {
            if (freqs[i] > min_f) { mask[i] = true; n++; }
        }
        // over cap: drop the slowest selected cores
        while (n > cap) {
            int slow = -1;
            for (int i = 0; i < ncpu; i++) {
                if (mask[i] && (slow < 0 || freqs[i] < freqs[slow])) { slow = i; }
            }
            if (slow < 0) { break; }
            mask[slow] = false;
            n--;
        }
    } else {
        // unreadable sysfs or uniform SoC: safe cap
        n = std::min(cap, ncpu);
        for (int i = 0; i < n; i++) { mask[i] = true; }
    }
    return std::max(1, n);
}

static std::string mask_str(const std::vector<bool>& mask) {
    std::string s;
    for (int i = 0; i < (int) mask.size(); i++) { s += mask[i] ? "1" : "0"; }
    return s;
}

struct result { int n_actual; std::string mask; };

static result run_case(const std::vector<int>& freqs, int cap) {
    std::vector<bool> mask(freqs.size(), false);
    const int n = select_big_cores(freqs, mask, cap);
    return { n, mask_str(mask) };
}

int main() {
    struct tc { const char* name; std::vector<int> f; int cap; int en; std::string emask; };
    const tc cases[] = {
        // Snapdragon 8 Gen 3-like: 1 X4 + 3 A720@3.2 + 2 A720@2.8 + 2 A520 (=1+3+2+2)
        { "sd8g3",  { 1900,1900,3300,3200,3200,3200,2800,2800 }, 6, 6, "00111111" },
        // 1+5+2 SoC: 10 cores, 8 fast, cap 6 keeps the 6 fastest
        { "sd8g2",  { 1900,1900,3400,3200,3200,3200,3200,3200,2800,2800 }, 6, 6, "0011111100" },
        // Dimensity 9300-like: 4 X4 only stay above the slower A720 cluster
        { "d9300",  { 3300,3300,3300,3300,2400,2400,2400,2400 }, 6, 4, "11110000" },
        // uniform all-big SoC: fallback to cap
        { "uniform",{ 2500,2500,2500,2500,2500,2500,2500,2500 }, 6, 6, "11111100" },
        // sysfs unreadable: same fallback
        { "zeros",  { 0,0,0,0,0,0,0,0 }, 6, 6, "11111100" },
        // 2 clusters, non-adjacent indices: cores 0,1,2 (2.8 above 1.8 min)
        { "4c",     { 2800,2800,2600,1800 }, 6, 3, "1110" },
        // tiny SoC: single core above min -> clamped to >=1
        { "2c",     { 2600,1800 }, 6, 1, "10" },
    };
    int fails = 0;
    for (const auto& c : cases) {
        const result r = run_case(c.f, c.cap);
        const bool ok = (r.n_actual == c.en) && (r.mask == c.emask);
        printf("[%s] n=%d mask=%s expect n=%d mask=%s %s\n",
               c.name, r.n_actual, r.mask.c_str(), c.en, c.emask.c_str(), ok ? "OK" : "FAIL");
        if (!ok) { fails++; }
    }
    // real host fallback smoke test (macOS: no sysfs -> fallback path)
    const result real = run_case(cpu_max_freqs(8), 6);
    printf("[host-real] n=%d mask=%s\n", real.n_actual, real.mask.c_str());
    const size_t ncases = sizeof(cases) / sizeof(cases[0]);
    if (fails == 0) { printf("[test] b6 selection logic OK (%zu cases)\n", ncases); }
    return fails ? 1 : 0;
}
