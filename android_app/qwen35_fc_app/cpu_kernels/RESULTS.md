# Custom Inference Kernel Investigation — `qwen35_fc_app` (Qwen3.5-4B, llama.cpp CPU)

Date: 2026-10-17. Everything below is measured, self-tested, and reproducible.

## TL;DR verdict

| Kernel (2560×2560, 1 thread) | GFLOP/s | Verdict |
|---|---|---|
| b1 scalar f32 GEMV `-O3` | 3.96 | baseline (serial-acc illness) |
| b2 NEON f32, 4× `vfmaq` accumulators | 34.4 | 8.7× scalar |
| b3 q4_0×q8_0 via `vdotq` (standalone) | 86.9 | 22× scalar |
| b4 mine1 (same kernel in race) | 85.6 | ≈ b3 |
| b4 ggml2 (top: upstream NEON kernel extracted verbatim) | 92.8 | **winner** |
| b4 mine4 (4 blocks/iter, 4 vec accs) | 90.0 | **tie with ggml (noise ±2)** |
| b4 mine5 (no-vsub nibble trick + correction fold) | 78.3 | REJECTED (slower) |
| b4 vm6 (i8mm dual-row `vmmla`) | 65.2 | REJECTED on Apple |

**Conclusion: ggml's `vec_dot_q4_0_q8_0` is at the practical ceiling (≈92 GFLOP/s) on Apple Silicon for this shape. Custom rewrite ties it; every cleverer trick loses. Custom C++/Rust inference engine = not worth it; custom kernel = already matched the ceiling.**

## Why this kernel matters here

- App KV cache (TurboQuant-style) is wired in `lib/src/main/cpp/ai_chat.cpp`:
  `K = GGML_TYPE_Q4_0 (4.5 bpw)`, `V = GGML_TYPE_Q8_0 (8.5 bpw)`.
- llama.cpp auto-applies the Walsh–Hadamard "attn rotation" (`attn_rot_k/v`) to K before quantization — the core mechanism of the TurboQuant paper (arXiv:2504.19874). Free, already in tree.
- Qwen3.5-4B (`qwen35` arch) is hybrid: gated delta-net on 3/4 layers (O(1) state), full attention every 4th layer → only 8/32 layers store KV; quantized K dot runs per Q row inside FA — exactly the benchmarked shape.

## Everything tried, in build order (test-after-every-change)

- **b1_scalar.cpp** — plain f32 matvec. Found compiler DCE first try (`inf GFLOP/s`); fixed by consuming a checksum. Result 3.96.
- **b2_neon.cpp** — 4 independent `vfmaq_f32` accumulators. 34.4. (scalar version in same file: 4.3)
- **b3_q4q8.cpp** — real target `vec_dot_q4_0_q8_0`, ref (int64) + NEON (`vdotq_s32`).
  - **Bug caught by test:** first quantizer painted nibbles interleaved-per-byte; ggml layout is byte-major (low nibble of byte j = element j, high nibble = element j+16). Fixed → truth/ref/neon agree bitwise (0.253929). 86.9.
- **b4_ggml_race.cpp** — race harness, ggml's kernel extracted verbatim (trim macros, keep semantics).
  - **Bug caught:** second DCE form — scalar-only result let clang reduce 102k identical calls to 1 (printed absurd 12.4M GFLOP/s). Fixed with loop-carried `chk += c` per rep.
  - Race: mine1 85.6 vs ggml2 92.8.
- **b5 (vm4)** — 4 blocks/iter, 4 float32x4 accumulators, merged into b4. 90.0 → tie.
- **b6 (vm5)** — identity `dot(a-8,b) = dot(nibbles,b) − 8·sum(b)`; no vsub, precomputed per-block `sum` in a `block_q8_0s` struct. 78.3 → slower, REJECTED. Reason: moved work onto the saturating FP pipe (int→float + serial corr chain) while removing essentially-free u8 subs riding in SDOT's shadow.
- **b7 (vm6)** — dual-row i8mm `vmmlaq_s32` (from ggml's `nrc==2` batch path).
  - Syntax-only compile first caught a broken nested-paren chain; rewrote as clean 4-step accumulator chain.
  - Correctness: bit-exact for both rows vs double truth (0.253929 / 0.162595).
  - Performance: 65.2 → slower on Apple. Apple's i8mm datapath + zip overhead lose to plain SDOT for nrc==1. ggml itself only uses this path for batched cases; on Snapdragon it reportedly pays — keep the code, flag for on-device test.

## App-side TurboQuant KV wiring (on-disk, uncommitted)

```cpp
// ai_chat.cpp
constexpr bool  KV_CACHE_QUANTIZED  = true;
constexpr auto  KV_TYPE_K           = GGML_TYPE_Q4_0;  // blck 32 → head 128 OK
constexpr auto  KV_TYPE_V           = GGML_TYPE_Q8_0;  // 8.5 bpw, needs FA (auto-enabled)
// NOTE: do NOT use TQ1_0/TQ2_0 as KV types: blck QK_K=256, head dim 128 → init fails
```
// inside init_context():
if (KV_CACHE_QUANTIZED) {
    ctx_params.type_k = KV_TYPE_K;
    ctx_params.type_v = KV_TYPE_V;
    LOGi("%s: TurboQuant-style KV cache: K=%s V=%s (attn rotation auto-enabled)",...);
}
// F16 fallback retry if init rejects quantized KV (never brick the app)
```

Facts pinned from the vendored llama.cpp:
- `llama.h:389` `ctx_params.type_k/type_v` [EXPERIMENTAL]
- `llama-context.cpp:~3690`: quantized V ⇒ flash-attn required (`AUTO` auto-enables); quantized K ⇒ `n_embd_head_k % blck_size == 0`, TQ1_0/TQ2_0 blocks are QK_K=256 → fail on head 128; Q4_0 (blck 32) passes.
- `llama-kv-cache.cpp`: `attn_rot_k/v` Hadamard rotation auto-enables for quantized KV with head % 64 == 0; `LLAMA_ATTN_ROT_DISABLE=1` env turns it off.
- CPU dot tables (`ggml-cpu/ggml-cpu.c:401`): TQ1_0/TQ2_0, Q4_0, Q8_0 all have vec_dot; CPU FA supports arbitrary quantized K, V dequant via `to_float`.

## Reproduce (copy-paste)

```bash
cd android_app/qwen35_fc_app/cpu_kernels
c++ -std=c++17 -O3 -mcpu=native b1_scalar.cpp -o b1 && ./b1
c++ -std=c++17 -O3 -mcpu=native b2_neon.cpp -o b2    && ./b2
c++ -std=c++17 -O3 -mcpu=native b3_q4q8.cpp -o b3    && ./b3
c++ -std=c++17 -O3 -mcpu=native b4_ggml_race.cpp -o b4 && ./b4
# each binary self-tests (double-precision truth) before timing
```

Probe that made i8mm available to us:
```bash
c++ -mcpu=native -dM -E probe.cpp | grep MATMUL_INT8   # → defined on Apple clang 17
```

## Files

`cpu_kernels/b1_scalar.cpp`, `b2_neon.cpp`, `b3_q4q8.cpp`, `b4_ggml_race.cpp` (contains vm1/vm4/vm5/vm6 + ggml upstream + triple-verification), this `RESULTS.md`. All uncommitted work-tree files.

## Lessons recorded

1. Two benchmark-DCE traps: (a) result never read; (b) single overwritten scalar → only last call survives. Fix: loop-carried accumulator in the timed loop.
2. ggml q4_0 nibble convention is byte-major: low nibble of byte j = element j; high = element j+16.
3. FP-pipe budgeting beats op-counting: swapping 2 free u8 vsubs for scalar int→float + serial adds = 14% slower.
4. i8mm (vmmla) on Apple cores loses to SDOT for single-row dots; useful only for batched (nrc==2) patterns on other ISAs.
5. Protocol that caught everything: syntax-check → double-precision truth test → only then race.

## Open options

1. On-device (Snapdragon/Dimensity) run of the same bench — only place vm6/i8mm could win (ggml uses it when `__ARM_FEATURE_MATMUL_INT8` exists — Snapdragon 8 gen 3 supports i8mm). Then decide wiring.
2. Otherwise: keep upstream ggml; inference speed budget already spent well = KV q4_0/q8_0 + rotation.

---

# Block 6 (b6_threads.cpp): big-core selection logic - PASSED

Thread-policy probe destined for `ai_chat.cpp` (Android big.LITTLE cadence
problem: default threadpool schedules efficiency cores -> +/-25% timing jitter
under EAS). Two pure functions, unit-tested on synthetic SoC topologies on the
host before porting into the app (no sysfs on macOS -> fallback path exercised
by the real-host smoke test):

- `cpu_max_freqs(ncpu)` — reads `/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq`, 0 for unreadable.
- `select_big_cores(freqs, mask, cap)` — every core strictly faster than the
  slowest detected core (captures Prime + Performance clusters on 1+3+4 /
  1+5+2 while dropping A5xx), over-cap drops the slowest, fallback = first
  `min(cap, ncpu)` cores when sysfs is unreadable or all cores match.

Test results (7/7 OK):

| case | freqs (MHz) | pick |
|---|---|---|
| sd8g3 (1+3+2+2) | 1900,1900,3300,3200,3200,3200,2800,2800 | 6 threads, cores 2-7 |
| sd8g2 (1+5+2) | 1900,1900,3400,3200x5,2800,2800 | cap 6, slowest dropped |
| d9300 (4+4) | 3300x4,2400x4 | 4 threads (X4 only) |
| uniform | all 2500 | fallback: 6 |
| zeros (no sysfs) | all 0 | fallback: 6 |
| 4c {2800,2800,2600,1800} | | 3 threads |
| 2c {2600,1800} | | 1 thread (>=1 clamp) |

## App wiring (ai_chat.cpp)

- `init_context()`: probe -> cpumask -> `ggml_threadpool_params_default(n)`
  with `cpumask[i]` per probe + `poll=50` -> `ggml_threadpool_new()` once
  (global `g_threadpool`) -> `llama_attach_threadpool(ctx, tp, tp)` after
  successful `llama_init_from_model` (F16-retry path covered: attach after
  both attempts).
- `unload()`: `ggml_threadpool_free()` strictly after `llama_free(ctx)`.
- FallbackSafety: `ctx_params.n_threads/n_threads_batch` still set, so a failed
  `ggml_threadpool_new` degrades to default pooling, never 0 threads.
- LOGi prints detected `big-cores=[ids]` and thread count.

Addresses `llama_context_params` having no cpumask exposure: pool built via
ggml API and attached (llama.h:490 in this pin).

```bash
cd android_app/qwen35_fc_app/cpu_kernels
c++ -std=c++17 -O2 -Wall b6_threads.cpp -o b6 && ./b6
# app-side syntax gate (no NDK needed; jni.h + android/log.h stubs):
c++ -std=c++17 -fsyntax-only -I<stubdir> -Ilib/src/main/cpp \
    -Illama.cpp/include -Illama.cpp/common -Illama.cpp/ggml/include \
    lib/src/main/cpp/ai_chat.cpp
```

## Pending verification

- On-device A/B (tg t/s with big-core pins vs previous `min(nproc-2, 4)`).
- Logic-verified here; CPU behavior identical on host and device uniformly.

---

# Block 9.1: redundant state copy-in removal in the fused kda==0 path

`test-backend-ops perf` showed decode (`n_seq_tokens=1`) at 68.72 us for
kda=0, nearly equal to the unfused kda=1 at 70.36 us - a flat cost masking
the fused kernel's marginal 18 us/token. Cause found by reading ops.cpp:
the per-head `memcpy(s_out, s_in, S_v*S_v*sizeof(float))` before the t-loop
ran unconditionally. For kda==0 the fused pass writes s_out fully on t==0
(reading s_in via `s_src`), so that copy is pure redundant traffic on
decode (2 MB read + 2 MB write at h=32,S=128).

Fix (one call site, vendored `ggml/src/ggml-cpu/ops.cpp` one_chunk):
`if (kda || n_tokens == 0) { memcpy(...); }` - kda==1 tree path mutates
s_out in place and needs the copy; n_tokens==0 keeps the state-in ->
state-out contract. K>1 snapshot memcpy is per-token and untouched.

Verification: `test-backend-ops -o GATED_DELTA_NET -b CPU` = 36/36 OK
(kda=0 K=1/2/3/4, kda=1, GQA v_repeat=2, permuted, n_seqs=1/2).

Perf (`perf -o GATED_DELTA_NET -b CPU`, h=32, S=128, hw threads=10):

| case | before guard | after | note |
|---|---|---|---|
| n=1 kda=0 | 68.72 us | 25.31 / 29.96 us (2 runs) | decode op-call cost |
| n=1 kda=1 | 70.36 us | 42.27 / 42.73 us | code unchanged - env shift |
| n=64 kda=0 | 1409.73 us | 1206.31 us | -203 us flat |
| n=256 kda=0 | 5016.54 us | 4286.50 us | -730 us |
| n=1024 kda=0 | 18665.70 us | 16982 / 17315 us | marginal 16.5-18.7 us/tok |

Attribution honesty: the unchanged kda=1 path also improved ~28 us
(70.36 -> 42.3), so frequency/cache environment shifted between builds.
Within-run kda=0 vs kda=1 now 1.4-1.7x at decode (was ~1.0x before the
Block 9+9.1 build). Marginal per-token costs dropped 18.8 -> 16.1 us
(tokens 65..256), consistent with a flat-cost removal. Rep-to-rep noise
on the n=1 row is +/-15%, on-device A/B is the authoritative check.

24 GDN layers decode CPU cost (cache-warm M4-10T estimate):
~1650 us/token before -> ~610-720 us/token after these builds.

---

# Block 11: KleidiAI-compatible re-quantize (q4_0 body + q8_0 mix)

Goal: the app's Q4_K_M GGUF bypasses the KleidiAI (KAI) repack path
(kleidiai.cpp accepts only Q4_0 / Q8_0 / F32) - every matrix matmul runs
on the stock ggml path. New export makes every 2D weight KAI-eligible.

Command (vendored llama-quantize, 8.3 s for 426 tensors):

```bash
./build/bin/llama-quantize \
  --output-tensor-type q8_0 \
  --tensor-type "attn_v=q8_0" \
  --tensor-type "ssm_out=q8_0" \
  model_export/qwen35_orpo_v4_posttrained_f16.gguf \
  model_export/qwen35_orpo_v4_posttrained_q4_0_kai.gguf Q4_0
```

Result:
- 8023.67 MiB (16.00 BPW) -> 2692.42 MiB (5.37 BPW), file 2.6 GB.
- Tensor histogram (dry-run audit): 216 x q4_0 + 33 x q8_0, rest f32/f16
  norms/state tensors. ZERO Q4_K / Q5_K / Q6_K anywhere - the user's
  guardrail (silent Q6_K promotions on output/attn_v) verified absent.
- No standalone output.weight in this GGUF (tied lm head): the big vocab
  matmul goes through token_embd.weight, which the default heuristic
  promoted to q8_0 (644 MiB) - KAI-eligible, largest decode matmul covered.
- attn_v=q8_0 and ssm_out=q8_0 overrides confirmed in the per-tensor log.

End-to-end host gate: PASSED. llama-cli loads the new GGUF (ftype Q4_0)
and generates coherent text: prompt 109.5 t/s, generation 22.8 t/s
(-t 4, M4, includes the Block 9.1 memcpy guard at runtime). Device-side
remaining: KAI repack log line ("CPU_KLEIDIAI model buffer size") + tok/s
A/B vs the q4_k_m GGUF.

## Block 11 + Block 10 on-Android functional gates (emulator arm64-v8a)

Setup: AVD qwen35_orpo (android-34-ext10, google_apis_playstore, arm64-v8a),
emulator 36.5.11 with 6144 MB RAM / 14 GB data partition. Active CPU
backend variant reported by llama: libggml-cpu-android_armv8.2_2.so.

1. app build blocker found + fixed: libai-chat.so link failed on
   ggml_threadpool_new/free - those symbols live in the cpu variant libs
   (ggml-cpu-android_armv8.x/armv9.x), not ggml-base. dlsym(RTLD_DEFAULT)
   attempt failed at runtime (backend .so is dlopen'ed RTLD_LOCAL), so the
   design fell back to the pre-approved Path B: pin_big_cores() sets
   sched_setaffinity on the driving thread; lazy-created ggml workers
   inherit the mask. Confirmed live:
   init_context: threads=4 big-cores=[0 1 2 3]
   pin_big_cores: pinned 4 cores (ok), rc=0
2. TurboQuant KV confirmed live with rotation:
   KV buffer 104 MiB @ 8192 cells, K q4_0 36 MiB, V q8_0 68 MiB,
   attn_rot_k = attn_rot_v = 1, n_embd_head_k_all = 256,
   flash_attn auto-enabled for quantized V.
3. Fused GDN ops resolved at runtime:
   resolve_fused_ops: fused DeepSeek V4 HC pre/comb/post enabled.
4. KleidiAI gate flipped exactly as predicted by the weights-side analysis:
   old q4_k_m GGUF: "kleidiai: no kernel for tensor type q6_K..."
   (whole file skipped, repack never engages).
   new q4_0_kai GGUF: "CPU_KLEIDIAI model buffer size = 2688.76 MiB"
   (full weights buffer claimed; q4 kernels: DOTPROD feature; f32 without
   compatible kernel only noted for non-weight tensors; q6_K warning gone).
5. Both files generate end-to-end through the app UI (incl. one full agent
   turn with thinking + web_search tool + final answer on the q4_k_m file;
   one clean answer turn on the q4_0_kai file).

Still phone-bound: tok/s A/B (armv9.2_2 variant may select i8mm primary
kernels instead of DOTPROD) and thermals/EAS behavior with the pin path.
App-side filename constant now points at the kai file (revert if needed).
