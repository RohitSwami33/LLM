# Qwen3.5-4B base vs fine-tuned adapter

## Outcome

On this 70-example sampled evaluation, the adapter's unweighted seven-task average was **28.46%**, versus **26.77%** for the base model: **+1.69 percentage points**. This small sample shows a mixed result rather than a uniform improvement: the adapter improved MMLU-Pro CS, GSM8K, and normalized ARC-Challenge; tied IFEval; and declined on TruthfulQA-MC2, normalized HellaSwag, and the selected BBH task.

## Primary results

| Benchmark task | Primary metric | Base | Adapter | Adapter delta |
|---|---|---:|---:|---:|
| MMLU-Pro — computer science | exact match | 0.00% | 10.00% | **+10.00 pp** |
| TruthfulQA-MC2 | MC2 accuracy | 57.38% | 49.24% | **-8.15 pp** |
| GSM8K | flexible exact match | 0.00% | 40.00% | **+40.00 pp** |
| HellaSwag | normalized accuracy | 70.00% | 40.00% | **-30.00 pp** |
| BBH — logical deduction, five objects | flexible exact match | 30.00% | 20.00% | **-10.00 pp** |
| IFEval | prompt-level strict accuracy | 0.00% | 0.00% | 0.00 pp |
| ARC-Challenge | normalized accuracy | 30.00% | 40.00% | **+10.00 pp** |
| **Unweighted average** | selected metrics above | **26.77%** | **28.46%** | **+1.69 pp** |

IFEval instruction-level strict accuracy was **22.22% for both models**, even though neither model fully satisfied every instruction in any of the ten prompts. This is consistent with the 256-token generation ceiling cutting off long thinking-mode responses.

## Additional metrics

| Benchmark | Metric | Base | Adapter |
|---|---|---:|---:|
| GSM8K | strict exact match | 0.00% | 30.00% |
| HellaSwag | raw accuracy | 30.00% | 30.00% |
| BBH selected task | strict exact match | 0.00% | 0.00% |
| IFEval | instruction-level loose accuracy | 22.22% | 22.22% |
| IFEval | prompt-level loose accuracy | 0.00% | 0.00% |
| ARC-Challenge | raw accuracy | 40.00% | 40.00% |

## Runtime and hardware

| Property | Base | Adapter |
|---|---:|---:|
| Evaluation wall time | 645.785 s (10m 45.8s) | 835.635 s (13m 55.6s) |
| Peak allocated model VRAM | 3.419 GiB | 3.602 GiB |
| Generation requests | 40 | 40 |
| Likelihood requests | 160 | 160 |

- GPU: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB.
- Base: `Qwen/Qwen3.5-4B`.
- Adapter: `RohitSwami33/qwen35-fc-adapter` applied through PEFT.
- Quantization: bitsandbytes NF4 4-bit, double quantization, FP16 compute.
- Runtime: PyTorch 2.14.0 + CUDA 13.0; Transformers 5.18.0.dev0; lm-evaluation-harness 0.4.14.dev0.
- Chat template applied; few-shot examples formatted as multi-turn where the task specifies them.
- Greedy decoding, `max_gen_toks=256`, batch size 1.
- Seeds: Python, NumPy, PyTorch, and few-shot sampler all set to 42.

## Scope and interpretation

This is a **sampled comparison**, not a claim of full benchmark performance. Exactly ten records were evaluated for each of seven tasks (70 records per model). MMLU-Pro and BBH are represented by one selected subtask each rather than every category. With only ten records, one question moves most accuracy figures by ten percentage points, so the results have high sampling variance.

GPQA Diamond was initially selected but could not be downloaded because its Hugging Face dataset is gated and this machine had no authenticated Hugging Face token. It was replaced with the public TruthfulQA-MC2 benchmark.

The strongest observed adapter gain was GSM8K (+40 pp). The largest decline was normalized HellaSwag (-30 pp). The adapter also took about **29.4% longer** end-to-end and used about **0.183 GiB** more peak allocated VRAM. Given the small sample, the correct conclusion is that the adapter changed the model's capability profile and slightly increased the sampled average; a larger stratified run is required to establish whether the net improvement is reliable.

## Saved evidence

- `base-limit10.json`: complete base metrics, task configs, run metadata, and logged per-sample outputs.
- `adapter-limit10.json`: complete adapter metrics, task configs, run metadata, and logged per-sample outputs.
- `smoke-base.json`: one-record-per-task validation run used before the full evaluation.

The JSON files are the source of truth for all numbers in this report.
