# Qwen3.5-4B Post-Training: Failure Analysis and Recovery Plan

**Date:** 2026-09-30  
**GPU:** NVIDIA RTX 4070 SUPER 12 GB  
**Base model:** `Qwen/Qwen3.5-4B`

## Executive conclusion

Neither completed adapter has demonstrated a reliable overall accuracy improvement over the base model. The first adapter produced a small positive average in a seven-task, 10-example-per-task screening run, driven mainly by GSM8K and a few multiple-choice gains, but it also regressed on TruthfulQA, HellaSwag and BBH. The later science-v2 adapter did not preserve those gains and showed several regressions.

These evaluations are screening tests, not publication-quality estimates: ten examples per task have very high sampling uncertainty. They are sufficient to reject the claim that the adapters produced a clear, broad improvement, but not sufficient to estimate a precise final capability score.

## Measured screening results

| Benchmark / metric | Base | Previous adapter | Science-v2 adapter | Science-v2 vs base |
|---|---:|---:|---:|---:|
| MMLU-Pro Computer Science exact match | 0% | 10% | 0% | 0 pp |
| TruthfulQA MC2 | 57.38% | 49.24% | 49.31% | -8.07 pp |
| GSM8K flexible exact match | 0% | 40% | 0% | 0 pp |
| HellaSwag normalized accuracy | 70% | 40% | 60% | -10 pp |
| BBH logical deduction flexible match | 30% | 20% | 10% | -20 pp |
| IFEval instruction-level strict | 22.22% | 22.22% | 22.22% | 0 pp |
| ARC-Challenge normalized accuracy | 30% | 40% | 30% | 0 pp |

The science-v2 scores were produced with the same NF4 inference configuration, prompts, harness version, deterministic decoding and sampled items as the earlier base/adapter run.

## Training runs

### Previous fact-checking adapter

- Public adapter: `RohitSwami33/qwen35-fc-adapter`
- Released Kaggle cache: 25,037 sequences, 6.58M total tokens, 2.89M supervised tokens.
- Training mixture reported by its metadata:
  - HotpotQA: 9,000
  - GSM8K: 5,500
  - OpenR1 Math: 2,500
  - SmolTalk: 2,000
  - Synthetic search decisions: 4,500
  - Synthetic hallucination examples: 1,840
  - Synthetic self-corrections: 500
- FEVER, AVeriTeC and ToolACE contributed zero rows in the released build.
- Manual decoding of random Arrow rows confirmed that assistant-only label masks are usable in the released cache.

### Science-v2 adapter

- Started from the original base model with a fresh NF4 QLoRA adapter.
- 16,430 usable examples from SciQ train, PubMedQA labelled data and OpenBookQA train.
- 2,054 optimizer steps, one epoch, maximum length 768.
- Final reported training loss: 0.03883.
- The low loss did not translate into benchmark improvement and is consistent with learning the narrow dataset/answer style too strongly.

## Why performance did not improve broadly

1. **Objective mismatch.** Supervised next-token loss rewards imitating reference answers; it does not directly reward calibrated truthfulness, refusal when evidence is absent, or correctness under changed prompts.
2. **Narrow data distribution.** Science-v2 was dominated by short supported QA and multiple-choice answers. It lacked enough adversarial false-premise, conflicting-evidence, citation-quality and open-ended helpfulness examples.
3. **Capability interference.** Training on a narrow domain without sufficient general reasoning, mathematics, instruction-following and conversation replay can overwrite useful behavior.
4. **Synthetic evidence weakness in the older set.** A significant fraction teaches a search/tool-call pattern with synthetic entities and placeholder-style sources rather than real source verification.
5. **Evaluation uncertainty.** Ten examples per benchmark can move by 10 percentage points from one answer, so small apparent changes must not be treated as robust.
6. **Training loss is not model quality.** A very small training loss can indicate memorization or an easy formatting objective rather than better reasoning.

## Recovery experiment: v3 replay continuation

The next experiment starts from the previous adapter, not science-v2. It uses the older released cache as a replay anchor and adds a smaller sample of grounded science data. It keeps the base quantized to NF4 4-bit, trains only LoRA parameters, and uses a much lower learning rate (`5e-6`).

Planned mixture:

- 8,000 randomly selected old-cache replay sequences.
- 4,000 grounded science examples.
- One epoch, effective batch size 8, maximum sequence length 768.

This is intended to preserve the previous adapter's mathematics and QA behavior while testing whether limited grounded-science exposure can improve domain performance without the regressions seen in science-v2.

## Required acceptance gate

Do not publish a claim of improvement unless v3:

- improves the targeted biology/chemistry/health aggregate;
- does not reduce TruthfulQA materially;
- retains GSM8K, HellaSwag, BBH, IFEval and ARC within an agreed tolerance;
- is evaluated on more than the tiny 10-item screening set before final reporting.

If v3 fails, the next step should be evidence-grounded preference optimization (DPO/ORPO) with chosen/rejected answers, not another large SFT epoch.

## Repository contents

- `training/build_science_sft.py` builds the grounded SciQ, PubMedQA and OpenBookQA SFT corpus.
- `training/train_4bit_qlora.py` runs fresh or adapter-continuation NF4 QLoRA and supports replay from the released Arrow cache.
- `benchmarks/run_real_benchmarks.py` runs deterministic `lm-eval` comparisons.
- `benchmarks/*.json` contains the raw base, old-adapter and science-v2 screening results, including logged samples.
- `dataset_audit/` records the released Kaggle cache composition and build manifest.
- `model_v3/` will contain the final replay-continuation adapter and training metrics after the active run completes.

## Reproduce the v3 run

The base model is not duplicated in this repository. Download `Qwen/Qwen3.5-4B`, the previous adapter and the Kaggle tokenized cache separately, then run:

```powershell
python training/train_4bit_qlora.py `
  --data path/to/science_sft.jsonl `
  --limit 4000 `
  --replay-cache path/to/tokenized_cache `
  --replay-limit 8000 `
  --init-adapter path/to/previous/fc_adapter `
  --learning-rate 5e-6 `
  --max-length 768 `
  --output outputs/qwen35-fc-science-replay-v3
```

Core packages used were PyTorch with CUDA, Transformers, PEFT, bitsandbytes, Datasets, Accelerate and lm-evaluation-harness.
