# Qwen3.5-4B Local Post-Training Comparison

All variants were evaluated with the same NF4 configuration, deterministic decoding, task prompts and sampled items. Each task contains only 10 examples, so this is a screening comparison rather than a publication-quality estimate.

| Task | Base | Old adapter | Science-v2 | Replay-v3 | ORPO-v4 checkpoint 500 (partial) |
|---|---:|---:|---:|---:|---:|
| mmlu_pro_computer_science | 0.00% | 10.00% | 0.00% | 20.00% | 30.00% |
| truthfulqa_mc2 | 57.38% | 49.24% | 49.31% | 48.50% | 44.64% |
| gsm8k | 0.00% | 40.00% | 0.00% | 40.00% | 70.00% |
| hellaswag | 70.00% | 40.00% | 60.00% | 40.00% | 50.00% |
| bbh_cot_zeroshot_logical_deduction_five_objects | 30.00% | 20.00% | 10.00% | 30.00% | 30.00% |
| ifeval | 22.22% | 22.22% | 22.22% | 27.78% | 22.22% |
| arc_challenge | 30.00% | 40.00% | 30.00% | 40.00% | 40.00% |

## Selected-metric average

| Variant | Average | Change vs base |
|---|---:|---:|
| Base | 29.94% | +0.00 pp |
| Old adapter | 31.64% | +1.69 pp |
| Science-v2 | 24.51% | -5.44 pp |
| Replay-v3 | 35.18% | +5.24 pp |
| ORPO-v4 checkpoint 500 (partial) | 40.98% | +11.04 pp |

## Interpretation rule

A final improvement claim requires gains on the targeted factual/truthfulness tasks without material regression on general reasoning. Because n=10 per task is noisy, any promising adapter should receive a larger held-out evaluation before publication.
