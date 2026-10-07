# Base vs final ORPO-v4: quick remaining benchmark comparison

This is a paired screening run, not the full GSM8K/IFEval benchmark. Fifty real examples per task were randomly selected with seed 42 before evaluation; both models receive exactly the same examples.

Original model: Qwen/Qwen3.5-4B (unmodified released checkpoint, not the separate pretrained Base model). Trained model: local final ORPO-v4 adapter. Both use NF4 4-bit, FP16 compute, batch 1, identical chat templates, greedy decoding and a 256-token output cap. GSM8K uses the harness's five-shot setup; IFEval is zero-shot.

The short token budget can truncate reasoning and instruction-following responses, especially long IFEval requirements. These scores are budget-constrained and not directly comparable to published full benchmark scores. Fifty examples have substantial sampling uncertainty; differences do not establish a general model improvement or hallucination reduction.

| Task / metric | Original Qwen | Trained ORPO-v4 | Change (percentage points) |
|---|---:|---:|---:|
| GSM8K flexible answer accuracy | 6.00% | 66.00% | +60.00 |
| GSM8K strict answer format accuracy | 0.00% | 60.00% | +60.00 |
| IFEval prompt-level strict | 14.00% | 14.00% | +0.00 |
| IFEval instruction-level strict | 30.67% | 30.67% | +0.00 |
| IFEval prompt-level loose | 14.00% | 18.00% | +4.00 |
| IFEval instruction-level loose | 30.67% | 33.33% | +2.67 |

## Validation and artifacts

Each finished result must contain exactly 50 examples and match the selected document indices. Paired document hashes are checked before calculating differences. Raw outputs, scoring details, references and the question set are stored alongside this report. Separate model/task response caches retain completed responses across interruptions.

Four already completed full-dataset trained-model results remain preserved in ../full-benchmarks/v4-orpo. They are not mixed into this paired sampled comparison. HellaSwag is excluded; no further full-dataset evaluation is scheduled.
