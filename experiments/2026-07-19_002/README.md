# Model Card

## Overview

- **Architecture**: Decoder-only Transformer
- **Parameters**: 68721984
- **Vocabulary Size**: 32000
- **Context Length**: 1024 tokens

## Architecture

| Component | Value |
|-----------|-------|
| d_model | 576 |
| n_heads | 8 |
| n_layers | 6 |
| d_ff | 2304 |
| Activation | swiglu |
| Normalization | rmsnorm |
| RoPE | True |
| Flash Attention | False |
| Gradient Checkpointing | True |

## Optimizer

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | muon |
| Learning Rate | 0.0003 |
| Min LR | 3e-05 |
| Weight Decay | 0.1 |
| Beta1 | 0.9 |
| Beta2 | 0.95 |
| Grad Clip | 1.0 |
| Scheduler | cosine |
| Warmup Steps | 200 |
| Batch Size | 4 |
| Grad Accum Steps | 8 |
| Effective Batch | 32 |
| Mixed Precision | fp32 |
| EMA | True (decay=0.9999) |

## Tokenizer

| Property | Value |
|----------|-------|
| Type | sentencepiece |
| Vocab Size | 32000 |
| Model Path | training/tokenizer/pretrain_tokenizer.model |

## Dataset

| Property | Value |
|----------|-------|
| Path | datasets/mixtures/tiny/corpus.jsonl |
| Max Seq Len | 1024 |
| Train/Val Split | N/A/0.05 |
| Packing | False |

## Training

| Metric | Value |
|--------|-------|
| Total Steps | 2000 |
| Training Time | 1h 0m |
| Tokens Processed | 7611232 |
| Avg Tokens/sec | 2082.7 |
| Avg Iteration Time | 1.5216 |

## Evaluation

| Metric | Value |
|--------|-------|
| best_val_loss | 9.9481 |
| best_perplexity | 20912.0048 |
| final_val_loss | 10.4137 |
| final_perplexity | 33312.7993 |
| final_accuracy | 0.0391 |

## Environment

- **OS**: macOS-15.6.1-arm64-arm-64bit-Mach-O
- **Python**: 3.14.4
- **PyTorch**: 2.13.0
