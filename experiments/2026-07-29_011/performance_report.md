# Performance Report

Generated: 2026-07-29 03:06:58

## System

| Property | Value |
|----------|-------|
| Platform | macOS-15.6.1-arm64-arm-64bit-Mach-O |
| Architecture | arm64 |
| CPU cores | 10 |
| RAM | 16.0 GB |
| MPS available | True |
| PyTorch | 2.13.0 |

## Model

| Property | Value |
|----------|-------|
| Parameters | 68.7M |
| Architecture | Transformer (Dense) |
| Sequence length | 2048 |

## Training Configuration

| Property | Value |
|----------|-------|
| Optimizer | muon |
| Learning rate | 0.0003 |
| Batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 65,536 tokens |
| Mixed precision | bf16 |
| torch.compile | False |

## Performance

| Metric | Value |
|--------|-------|
| Total training time | 555.1s |
| Total tokens processed | 0.8M |
| **Throughput** | **1489 tokens/sec** |
| Samples/sec | 0.90 |

### Timing Breakdown

| Stage | Avg Time | % of Total |
|-------|----------|------------|
| Data loading | 1.5ms | 0.1% |
| Forward pass | 11.1ms | - |
| Backward pass | 991.8ms | - |
| Optimizer step | 0.0ms | - |
| **Compute total** | **1002.9ms** | **90.3%** |

### Memory

| Metric | Value |
|--------|-------|
| Peak MPS memory | 1474 MB |
| Model size (FP32) | 275 MB |

## Optimization Suggestions

Training is well-optimized. No changes needed.

## Bottleneck Analysis

**Primary bottleneck: Compute** - The GPU is fully utilized.
