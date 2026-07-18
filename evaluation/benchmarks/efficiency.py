"""Efficiency benchmarks: throughput, memory, latency, FLOPs, parameter count."""

import time
import torch
import math
from .base import Benchmark, BenchmarkResult


class EfficiencyBenchmark(Benchmark):
    """Measure computational efficiency of a model."""

    def __init__(self, max_seq_len: int = 1024, batch_size: int = 1,
                 warmup_steps: int = 5, measure_steps: int = 20,
                 measure_memory: bool = True):
        self.name = "efficiency"
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.measure_memory = measure_memory

    def evaluate(self, model, tokenizer, device: torch.device) -> BenchmarkResult:
        model.eval()
        vocab_size = getattr(model, "vocab_size",
                             getattr(model.config, "vocab_size", 32000))
        if hasattr(model, "lm_head"):
            vocab_size = model.lm_head.weight.shape[0]

        input_ids = torch.randint(0, vocab_size, (self.batch_size, self.max_seq_len), device=device)

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # FLOPs estimate: ~2 * params * seq_len per token (forward only)
        forward_flops = 2 * total_params * self.max_seq_len * self.batch_size
        # Training: ~6x forward (forward + backward + activation)
        training_flops_per_step = 6 * forward_flops

        # Loading time (simulated: just measure a forward pass setup)
        t_load_start = time.time()
        _ = model(input_ids=input_ids)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        load_time = time.time() - t_load_start

        # Inference latency
        latencies = []
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = model(input_ids=input_ids)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            for _ in range(self.measure_steps):
                t0 = time.time()
                _ = model(input_ids=input_ids)
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.time() - t0)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Throughput
        tokens_per_step = self.batch_size * self.max_seq_len
        throughput = tokens_per_step / avg_latency if avg_latency > 0 else 0

        # Memory
        peak_memory_gb = 0.0
        if self.measure_memory:
            if device.type == "cuda":
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "current_allocated_memory"):
                    peak_memory_gb = torch.mps.current_allocated_memory() / 1e9

        # Model size on disk (approximate)
        model_size_gb = total_params * 4 / 1e9  # FP32

        return BenchmarkResult(
            name=self.name,
            metrics={
                "throughput_tok_s": throughput,
                "inference_latency_ms": avg_latency * 1000,
                "peak_memory_gb": peak_memory_gb,
                "model_size_gb": model_size_gb,
                "forward_flops": forward_flops,
                "training_flops_per_step": training_flops_per_step,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "total_params_m": total_params / 1e6,
            },
            num_samples=self.measure_steps,
            eval_time=sum(latencies),
            metadata={
                "max_seq_len": self.max_seq_len,
                "batch_size": self.batch_size,
                "warmup_steps": self.warmup_steps,
                "measure_steps": self.measure_steps,
            },
        )
