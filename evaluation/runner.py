"""Main evaluation runner — orchestrates all benchmarks, generation, and reporting."""

import os
import sys
import json
import time
import torch
import yaml
from datetime import datetime
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks import get_benchmark, ALL_BENCHMARKS
from evaluation.benchmarks.efficiency import EfficiencyBenchmark
from evaluation.benchmarks.base import EvalResults, BenchmarkResult
from evaluation.generation import generate_all_samples
from evaluation.prompts import DEFAULT_PROMPTS, get_prompts


class EvalRunner:
    """Unified evaluation runner for any decoder-only Transformer.

    Usage:
        runner = EvalRunner.from_checkpoint("checkpoints/best_model")
        results = runner.evaluate(benchmarks=["wikitext2", "hellaswag"])
        runner.save_results(results, "output_dir/")
    """

    def __init__(self, model, tokenizer, device: torch.device = None,
                 config: Dict[str, Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (next(model.parameters()).device if model else torch.device("cpu"))
        self.config = config or {}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config_path: str = None,
                        device: torch.device = None) -> "EvalRunner":
        """Load model from checkpoint and create runner."""
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load config
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        elif "config" in ckpt:
            config = ckpt["config"]
        else:
            config = {}

        # Build model
        from training.model.config import ModelConfig
        from training.model.model import TransformerLM

        model_config_dict = config.get("model", {})
        if not model_config_dict and "model_config" in ckpt:
            model_config_dict = ckpt["model_config"]

        model_config = ModelConfig.from_dict(model_config_dict)
        if "model" in ckpt:
            model = TransformerLM(model_config)
            model.load_state_dict(ckpt["model"])
        else:
            raise ValueError("Checkpoint does not contain 'model' state dict")

        model = model.to(device)
        model.eval()

        # Build tokenizer
        tok_config = config.get("tokenizer", {})
        from training.tokenizer.tokenizer import build_tokenizer
        data_path = config.get("dataset", {}).get("path")
        tokenizer = build_tokenizer(tok_config, data_path=data_path)

        step = ckpt.get("step", 0)
        arch = "MoE" if model.is_moe else "Transformer"
        print(f"Loaded checkpoint: {checkpoint_path} (step {step})")
        print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M [{arch}]")
        if model.is_moe:
            total, active = model.get_total_and_active_params()
            print(f"  Total: {total/1e6:.1f}M | Active/token: {active/1e6:.1f}M")
            print(f"  Experts: {model_config.moe_num_experts} | Top-k: {model_config.moe_top_k}")

        runner = cls(model, tokenizer, device, config)
        runner.checkpoint_path = checkpoint_path
        runner.checkpoint_step = step
        return runner

    def evaluate(self, benchmarks: List[str] = None, generate: bool = True,
                 efficiency: bool = True, prompts: List[str] = None,
                 max_new_tokens: int = 200, max_samples: int = None) -> EvalResults:
        """Run full evaluation suite.

        Args:
            benchmarks: List of benchmark names. None = all available.
            generate: Whether to generate text samples.
            efficiency: Whether to measure efficiency metrics.
            prompts: Custom prompts for generation. None = defaults.
            max_new_tokens: Max tokens for generation.
            max_samples: Max samples per benchmark (None = all).
        """
        results = EvalResults()
        results.checkpoint_path = self.checkpoint_path
        results.timestamp = datetime.now().isoformat()

        # Model info
        results.model_info = self._get_model_info()

        # Benchmarks
        if benchmarks is None:
            benchmarks = list(ALL_BENCHMARKS.keys())

        for bench_name in benchmarks:
            print(f"\n{'='*50}")
            print(f"Running benchmark: {bench_name}")
            print(f"{'='*50}")

            try:
                benchmark = get_benchmark(bench_name)
                if max_samples and hasattr(benchmark, "max_samples"):
                    benchmark.max_samples = max_samples

                bench_result = benchmark.evaluate(self.model, self.tokenizer, self.device)
                results.benchmarks[bench_name] = bench_result
                print(f"  {bench_result}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.benchmarks[bench_name] = BenchmarkResult(
                    name=bench_name, metrics={"error": str(e)}, num_samples=0
                )

        # Efficiency
        if efficiency:
            print(f"\n{'='*50}")
            print("Running efficiency benchmark")
            print(f"{'='*50}")

            try:
                eff_bench = EfficiencyBenchmark(
                    max_seq_len=self.config.get("model", {}).get("max_seq_len", 1024),
                    measure_memory=True,
                )
                eff_result = eff_bench.evaluate(self.model, self.tokenizer, self.device)
                results.efficiency = eff_result.metrics
                print(f"  {eff_result}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.efficiency = {"error": str(e)}

        # Generation
        if generate:
            print(f"\n{'='*50}")
            print("Generating text samples")
            print(f"{'='*50}")

            gen_prompts = prompts or DEFAULT_PROMPTS
            strategies = {
                "greedy": {},
                "top_p": {"temperature": 0.8, "top_p": 0.95},
                "top_k": {"temperature": 0.8, "top_k": 50},
            }

            results.generation_samples = generate_all_samples(
                self.model, self.tokenizer, gen_prompts,
                strategies=strategies, max_new_tokens=max_new_tokens,
                device=self.device,
            )
            for sample in results.generation_samples:
                print(f"\n  [{sample['strategy']}] {sample['prompt'][:60]}...")
                print(f"  → {sample['output'][:120]}...")

        return results

    def _get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        vocab_size = getattr(self.model, "vocab_size", 0)
        if not vocab_size and hasattr(self.model, "lm_head"):
            vocab_size = self.model.lm_head.weight.shape[0]

        info = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_m": round(total_params / 1e6, 2),
            "vocab_size": vocab_size,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_path": self.checkpoint_path,
        }

        # MoE-specific metrics
        if hasattr(self.model, "is_moe") and self.model.is_moe:
            total, active = self.model.get_total_and_active_params()
            info["is_moe"] = True
            info["active_params"] = active
            info["active_params_m"] = round(active / 1e6, 2)
            info["moe_num_experts"] = self.model.config.moe_num_experts
            info["moe_top_k"] = self.model.config.moe_top_k
            info["moe_capacity_factor"] = self.model.config.moe_capacity_factor
            info["moe_shared_expert"] = self.model.config.moe_shared_expert
            # Calculate parameter efficiency
            info["param_efficiency"] = round(active / total * 100, 2) if total > 0 else 0
        else:
            info["is_moe"] = False

        # Try to extract architecture details from config
        model_cfg = self.config.get("model", {})
        for key in ["d_model", "n_heads", "n_layers", "d_ff", "max_seq_len",
                     "norm_type", "activation", "rope", "flash_attention", "architecture"]:
            if key in model_cfg:
                info[key] = model_cfg[key]

        return info

    def save_results(self, results: EvalResults, output_dir: str):
        """Save all results to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

        self._save_json(results, output_dir)
        self._save_markdown(results, output_dir)
        self._save_csv(results, output_dir)
        self._save_generation_samples(results, output_dir)
        self._save_plots(results, output_dir)

        print(f"\nAll results saved to {output_dir}/")

    def _save_json(self, results: EvalResults, output_dir: str):
        data = {
            "checkpoint": results.checkpoint_path,
            "timestamp": results.timestamp,
            "model_info": results.model_info,
            "benchmarks": {},
            "efficiency": results.efficiency,
        }
        for name, bench in results.benchmarks.items():
            data["benchmarks"][name] = {
                "metrics": bench.metrics,
                "num_samples": bench.num_samples,
                "eval_time": bench.eval_time,
                "metadata": bench.metadata,
            }

        with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_markdown(self, results: EvalResults, output_dir: str):
        lines = []
        lines.append("# Evaluation Report\n")
        lines.append(f"**Checkpoint:** `{results.checkpoint_path}`")
        lines.append(f"**Timestamp:** {results.timestamp}")
        lines.append(f"**Parameters:** {results.model_info.get('total_params_m', '?')}M\n")

        lines.append("## Benchmark Results\n")
        lines.append("| Benchmark | Metric | Value | Samples | Time (s) |")
        lines.append("|-----------|--------|------:|--------:|---------:|")
        for name, bench in results.benchmarks.items():
            for metric, value in bench.metrics.items():
                if isinstance(value, float):
                    val_str = f"{value:.4f}"
                else:
                    val_str = str(value)
                lines.append(f"| {name} | {metric} | {val_str} | {bench.num_samples} | {bench.eval_time:.1f} |")

        if results.efficiency:
            lines.append("\n## Efficiency\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|------:|")
            for metric, value in results.efficiency.items():
                if isinstance(value, float):
                    lines.append(f"| {metric} | {value:.4f} |")
                else:
                    lines.append(f"| {metric} | {value} |")

        if results.generation_samples:
            lines.append("\n## Generation Samples\n")
            for sample in results.generation_samples:
                lines.append(f"### [{sample['strategy']}] {sample['prompt'][:80]}\n")
                lines.append(f"```\n{sample['output']}\n```\n")

        with open(os.path.join(output_dir, "evaluation.md"), "w") as f:
            f.write("\n".join(lines))

    def _save_csv(self, results: EvalResults, output_dir: str):
        lines = ["benchmark,metric,value,num_samples,eval_time"]
        for name, bench in results.benchmarks.items():
            for metric, value in bench.metrics.items():
                lines.append(f"{name},{metric},{value},{bench.num_samples},{bench.eval_time:.2f}")
        for metric, value in results.efficiency.items():
            lines.append(f"efficiency,{metric},{value},,")

        with open(os.path.join(output_dir, "comparison.csv"), "w") as f:
            f.write("\n".join(lines))

    def _save_generation_samples(self, results: EvalResults, output_dir: str):
        path = os.path.join(output_dir, "generation_samples.txt")
        with open(path, "w") as f:
            for sample in results.generation_samples:
                f.write(f"{'='*60}\n")
                f.write(f"Strategy: {sample['strategy']}\n")
                f.write(f"Prompt: {sample['prompt']}\n")
                f.write(f"Output:\n{sample['output']}\n\n")

    def _save_plots(self, results: EvalResults, output_dir: str):
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not available, skipping plots")
            return

        # Benchmark radar chart
        self._plot_radar(results, plots_dir)

        # Efficiency bar chart
        self._plot_efficiency(results, plots_dir)

    def _plot_radar(self, results: EvalResults, plots_dir: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return

        metrics = {}
        for name, bench in results.benchmarks.items():
            if "accuracy" in bench.metrics:
                metrics[name] = bench.metrics["accuracy"]
            elif "perplexity" in bench.metrics:
                ppl = bench.metrics["perplexity"]
                metrics[name] = min(1.0, 1.0 / max(ppl, 1.0))

        if len(metrics) < 3:
            return

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        labels = list(metrics.keys())
        values = list(metrics.values())
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += [angles[0]]

        ax.plot(angles, values_plot, "o-", linewidth=2)
        ax.fill(angles, values_plot, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_title("Benchmark Performance", size=14, y=1.08)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "radar_chart.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_efficiency(self, results: EvalResults, plots_dir: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        eff = results.efficiency
        if not eff or "error" in eff:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Throughput
        if "throughput_tok_s" in eff:
            axes[0].bar(["Throughput"], [eff["throughput_tok_s"]], color="#2196F3")
            axes[0].set_ylabel("tokens/sec")
            axes[0].set_title("Inference Throughput")

        # Memory
        if "peak_memory_gb" in eff:
            axes[1].bar(["Peak Memory"], [eff["peak_memory_gb"]], color="#FF9800")
            axes[1].set_ylabel("GB")
            axes[1].set_title("Peak Memory Usage")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "efficiency.png"), dpi=150, bbox_inches="tight")
        plt.close()
