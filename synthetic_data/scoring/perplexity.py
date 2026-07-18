"""Perplexity scoring using a small causal LM.

Model inference is synchronous, so :meth:`ascore` runs it in a thread pool to
stay compatible with the async pipeline.  The model is loaded lazily and
cached for reuse across samples.
"""

from __future__ import annotations

import asyncio
import math
from typing import Optional

from ..core.schema import Sample


class PerplexityScorer:
    """Compute token-level perplexity of a response with a tiny LM."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional
            raise RuntimeError(
                "transformers is required for perplexity scoring. "
                "Install with: pip install transformers torch"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()

    def score(self, text: str) -> float:
        self._ensure_loaded()
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"]
        import torch

        with torch.no_grad():
            out = self._model(input_ids)
            logits = out.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            n = shift_labels.numel()
            return math.exp(loss.item() / n)

    async def ascore(self, sample: Sample) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.score, sample.response or "")
