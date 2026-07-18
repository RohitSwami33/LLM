"""Text generation with multiple decoding strategies."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict


def _encode(tokenizer, text: str) -> List[int]:
    result = tokenizer.encode(text)
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        return result[0]
    return result


def _decode(tokenizer, token_ids: List[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    return "".join(tokenizer.id_to_piece(i) for i in token_ids)


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 200,
                    device: torch.device = None) -> str:
    """Greedy decoding: always pick the highest-probability token."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.tensor([_encode(tokenizer, prompt)], dtype=torch.long, device=device)
    max_len = getattr(model, "max_seq_len", getattr(model.config if hasattr(model, "config") else None, "max_seq_len", 2048))

    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.size(1) <= max_len else input_ids[:, -max_len:]
        logits, _ = model(input_ids=idx_cond)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == getattr(tokenizer, "eos_token_id", 2):
            break

    return _decode(tokenizer, input_ids[0].tolist()[len(_encode(tokenizer, prompt)):])


@torch.no_grad()
def generate_temperature(model, tokenizer, prompt: str, max_new_tokens: int = 200,
                         temperature: float = 0.8, device: torch.device = None) -> str:
    """Temperature sampling: scale logits by temperature before softmax."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.tensor([_encode(tokenizer, prompt)], dtype=torch.long, device=device)
    max_len = getattr(model, "max_seq_len", getattr(model.config if hasattr(model, "config") else None, "max_seq_len", 2048))

    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.size(1) <= max_len else input_ids[:, -max_len:]
        logits, _ = model(input_ids=idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == getattr(tokenizer, "eos_token_id", 2):
            break

    return _decode(tokenizer, input_ids[0].tolist()[len(_encode(tokenizer, prompt)):])


@torch.no_grad()
def generate_top_k(model, tokenizer, prompt: str, max_new_tokens: int = 200,
                   temperature: float = 0.8, top_k: int = 50,
                   device: torch.device = None) -> str:
    """Top-k sampling: only sample from the top-k most probable tokens."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.tensor([_encode(tokenizer, prompt)], dtype=torch.long, device=device)
    max_len = getattr(model, "max_seq_len", getattr(model.config if hasattr(model, "config") else None, "max_seq_len", 2048))

    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.size(1) <= max_len else input_ids[:, -max_len:]
        logits, _ = model(input_ids=idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = top_k_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < threshold, torch.tensor(float("-inf"), device=device), logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == getattr(tokenizer, "eos_token_id", 2):
            break

    return _decode(tokenizer, input_ids[0].tolist()[len(_encode(tokenizer, prompt)):])


@torch.no_grad()
def generate_top_p(model, tokenizer, prompt: str, max_new_tokens: int = 200,
                   temperature: float = 0.8, top_p: float = 0.95,
                   device: torch.device = None) -> str:
    """Top-p (nucleus) sampling: sample from the smallest set of tokens with cumulative probability >= top_p."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.tensor([_encode(tokenizer, prompt)], dtype=torch.long, device=device)
    max_len = getattr(model, "max_seq_len", getattr(model.config if hasattr(model, "config") else None, "max_seq_len", 2048))

    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.size(1) <= max_len else input_ids[:, -max_len:]
        logits, _ = model(input_ids=idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == getattr(tokenizer, "eos_token_id", 2):
            break

    return _decode(tokenizer, input_ids[0].tolist()[len(_encode(tokenizer, prompt)):])


STRATEGIES = {
    "greedy": generate_greedy,
    "temperature": generate_temperature,
    "top_k": generate_top_k,
    "top_p": generate_top_p,
}


def generate(model, tokenizer, prompt: str, strategy: str = "top_p",
             max_new_tokens: int = 200, device: torch.device = None, **kwargs) -> str:
    """Generate text using the specified decoding strategy."""
    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGIES.keys())}")
    return fn(model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device, **kwargs)


def generate_all_samples(model, tokenizer, prompts: List[str],
                         strategies: Optional[Dict[str, dict]] = None,
                         max_new_tokens: int = 200, device: torch.device = None) -> List[Dict]:
    """Generate samples for all prompts and strategies.

    Returns list of dicts with keys: prompt, strategy, output.
    """
    if strategies is None:
        strategies = {"top_p": {"temperature": 0.8, "top_p": 0.95}}

    results = []
    for prompt in prompts:
        for strat_name, strat_kwargs in strategies.items():
            output = generate(
                model, tokenizer, prompt,
                strategy=strat_name, max_new_tokens=max_new_tokens,
                device=device, **strat_kwargs,
            )
            results.append({
                "prompt": prompt,
                "strategy": strat_name,
                "output": output,
            })
    return results
