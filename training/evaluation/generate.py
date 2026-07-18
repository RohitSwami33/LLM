"""Sample generation for evaluation and monitoring."""

import torch
from typing import Optional, List


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """Generate text samples from prompts.

    Args:
        model: The language model.
        tokenizer: Tokenizer for encoding/decoding.
        prompts: List of prompt strings.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        top_p: Nucleus sampling.
        device: Compute device.

    Returns:
        List of generated strings (prompt + completion).
    """
    model.eval()
    results = []

    for prompt in prompts:
        # Tokenize prompt
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Decode
        generated = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
        results.append(generated)

    return results


DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list of numbers.",
    "What are the benefits of regular exercise?",
    "Describe the process of photosynthesis.",
    "How does a neural network learn?",
]


@torch.no_grad()
def generate_evaluation_samples(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 200,
) -> str:
    """Generate and format samples for logging.

    Returns a formatted string suitable for logging.
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    samples = generate_samples(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    output = []
    for prompt, sample in zip(prompts, samples):
        output.append(f"Prompt: {prompt}")
        output.append(f"Generated: {sample}")
        output.append("---")

    return "\n".join(output)
