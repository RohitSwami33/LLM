"""Training framework for decoder-only Transformer.

Modular design for future DeepSeek-inspired research:
    - Mixture of Experts (MoE)
    - Multi-head Latent Attention (MLA)
    - Manifold-Constrained Hyper Connections (mHC)
    - Multi-Token Prediction (MTP)
"""

from .trainer import Trainer, train_from_yaml
from .model import TransformerLM, ModelConfig
from .tokenizer import build_tokenizer, train_tokenizer
from .ema import EMA

__all__ = ["Trainer", "train_from_yaml", "TransformerLM", "ModelConfig",
           "build_tokenizer", "train_tokenizer", "EMA"]
