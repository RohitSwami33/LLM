"""Tokenizer wrapper supporting SentencePiece and HuggingFace Tokenizers.

Handles:
    - Auto-training tokenizer from data if no tokenizer exists
    - Encoding / decoding with special token handling
    - Vocab size management
"""

import os
import json
from typing import List, Optional, Union


class SentencePieceTokenizer:
    """Wrapper around SentencePiece tokenizer."""

    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece is required: pip install sentencepiece")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_special_tokens:
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.sp.DecodeIds(ids)

    @property
    def bos_token_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_token_id(self) -> int:
        return self.sp.eos_id()

    @property
    def pad_token_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_token_id(self) -> int:
        return self.sp.unk_id()


class HuggingFaceTokenizer:
    """Wrapper around HuggingFace Tokenizers library."""

    def __init__(self, model_path: str):
        try:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(model_path)
        except ImportError:
            raise ImportError("tokenizers is required: pip install tokenizers")

        self.vocab_size = self.tok.get_vocab_size()

        # Add special tokens if not present
        special = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.tok.add_special_tokens(special)

        self._pad_id = self.tok.token_to_id("<pad>")
        self._bos_id = self.tok.token_to_id("<bos>")
        self._eos_id = self.tok.token_to_id("<eos>")
        self._unk_id = self.tok.token_to_id("<unk>")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        encoding = self.tok.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            ids = [self._bos_id] + ids + [self._eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(ids)

    @property
    def bos_token_id(self) -> int:
        return self._bos_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_id

    @property
    def unk_token_id(self) -> int:
        return self._unk_id


class TokenizerWrapper:
    """Unified tokenizer interface.

    Wraps SentencePiece or HuggingFace Tokenizer with a common API.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id


def train_tokenizer(
    data_path: str,
    output_path: str,
    vocab_size: int = 32000,
    tokenizer_type: str = "sentencepiece",
    model_type: str = "bpe",
) -> str:
    """Train a tokenizer from scratch on a text file or JSONL.

    Args:
        data_path: Path to training data (text file or JSONL).
        output_path: Path to save trained tokenizer.
        vocab_size: Vocabulary size.
        tokenizer_type: "sentencepiece" or "huggingface".
        model_type: For SentencePiece: "bpe", "unigram", "word", "char".

    Returns:
        Path to trained tokenizer.
    """
    # Extract texts from data
    texts = []
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                    if text.strip():
                        texts.append(text.strip())
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if tokenizer_type == "sentencepiece":
        return _train_sentencepiece(texts, output_path, vocab_size, model_type)
    elif tokenizer_type == "huggingface":
        return _train_huggingface(texts, output_path, vocab_size)
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")


def _train_sentencepiece(
    texts: List[str], output_path: str, vocab_size: int, model_type: str
) -> str:
    """Train SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("sentencepiece is required: pip install sentencepiece")

    # Write texts to temp file for SentencePiece training
    temp_path = output_path + ".tmp.txt"
    with open(temp_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    spm.SentencePieceTrainer.Train(
        input=temp_path,
        model_prefix=output_path.replace(".model", ""),
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        character_coverage=1.0,
        num_threads=os.cpu_count() or 4,
    )

    os.remove(temp_path)
    return output_path


def _train_huggingface(texts: List[str], output_path: str, vocab_size: int) -> str:
    """Train HuggingFace tokenizer."""
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    except ImportError:
        raise ImportError("tokenizers is required: pip install tokenizers")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
    )

    # Write to temp file
    temp_path = output_path + ".tmp.txt"
    with open(temp_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    tokenizer.train([temp_path], trainer)
    tokenizer.save(output_path)

    os.remove(temp_path)
    return output_path


def build_tokenizer(config: dict, data_path: Optional[str] = None) -> TokenizerWrapper:
    """Build tokenizer from config.

    Args:
        config: Tokenizer config dict with keys: type, vocab_size, model_path.
        data_path: Training data path (used if tokenizer needs training).

    Returns:
        TokenizerWrapper instance.
    """
    tok_type = config.get("type", "sentencepiece")
    vocab_size = config.get("vocab_size", 32000)
    model_path = config.get("model_path", f"training/tokenizer/tokenizer.model")

    # Train tokenizer if model doesn't exist
    if not os.path.exists(model_path):
        print(f"Tokenizer not found at {model_path}. Training new tokenizer...")
        if data_path is None:
            raise ValueError("data_path required to train tokenizer")
        model_path = train_tokenizer(
            data_path=data_path,
            output_path=model_path,
            vocab_size=vocab_size,
            tokenizer_type=tok_type,
        )
        print(f"Tokenizer trained and saved to {model_path}")

    # Load tokenizer
    if tok_type == "sentencepiece":
        tok = SentencePieceTokenizer(model_path)
    elif tok_type == "huggingface":
        tok = HuggingFaceTokenizer(model_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")

    return TokenizerWrapper(tok)
