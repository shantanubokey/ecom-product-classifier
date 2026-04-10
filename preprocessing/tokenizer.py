"""
TikTok-style BPE Tokenizer wrapper
Uses HuggingFace tokenizers (BPE) trained on the dataset vocab,
with fallback to a simple whitespace tokenizer for small datasets.
Also wraps the BERT tokenizer for the BERT model.
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence as NormSeq
from transformers import AutoTokenizer
from typing import List
import os


# ── BPE Tokenizer (TikTok-style) ──────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer trained on the product corpus.
    Handles Hinglish subword units efficiently.
    """

    def __init__(self, vocab_size: int = 8000, save_path: str = "bpe_tokenizer.json"):
        self.vocab_size = vocab_size
        self.save_path  = save_path
        self.tokenizer  = None

    def train(self, texts: List[str]):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = NormSeq([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
            min_frequency=1,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer = tokenizer
        tokenizer.save(self.save_path)
        print(f"[BPE] Trained on {len(texts)} texts. Vocab size: {tokenizer.get_vocab_size()}")

    def load(self):
        self.tokenizer = Tokenizer.from_file(self.save_path)

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is None:
            self.load()
        return self.tokenizer.encode(text).ids

    def encode_batch(self, texts: List[str], max_len: int = 32) -> List[List[int]]:
        if self.tokenizer is None:
            self.load()
        encoded = self.tokenizer.encode_batch(texts)
        result = []
        for enc in encoded:
            ids = enc.ids[:max_len]
            ids += [1] * (max_len - len(ids))   # pad with [PAD]=1
            result.append(ids)
        return result

    @property
    def vocab_size_actual(self) -> int:
        if self.tokenizer is None:
            self.load()
        return self.tokenizer.get_vocab_size()


# ── BERT Tokenizer wrapper ─────────────────────────────────────────────────────

class BERTTokenizerWrapper:
    """Wraps HuggingFace BERT tokenizer for multilingual support."""

    MODEL_NAME = "bert-base-multilingual-cased"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

    def encode(self, texts: List[str], max_length: int = 64):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
