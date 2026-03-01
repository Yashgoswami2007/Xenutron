from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class XenutronTokenizer:
    def __init__(self, model_path: str):
        self.processor = spm.SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self.processor.encode(text)
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)


def train_sentencepiece(input_file: str, model_prefix: str, vocab_size: int) -> str:
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    return f"{model_prefix}.model"
