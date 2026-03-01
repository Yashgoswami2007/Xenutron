from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .tokenizer import XenutronTokenizer


def normalize_chat_example(record: dict, text_field: str) -> str:
    value = record[text_field]
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        chunks = []
        for turn in value:
            if isinstance(turn, dict):
                role = turn.get("role", "user").upper()
                content = turn.get("content", "")
                chunks.append(f"[{role}] {content}")
            else:
                chunks.append(str(turn))
        return "\n".join(chunks).strip()

    return json.dumps(value, ensure_ascii=False)


def export_text_corpus(dataset_name: str, split: str, text_field: str, output_file: str) -> str:
    ds = load_dataset(dataset_name, split=split)
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for ex in ds:
            text = normalize_chat_example(ex, text_field)
            if text:
                f.write(text.replace("\n", " ") + "\n")
    return str(out)


class PackedTokenDataset(Dataset):
    def __init__(self, token_sequences: Iterable[list[int]], seq_len: int):
        buffer: list[int] = []
        packed: list[torch.Tensor] = []
        for seq in token_sequences:
            buffer.extend(seq)
            while len(buffer) >= seq_len + 1:
                chunk = buffer[: seq_len + 1]
                packed.append(torch.tensor(chunk, dtype=torch.long))
                buffer = buffer[seq_len + 1 :]
        self.samples = packed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.samples[idx]
        return {"input_ids": tokens.clone(), "labels": tokens.clone()}


def build_train_dataset(
    dataset_name: str,
    split: str,
    text_field: str,
    tokenizer: XenutronTokenizer,
    seq_len: int,
    max_samples: int | None = None,
) -> PackedTokenDataset:
    ds = load_dataset(dataset_name, split=split)
    tokenized = []
    for i, ex in enumerate(ds):
        text = normalize_chat_example(ex, text_field)
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        tokenized.append(ids)
        if max_samples and i + 1 >= max_samples:
            break
    return PackedTokenDataset(tokenized, seq_len)
