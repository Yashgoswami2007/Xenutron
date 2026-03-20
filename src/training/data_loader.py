import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, List


def _encode(tokenizer, text: str) -> List[int]:
    """Encode text to a plain list of int token IDs.

    Works with both:
    - HuggingFace `tokenizers` library  → returns tokenizers.Encoding object
    - SentencePiece                     → returns a list directly
    """
    result = tokenizer.encode(text)
    # tokenizers.Encoding is not subscriptable — extract .ids
    if hasattr(result, 'ids'):
        return result.ids
    return list(result)


# ── Arrow (Hugging Face) dataset ──────────────────────────────────────────────

class ArrowDataset(Dataset):
    """
    Wraps a Hugging Face Dataset. Tokenization is done LAZILY in __getitem__
    so the entire dataset is never loaded into RAM all at once.
    """

    def __init__(self, hf_dataset, tokenizer, max_seq_len: int = 512, cache_to_ram: bool = False):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache_to_ram = cache_to_ram
        self._cache = {}  # Stores tokenized lists

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.cache_to_ram and idx in self._cache:
            tokens = self._cache[idx]
        else:
            example = self.dataset[idx]
            prompt   = example.get('prompt',   '') or ''
            response = example.get('response', '') or ''
            text = f"{prompt} {response}".strip()

            tokens = _encode(self.tokenizer, text)
            # Truncate
            tokens = tokens[: self.max_seq_len + 1]

            if self.cache_to_ram:
                self._cache[idx] = tokens

        input_ids = tokens[:-1]
        labels    = tokens[1:]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0]      * pad_len
            labels    = labels    + [-100]   * pad_len
        else:
            input_ids = input_ids[:self.max_seq_len]
            labels    = labels   [:self.max_seq_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels':    torch.tensor(labels,    dtype=torch.long),
        }


# ── Plain-text file dataset ───────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Reads a text file (or directory of .txt files) and produces
    max_seq_len-length token chunks. Tokenization happens once at init
    but only on text (not tensors), keeping RAM usage manageable.
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 512):
        self.tokenizer  = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[List[int]] = []
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        if os.path.isdir(data_path):
            for filename in sorted(os.listdir(data_path)):
                if filename.endswith('.txt'):
                    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                        self._chunk_text(f.read())
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                self._chunk_text(f.read())

    def _chunk_text(self, text: str):
        tokens = _encode(self.tokenizer, text)
        # Slide a window of (max_seq_len + 1) with stride max_seq_len
        stride = self.max_seq_len
        for i in range(0, len(tokens) - 1, stride):
            chunk = tokens[i: i + self.max_seq_len + 1]
            if len(chunk) > 1:
                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = tokens[:-1]
        labels    = tokens[1:]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0]    * pad_len
            labels    = labels    + [-100] * pad_len
        else:
            input_ids = input_ids[:self.max_seq_len]
            labels    = labels   [:self.max_seq_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels':    torch.tensor(labels,    dtype=torch.long),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def create_data_loaders(
    data_source,
    tokenizer,
    batch_size: int = 2,
    max_seq_len: int = 512,
    # ALWAYS 0 on Windows to avoid multiprocessing deadlocks
    num_workers: int = 0,
    use_ram_cache: bool = False,
) -> Tuple[DataLoader, DataLoader]:

    if hasattr(data_source, '__class__') and 'Dataset' in data_source.__class__.__name__:
        dataset = ArrowDataset(data_source, tokenizer, max_seq_len, cache_to_ram=use_ram_cache)
    elif isinstance(data_source, str):
        dataset = TextDataset(data_source, tokenizer, max_seq_len)
    else:
        raise ValueError(f"Unsupported data source type: {type(data_source)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Check your data files.")

    train_size = max(1, int(0.9 * len(dataset)))
    val_size   = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # pin_memory only helps with CUDA
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,   # avoids partial-batch issues with AMP
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    ) if val_size > 0 else None

    return train_loader, val_loader