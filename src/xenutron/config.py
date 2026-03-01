from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8
    d_ff: int = 8192
    dropout: float = 0.0
    rope_theta: float = 10000.0
    use_flash_attn: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    train_dataset: str = "HuggingFaceH4/ultrachat_200k"
    train_split: str = "train_sft"
    text_field: str = "messages"
    output_dir: str = "artifacts/checkpoints"

    tokenizer_prefix: str = "artifacts/tokenizer/xenutron_spm"
    tokenizer_vocab_size: int = 32000

    micro_batch_size: int = 2
    grad_accum_steps: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_steps: int = 10000
    grad_clip: float = 1.0

    sequence_length: int = 2048
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 500

    amp_dtype: str = "bfloat16"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig(**load_yaml_config(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    raw = load_yaml_config(path)
    if "betas" in raw:
        raw["betas"] = tuple(raw["betas"])
    return TrainingConfig(**raw)


def dump_config(path: str | Path, cfg: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
