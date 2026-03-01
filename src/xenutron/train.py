from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_model_config, load_training_config
from .data import build_train_dataset, export_text_corpus
from .model import XenutronLM
from .tokenizer import XenutronTokenizer, train_sentencepiece


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Xenutron from scratch")
    parser.add_argument("--model-config", default="configs/model_350m.yaml")
    parser.add_argument("--train-config", default="configs/train_base.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    mcfg = load_model_config(args.model_config)
    tcfg = load_training_config(args.train_config)
    set_seed(tcfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Preparing tokenizer corpus...")
    corpus_txt = export_text_corpus(
        dataset_name=tcfg.train_dataset,
        split=tcfg.train_split,
        text_field=tcfg.text_field,
        output_file="artifacts/tokenizer/corpus.txt",
    )
    tokenizer_model = train_sentencepiece(corpus_txt, tcfg.tokenizer_prefix, tcfg.tokenizer_vocab_size)
    tokenizer = XenutronTokenizer(tokenizer_model)

    mcfg.vocab_size = tokenizer.vocab_size
    model = XenutronLM(mcfg).to(device)
    print(f"Model params: {model.num_parameters() / 1e9:.3f}B")

    dataset = build_train_dataset(
        dataset_name=tcfg.train_dataset,
        split=tcfg.train_split,
        text_field=tcfg.text_field,
        tokenizer=tokenizer,
        seq_len=tcfg.sequence_length,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=tcfg.micro_batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    optimizer = AdamW(model.parameters(), lr=tcfg.learning_rate, betas=tcfg.betas, weight_decay=tcfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and tcfg.amp_dtype != "bfloat16"))
    amp_dtype = torch.bfloat16 if tcfg.amp_dtype == "bfloat16" else torch.float16

    output_dir = Path(tcfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    pbar = tqdm(range(1, tcfg.max_steps + 1), desc="training")
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(tcfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                out = model(input_ids, labels)
                loss = out["loss"] / tcfg.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        lr = cosine_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.learning_rate, tcfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()

        if step % tcfg.log_every == 0:
            pbar.set_postfix(loss=f"{total_loss:.4f}", lr=f"{lr:.2e}")

        if step % tcfg.save_every == 0 or step == tcfg.max_steps:
            ckpt_path = output_dir / f"step_{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": vars(mcfg),
                    "train_config": vars(tcfg),
                    "tokenizer_model": tokenizer.model_path,
                    "step": step,
                },
                ckpt_path,
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
