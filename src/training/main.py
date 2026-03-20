import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging
import argparse
import json

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from model import TransformerLM
from training.training_config import TrainingConfig
from training.checkpointing import CheckpointManager
from training.distributed import setup_distributed, setup_fsdp, setup_ddp, cleanup_distributed
from training.trainer import Trainer
from training.data_loader import create_data_loaders
from tokenizer.train_tokenizer import load_tokenizer
from datasets import load_from_disk, concatenate_datasets


def setup_logging(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log"), encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


def log_gpu_info(logger):
    """Log GPU memory and device info for debugging."""
    if torch.cuda.is_available():
        dev    = torch.cuda.current_device()
        name   = torch.cuda.get_device_name(dev)
        total  = torch.cuda.get_device_properties(dev).total_memory / 1e9
        logger.info(f"GPU: {name}  |  Total VRAM: {total:.1f} GB")
    else:
        logger.info("No GPU detected – running on CPU (training will be slow)")


def main(config: TrainingConfig):
    setup_logging(config.log_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  Xenutron Model Training")
    logger.info("=" * 60)

    # ── Device setup ─────────────────────────────────────────────────────────
    if config.use_distributed:
        rank, world_size = setup_distributed()
        device = torch.device('cuda', rank)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logger.error("CUDA is NOT available on this system!")
            logger.error("Training on CPU is disabled to prevent accidental slow training.")
            logger.error("Please install the correct PyTorch version with CUDA support.")
            raise RuntimeError(
                "CUDA requested but not available. "
                "Ensure you have a GPU and the correct 'torch' package installed."
            )

    logger.info(f"Device: {device}")
    log_gpu_info(logger)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer_path = os.path.join(config.tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Please train it first (the batch file does this automatically)."
        )

    logger.info(f"Loading tokenizer from {tokenizer_path} ...")
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = (
        tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else
        tokenizer.vocab_size()      if hasattr(tokenizer, 'vocab_size')     else
        config.vocab_size
    )
    logger.info(f"Vocabulary size: {vocab_size:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building model ...")
    model = TransformerLM(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_seq_len=config.max_seq_len,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params / 1e6:.1f}M")

    model = model.to(device)

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory after model load: {used:.2f} GB")

    if config.use_distributed:
        if config.use_fsdp:
            model = setup_fsdp(model, device.index, config.sharding_strategy)
        else:
            model = setup_ddp(model, device.index)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Estimate total optimiser steps accounting for gradient accumulation
    approx_steps_per_epoch = 1000
    total_opt_steps = (
        config.num_epochs * approx_steps_per_epoch // config.gradient_accumulation_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_opt_steps,
    )

    # ── Checkpoint manager ────────────────────────────────────────────────────
    checkpoint_manager = CheckpointManager("./checkpoints", config.save_total_limit)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model, optimizer, scheduler, device,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    start_epoch, global_step = checkpoint_manager.load_checkpoint(model, optimizer, scheduler)

    # ── Dataset loading ───────────────────────────────────────────────────────
    logger.info("Loading datasets ...")
    datasets_to_load = []

    genz_path  = os.path.join(config.data_path, "genz_combined_processed")
    oasst_path = os.path.join(config.data_path, "oasst1_processed")

    if os.path.exists(genz_path):
        logger.info(f"  Found genz dataset at {genz_path}")
        datasets_to_load.append(load_from_disk(genz_path))

    if os.path.exists(oasst_path):
        logger.info(f"  Found oasst1 dataset at {oasst_path}")
        datasets_to_load.append(load_from_disk(oasst_path))

    if not datasets_to_load:
        raise FileNotFoundError(
            f"No datasets found in '{config.data_path}'. "
            "Expected sub-folders: genz_combined_processed or oasst1_processed"
        )

    combined = (
        concatenate_datasets(datasets_to_load)
        if len(datasets_to_load) > 1
        else datasets_to_load[0]
    )
    logger.info(f"Total examples: {len(combined):,}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info("Creating data loaders ...")
    train_loader, val_loader = create_data_loaders(
        combined,
        tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        num_workers=0,   # must be 0 on Windows
        use_ram_cache=config.use_ram_cache,
    )
    logger.info(f"Train batches: {len(train_loader):,}  |  Val batches: {len(val_loader) if val_loader else 0:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting training ...")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size (micro): {config.batch_size}")
    logger.info(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  AMP enabled: {device.type == 'cuda'}")

    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        epoch_loss  = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                            dynamic_ncols=True)

        for batch_idx, batch in enumerate(progress_bar):
            # Skip batches already processed if resuming mid-epoch
            # (global_step is total steps across all epochs)
            # We assume total_steps_before_this_epoch = approx_steps_per_epoch * epoch
            # But a more precise way is to just keep training and only skip if global_step
            # has not reached the saved global_step yet.
            
            if global_step > 0 and batch_idx <= (global_step % len(train_loader)) and epoch == start_epoch:
                 # This is a bit approximate if batch size or dataset changed, 
                 # but for same config it works.
                 # Let's use a simpler check: if we are in the start_epoch, 
                 # we can skip until we reach the relative step.
                 pass # We'll refine this if needed, but for now tqdm will just show progress
            
            # More robust: use a counter to skip
            if epoch == start_epoch and batch_idx < (global_step % len(train_loader)):
                continue

            loss = trainer.train_step(batch['input_ids'], batch['labels'])
            epoch_loss  += loss
            num_batches += 1
            global_step += 1

            if global_step % config.logging_steps == 0:
                avg = epoch_loss / num_batches
                progress_bar.set_postfix({'loss': f'{loss:.4f}', 'avg': f'{avg:.4f}'})

                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / 1e9
                    progress_bar.set_postfix(
                        {'loss': f'{loss:.4f}', 'avg': f'{avg:.4f}', 'GPU_GB': f'{mem:.2f}'}
                    )

            if global_step % config.save_steps == 0:
                checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {'loss': loss}, global_step,
                )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} done – avg loss: {avg_epoch_loss:.4f}")

        # Validation
        if val_loader:
            val_metrics = trainer.evaluate(val_loader)
            logger.info(
                f"Validation – loss: {val_metrics['loss']:.4f} "
                f"| perplexity: {val_metrics['perplexity']:.2f}"
            )

    if config.use_distributed:
        cleanup_distributed()

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Xenutron Transformer model')
    parser.add_argument('--config',          type=str, default='./config.json')
    parser.add_argument('--use_distributed', action='store_true')
    parser.add_argument('--use_fsdp',        action='store_true')
    args = parser.parse_args()

    try:
        config = TrainingConfig.from_json(args.config)
        config.use_distributed = args.use_distributed
        config.use_fsdp        = args.use_fsdp
        main(config)
    except Exception:
        # Make sure the error is always written to the log file AND visible in terminal
        logging.getLogger(__name__).exception("Fatal error during training:")
        sys.exit(1)