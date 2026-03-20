import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, Tuple
import time


class Trainer:
    def __init__(self, model: nn.Module, optimizer, scheduler, device,
                 gradient_accumulation_steps: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logger = logging.getLogger(__name__)

        # AMP scaler – only used when a CUDA device is available
        self.use_amp = device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self._accum_step = 0  # tracks micro-steps within an accumulation window

    # ── Single training micro-step (called once per batch) ──────────────────

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Performs one micro-step. Gradient accumulation is handled internally.
        Returns the loss value for this micro-step.
        """
        self.model.train()

        input_ids = input_ids.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Zero gradients at the START of every accumulation window
        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(input_ids)
            # Scale loss by accumulation steps so effective LR is unchanged
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                outputs.view(-1, outputs.size(-1)), labels.view(-1)
            ) / self.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        self._accum_step += 1

        # Optimizer step at the END of each accumulation window
        if self._accum_step >= self.gradient_accumulation_steps:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            self._accum_step = 0

        return loss.item() * self.gradient_accumulation_steps  # return unscaled loss

    # ── Validation step ──────────────────────────────────────────────────────

    def evaluate_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.eval()

        with torch.no_grad():
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(input_ids)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    outputs.view(-1, outputs.size(-1)), labels.view(-1)
                )

        return loss.item()

    # ── Full evaluation pass ─────────────────────────────────────────────────

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(input_ids)
                    loss = nn.CrossEntropyLoss(ignore_index=-100)(
                        outputs.view(-1, outputs.size(-1)), labels.view(-1)
                    )

                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {'loss': avg_loss, 'perplexity': perplexity}

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, save_path: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
        }
        path = f"{save_path}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved at {path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")