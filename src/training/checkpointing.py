import torch
import os
from pathlib import Path
from typing import Dict, Any
import json

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                       metrics: Dict[str, float], global_step: int):
        """Save a checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Remove old checkpoints if needed
        self._cleanup_old_checkpoints()
        
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path: str = None):
        """Load a checkpoint"""
        if checkpoint_path is None:
            # Find the latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                print("No checkpoints found")
                return 0, 0
            
            # Sort by epoch and step (extracted from filename)
            def get_epoch_step(path):
                name = path.name
                try:
                    # Example: checkpoint_epoch_1_step_500.pt
                    parts = name.replace(".pt", "").split("_")
                    epoch = int(parts[2])
                    step = int(parts[4])
                    return epoch, step
                except (IndexError, ValueError):
                    return -1, -1

            checkpoints.sort(key=get_epoch_step)
            if not checkpoints:
                print("No valid checkpoints found")
                return 0, 0
                
            checkpoint_path = checkpoints[-1]
            
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return epoch, global_step
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > self.max_checkpoints:
            # Sort by creation time and remove oldest
            checkpoints.sort(key=os.path.getctime)
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()