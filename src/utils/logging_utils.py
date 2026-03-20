import logging
from torch.utils.tensorboard import SummaryWriter
import wandb
import os

class Logger:
    def __init__(self, log_dir: str, use_wandb: bool = False):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging to file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Setup TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Optionally initialize wandb
        if self.use_wandb:
            wandb.init(project="training", dir=log_dir)
    
    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """Log a dictionary of metrics"""
        for key, value in metrics.items():
            self.logger.info(f"{prefix}{key}: {value}")
            self.writer.add_scalar(f"{prefix}{key}", value, step)
            if self.use_wandb:
                wandb.log({f"{prefix}{key}": value}, step=step)
    
    def log_text(self, tag: str, text: str, step: int):
        self.writer.add_text(tag, text, step)
        if self.use_wandb:
            wandb.log({tag: text}, step=step)
    
    def close(self):
        self.writer.close()
        if self.use_wandb:
            wandb.finish()