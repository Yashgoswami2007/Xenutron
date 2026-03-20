import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Model parameters
    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 32
    intermediate_size: int = 5504
    max_seq_len: int = 2048
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_epochs: int = 3
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    
    # Logging
    logging_steps: int = 100
    log_dir: str = "./logs"
    
    # Distributed training
    use_distributed: bool = False
    use_fsdp: bool = False
    sharding_strategy: str = "FULL_SHARD"
    
    # Data
    data_path: str = "./data"
    tokenizer_path: str = "./tokenizer"
    use_ram_cache: bool = False
    
    # Inference
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    
    @classmethod
    def from_json(cls, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, config_path: str):
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)