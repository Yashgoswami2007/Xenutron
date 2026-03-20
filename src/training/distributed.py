import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy
from typing import Optional
import os

def setup_distributed():
    """Setup distributed training"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training")
    
    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    return rank, world_size

def setup_fsdp(model: nn.Module, device_id: int, sharding_strategy: str = 'FULL_SHARD'):
    """Setup Fully Sharded Data Parallel"""
    # Define sharding strategy
    if sharding_strategy == 'FULL_SHARD':
        strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == 'SHARD_GRAD_OP':
        strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        strategy = ShardingStrategy.NO_SHARD
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy,
        device_id=device_id,
        # Enable mixed precision
        mixed_precision=True,
    )
    
    return fsdp_model

def setup_ddp(model: nn.Module, device_id: int):
    """Setup Distributed Data Parallel"""
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    return ddp_model

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()