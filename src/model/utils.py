import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_causal_mask(seq_len: int):
    """Create a causal mask for attention"""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask

def init_weights(module):
    """Initialize weights with normal distribution"""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def get_activation_function(activation: str = 'swiglu'):
    """Return activation function"""
    if activation == 'swiglu':
        return lambda x: F.silu(x) * x
    elif activation == 'gelu':
        return F.gelu
    else:
        raise ValueError(f"Unknown activation function: {activation}")