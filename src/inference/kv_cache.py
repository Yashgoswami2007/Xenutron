import torch

class KVCache:
    def __init__(self, num_layers: int, batch_size: int, num_heads: int, seq_len: int, head_dim: int):
        self.k_cache = [torch.zeros(batch_size, num_heads, seq_len, head_dim) for _ in range(num_layers)]
        self.v_cache = [torch.zeros(batch_size, num_heads, seq_len, head_dim) for _ in range(num_layers)]
        self.seq_len = seq_len
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Update KV cache for a given layer"""
        self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
        self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)
    
    def get(self, layer_idx: int, start_pos: int, end_pos: int):
        """Get KV cache for a given layer"""
        return self.k_cache[layer_idx][:, :, start_pos:end_pos, :], self.v_cache[layer_idx][:, :, start_pos:end_pos, :]