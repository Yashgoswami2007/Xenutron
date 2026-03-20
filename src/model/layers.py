import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute the inverse frequency
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute the cos/sin tables
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int):
        cos = self.cos_cached[:, :, :seq_len, : self.dim]
        sin = self.sin_cached[:, :, :seq_len, : self.dim]
        return cos, sin

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Apply rotary embeddings to query and key tensors
    cos = cos[:, :, :q.shape[-2], :]
    sin = sin[:, :, :q.shape[-2], :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        
        assert self.head_dim * num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = causal_mask.unsqueeze(0) | (attention_mask == 0)
        else:
            combined_mask = causal_mask.unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(combined_mask, float('-inf'))

        attention_probs = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_probs, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)

        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        
    def forward(self, x):
        # SwiGLU activation
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, max_seq_len: int = 2048):
        super().__init__()
        self.attention = Attention(hidden_size, num_heads, max_seq_len)
        self.feed_forward = FeedForward(hidden_size, intermediate_size)
        self.attention_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Attention with residual connection
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask)
        x = residual + x
        
        # Feed forward with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, 
                 intermediate_size: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, num_heads, intermediate_size, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # Final RMSNorm
        self.norm = RMSNorm(hidden_size)
        
        # Language model head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Get embeddings
        x = self.embeddings(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Compute logits
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, 
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0):
        was_training = self.training
        self.eval()
        
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if input_ids.size(1) > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]

                logits = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Correctly mask the original logits
                    for i in range(next_token_logits.size(0)):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)

        if was_training:
            self.train()
        
        return input_ids