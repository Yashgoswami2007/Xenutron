from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos) + (rot * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = cfg.use_flash_attn and hasattr(F, "scaled_dot_product_attention")

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.dropout = nn.Dropout(cfg.dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        repeat = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seqlen)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k, v = self._repeat_kv(k), self._repeat_kv(v)

        if self.use_flash:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device).triu(1)
            att = att + mask
            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class XenutronLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.cfg.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.cfg.max_seq_len :]
            logits = self.forward(x)["logits"][:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_idx.gather(-1, sampled)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_id is not None and torch.all(next_token == eos_id):
                break
        return input_ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
