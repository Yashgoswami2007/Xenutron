import torch

from xenutron.config import ModelConfig
from xenutron.model import XenutronLM


def test_forward_shapes_and_loss():
    cfg = ModelConfig(
        vocab_size=128,
        max_seq_len=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        dropout=0.0,
        use_flash_attn=False,
    )
    model = XenutronLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(x, x)

    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"]).item()


def test_generate_progresses_sequence():
    cfg = ModelConfig(
        vocab_size=128,
        max_seq_len=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        dropout=0.0,
        use_flash_attn=False,
    )
    model = XenutronLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = model.generate(x, max_new_tokens=4, temperature=0.0)
    assert y.shape[1] == 12
