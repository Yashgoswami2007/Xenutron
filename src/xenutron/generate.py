from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .model import XenutronLM
from .personality import format_chat_prompt
from .tokenizer import XenutronTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with Xenutron checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)

    cfg = ModelConfig(**ckpt["model_config"])
    model = XenutronLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    tokenizer = XenutronTokenizer(args.tokenizer)
    prompt = format_chat_prompt(args.prompt)
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long, device=device)

    out_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=tokenizer.eos_id,
    )
    print(tokenizer.decode(out_ids[0].tolist()))


if __name__ == "__main__":
    main()
