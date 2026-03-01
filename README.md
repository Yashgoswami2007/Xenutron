# Xenutron

Xenutron is a **from-scratch GPT-style LLM stack** you can train and serve yourself.

Yes, this is the real thing (Transformer blocks, RoPE, GQA, SwiGLU, causal LM loss, tokenizer training, checkpointing) — not a toy single-file next-token demo.

## What you get

- Decoder-only Transformer (`RMSNorm + RoPE + GQA + SwiGLU`)
- Weight-tied token embedding / LM head
- SentencePiece BPE tokenizer training from your corpus
- Data ingestion from Hugging Face datasets
- Packed sequence training pipeline
- Mixed precision training + gradient accumulation + cosine decay
- Checkpoint save/load and text generation script
- Persona prompt template for **sarcastic Gen-Z** assistant style

## Project layout

- `src/xenutron/model.py` — core LLM architecture
- `src/xenutron/train.py` — training entrypoint
- `src/xenutron/generate.py` — inference script
- `src/xenutron/tokenizer.py` — tokenizer train/load utilities
- `src/xenutron/data.py` — dataset preprocessing and token packing
- `src/xenutron/personality.py` — Xenutron system prompt style
- `configs/` — model/training configs
- `scripts/train.sh` — quick training launcher
- `tests/` — sanity tests

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Train Xenutron

### 1) Pick a model size

- `configs/model_350m.yaml` for early experiments
- `configs/model_1_3b.yaml` for larger-scale runs

### 2) Launch training

```bash
./scripts/train.sh configs/model_350m.yaml configs/train_base.yaml
```

Quick smoke run:

```bash
./scripts/train.sh configs/model_350m.yaml configs/train_base.yaml 2000
```

## Generate text

```bash
python -m xenutron.generate \
  --checkpoint artifacts/checkpoints/step_10000.pt \
  --tokenizer artifacts/tokenizer/xenutron_spm.model \
  --prompt "Explain backprop like I'm sleep-deprived"
```

## Making this truly production-scale

To reach Llama/Qwen/DeepSeek class quality, you still need:

1. **Massive compute** (multi-node, high-bandwidth GPU clusters)
2. **Large curated corpora** (trillions of high-quality tokens)
3. **Distributed training stack** (FSDP/ZeRO/tensor parallel/pipeline parallel)
4. **Long training schedule** (weeks to months)
5. **Post-training** (SFT, preference optimization, safety tuning)
6. **Rigorous eval suite** (MMLU, GSM8K, HumanEval, MT-Bench, safety evals)

This repository gives you the core architecture and training workflow foundation to start that journey.

## Xenutron personality

Xenutron's default persona is in `src/xenutron/personality.py`: sarcastic and Gen-Z flavored but still safe/helpful.

Use this persona in SFT data to make style consistent at inference time.

## Notes

- First training run downloads dataset(s), so internet access is required.
- For serious runs, change `train_dataset` in `configs/train_base.yaml` to your prepared corpus.
- Start on small config, validate loss curves, then scale model/data/compute.
