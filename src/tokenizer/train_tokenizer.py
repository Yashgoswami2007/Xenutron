import os
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import sentencepiece as spm
from datasets import load_dataset
import argparse

logger = logging.getLogger(__name__)


def train_bpe_tokenizer(files: list, vocab_size: int = 50000, save_path: str = "./tokenizer"):
    """Train a BPE tokenizer on text files."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # BpeTrainer: initial_alphabet is inferred from training data automatically
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", 2),
            ("[EOS]", 3),
        ],
    )

    tokenizer.train(files, trainer)

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    print(f"[INFO] BPE tokenizer saved to {save_path}/tokenizer.json")
    return tokenizer


def train_sentencepiece_tokenizer(text_files: list, vocab_size: int = 50000, save_path: str = "./tokenizer"):
    """Train a SentencePiece tokenizer."""
    temp_file = os.path.join(save_path, "temp_train.txt")
    os.makedirs(save_path, exist_ok=True)
    with open(temp_file, 'w', encoding='utf-8') as f:
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as tf:
                f.write(tf.read() + '\n')

    model_prefix = os.path.join(save_path, "tokenizer")
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        normalization_rule_name='nmt_nfkc_cf',
    )

    os.remove(temp_file)
    print(f"[INFO] SentencePiece tokenizer saved to {model_prefix}.model")
    return model_prefix


def load_tokenizer(tokenizer_path: str):
    """Load a previously saved tokenizer (BPE or SentencePiece)."""
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load BPE tokenizer: {e}")
        try:
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            return sp
        except Exception as e2:
            logger.error(f"Failed to load SentencePiece tokenizer: {e2}")
            raise RuntimeError(f"Could not load tokenizer from {tokenizer_path}")


def main():
    parser = argparse.ArgumentParser(description='Train tokenizer')
    parser.add_argument('--data_path',      type=str, required=True)
    parser.add_argument('--vocab_size',     type=int, default=50000)
    parser.add_argument('--save_path',      type=str, default='./tokenizer')
    parser.add_argument('--tokenizer_type', type=str, default='bpe',
                        choices=['bpe', 'sentencepiece'])
    args = parser.parse_args()

    # Load dataset and write temp files
    dataset = load_dataset("text", data_files=args.data_path)

    temp_files = []
    for i, split in enumerate(dataset):
        temp_file = f"temp_train_{i}.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in dataset[split]['text']:
                f.write(text + '\n')
        temp_files.append(temp_file)

    if args.tokenizer_type == 'bpe':
        train_bpe_tokenizer(temp_files, args.vocab_size, args.save_path)
    else:
        train_sentencepiece_tokenizer(temp_files, args.vocab_size, args.save_path)

    for tf in temp_files:
        if os.path.exists(tf):
            os.remove(tf)


if __name__ == "__main__":
    main()