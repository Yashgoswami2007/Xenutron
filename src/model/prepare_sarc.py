# scripts/prepare_sarc.py
from datasets import load_dataset
import os

def main(out_path="./data/sarc.txt"):
    # download the entire dataset (train/valid/test splits)
    ds = load_dataset("sarc", "all")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for split in ds:
            for example in ds[split]:
                # the corpus stores parent/child reddit comments,
                # `example["sarcastic"]` is a 0/1 label.
                # we'll just use the child comment (text) for modelling:
                text = example["comment"]  # or "parent"
                f.write(text.replace("\n", " ") + "\n")

if __name__ == "__main__":
    main()