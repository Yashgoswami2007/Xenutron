import torch
import os
import sys
from pathlib import Path
import json

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import TransformerLM
from training.training_config import TrainingConfig
from tokenizer.train_tokenizer import load_tokenizer

def find_latest_checkpoint(checkpoint_dir: str):
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    
    def get_epoch_step(path):
        name = path.name
        try:
            # Example: checkpoint_epoch_1_step_500.pt
            parts = name.replace(".pt", "").split("_")
            epoch = int(parts[2])
            step = int(parts[4])
            return epoch, step
        except (IndexError, ValueError):
            return -1, -1

    checkpoints.sort(key=get_epoch_step)
    return checkpoints[-1]

def chat():
    parser = argparse.ArgumentParser(description="Xenutron Model Chat")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    print("=" * 60)
    print("  Xenutron Model Chat Interface")
    print("=" * 60)
    print("[INFO] Loading configuration and model...")

    # Load config
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"[ERROR] {config_path} not found.")
        return

    config = TrainingConfig.from_json(config_path)
    
    # Device setup
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Load tokenizer
    tokenizer_path = os.path.join(config.tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"[ERROR] Tokenizer not found at {tokenizer_path}.")
        return
    
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = (
        tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size') else
        tokenizer.vocab_size()      if hasattr(tokenizer, 'vocab_size')     else
        config.vocab_size
    )

    # Instantiate model
    model = TransformerLM(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_seq_len=config.max_seq_len,
    )
    
    # Load latest checkpoint
    checkpoint_file = find_latest_checkpoint("./checkpoints")
    if checkpoint_file:
        print(f"[INFO] Loading checkpoint from {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("[WARNING] No checkpoint found. Using unitialized model.")

    model = model.to(device)
    model.eval()

    print("\n[INFO] Chat started. Type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input:
                continue

            # Model generate
            # Adding [BOS] prefix as used in training
            input_ids = tokenizer.encode(user_input).ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

            with torch.inference_mode():
                # We can do manually to debug
                curr_ids = input_ids
                for _ in range(config.max_new_tokens):
                    logits = model(curr_ids)
                    next_token_logits = logits[:, -1, :] / config.temperature
                    
                    # Sanity check
                    if torch.isnan(next_token_logits).any():
                        print("[ERROR] NaNs detected in logits!")
                        break
                    
                    # Top-k
                    if config.top_k > 0:
                        v, _ = torch.topk(next_token_logits, config.top_k)
                        next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

                    # Top-p
                    if config.top_p > 0.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > config.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits.scatter_(1, indices_to_remove.unsqueeze(0), float('-inf'))

                    probs = torch.softmax(next_token_logits, dim=-1)
                    
                    if torch.sum(probs) <= 0:
                        print("[ERROR] Probabilities sum to zero!")
                        break

                    next_token = torch.multinomial(probs, num_samples=1)
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)
                    
                    token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    print(token_text, end='', flush=True)
                    
                    if next_token.item() == tokenizer.token_to_id("[EOS]"):
                        break
                
                print() # New line after generation

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    chat()
