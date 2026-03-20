import torch
import torch.nn as nn
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    try:
        print("Testing basic tensor operation...")
        x = torch.randn(1, 1).to(device)
        print("Success.")
        
        print("Testing Embedding layer...")
        emb = nn.Embedding(10, 10).to(device)
        idx = torch.tensor([1, 2]).to(device)
        res = emb(idx)
        print("Success.")
        
        print("Testing Linear layer...")
        lin = nn.Linear(10, 10).to(device)
        res = lin(res)
        print("Success.")

        print("Testing AMP (autocast)...")
        with torch.cuda.amp.autocast():
             res = lin(res)
        print("Success.")
        
    except Exception as e:
        print(f"\nFAILURE during GPU test: {e}")
        import traceback
        traceback.print_exc()
else:
    print("CUDA is NOT available.")
