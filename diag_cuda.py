import torch
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Check if a simple tensor operation works on GPU
    try:
        x = torch.randn(1, 1).to("cuda")
        print("Successfully moved tensor to CUDA.")
    except Exception as e:
        print(f"Error moving tensor to CUDA: {e}")
else:
    print("CUDA is NOT available.")
