import torch
import sys
import os

print(f"System Path Executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Torch file location: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA NOT available.")

print("\nEnvironment Variables:")
for k, v in os.environ.items():
    if "PYTHON" in k or "PATH" == k:
        print(f"{k}: {v[:100]}...")
