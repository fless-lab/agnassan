"""Debug script to check torch import and functionality."""

import sys
import os

def check_torch():
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    
    try:
        import torch
        print("Torch imported successfully")
        print("Torch version:", torch.__version__)
        print("Torch path:", torch.__file__)
        print("CUDA available:", torch.cuda.is_available() if hasattr(torch, 'cuda') else "N/A")
    except ImportError as e:
        print("Failed to import torch:", e)
    except Exception as e:
        print("Error with torch:", e)

if __name__ == "__main__":
    check_torch()