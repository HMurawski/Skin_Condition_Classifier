
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(pref: str = "auto"):
    """
    Pick device. If pref == 'cuda' and GPU is available, use it.
    Otherwise use CPU. 'auto' picks GPU if available.
    """
    if pref.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif pref.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    return device

def save_checkpoint(model: torch.nn.Module, path: str):
    """Save model state_dict to a given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved model to {path}")

def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device):
    """Load model weights from a checkpoint file."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    print(f"[INFO] Loaded model from {path}")
    return model
