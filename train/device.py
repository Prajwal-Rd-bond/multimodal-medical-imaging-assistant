"""Single source of truth for picking a compute device -- CUDA (Colab/Kaggle
T4/P100), MPS (Apple Silicon), or CPU fallback."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
