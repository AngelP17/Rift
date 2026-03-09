"""Deterministic seeding for reproducibility."""

import random
from typing import Any

import numpy as np

from rift.utils.config import SEED

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int | None = None) -> int:
    """Set all random seeds for reproducibility."""
    s = seed if seed is not None else SEED
    random.seed(s)
    np.random.seed(s)

    if TORCH_AVAILABLE:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return s


def get_seed() -> int:
    """Get current default seed."""
    return SEED
