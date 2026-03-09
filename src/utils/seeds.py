"""Deterministic seed control for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np

from utils.config import cfg


def set_global_seeds(seed: int | None = None) -> int:
    seed = seed if seed is not None else cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
