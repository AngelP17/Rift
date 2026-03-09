"""Time-aware data splitting for fraud detection."""

from enum import Enum
from typing import Tuple

import numpy as np
import polars as pl


class SplitStrategy(str, Enum):
    RANDOM = "random"
    CHRONOLOGICAL = "chronological"
    ROLLING = "rolling"


def train_val_test_split(
    df: pl.DataFrame,
    strategy: SplitStrategy = SplitStrategy.CHRONOLOGICAL,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    time_col: str = "timestamp",
    seed: int | None = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split data into train/val/test sets.

    - random: Standard random split (leaks future info - for baseline comparison)
    - chronological: Time-based split (no leakage)
    - rolling: Uses rolling windows (for temporal evaluation)
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.0")

    n = len(df)

    if strategy == SplitStrategy.RANDOM:
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(n)
        t1, t2 = int(n * train_frac), int(n * (train_frac + val_frac))
        train_idx = set(perm[:t1])
        val_idx = set(perm[t1:t2])
        test_idx = set(perm[t2:])
        idx_col = pl.int_range(0, n).alias("_split_idx")
        train_df = df.with_columns(idx_col).filter(pl.col("_split_idx").is_in(train_idx)).drop("_split_idx")
        val_df = df.with_columns(idx_col).filter(pl.col("_split_idx").is_in(val_idx)).drop("_split_idx")
        test_df = df.with_columns(idx_col).filter(pl.col("_split_idx").is_in(test_idx)).drop("_split_idx")

    elif strategy == SplitStrategy.CHRONOLOGICAL:
        df_sorted = df.sort(time_col)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        train_df = df_sorted.head(t1)
        val_df = df_sorted.slice(t1, t2 - t1)
        test_df = df_sorted.tail(n - t2)

    else:
        # Rolling: treat as chronological for simplicity; full rolling needs window param
        df_sorted = df.sort(time_col)
        t1 = int(n * train_frac)
        t2 = int(n * (train_frac + val_frac))
        train_df = df_sorted.head(t1)
        val_df = df_sorted.slice(t1, t2 - t1)
        test_df = df_sorted.tail(n - t2)

    return train_df, val_df, test_df


def get_split_indices(
    n: int,
    strategy: SplitStrategy,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    sorted_indices: np.ndarray | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index arrays for train/val/test splits."""
    if sorted_indices is None:
        sorted_indices = np.arange(n)

    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))

    if strategy == SplitStrategy.RANDOM:
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(n)
        train_idx = sorted_indices[perm[:t1]]
        val_idx = sorted_indices[perm[t1:t2]]
        test_idx = sorted_indices[perm[t2:]]
    else:
        train_idx = sorted_indices[:t1]
        val_idx = sorted_indices[t1:t2]
        test_idx = sorted_indices[t2:]

    return train_idx, val_idx, test_idx
