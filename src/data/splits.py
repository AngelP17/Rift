"""Time-aware and random data splitting strategies."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from utils.logging import get_logger

log = get_logger(__name__)


def split_data(
    df: pl.DataFrame,
    strategy: Literal["random", "temporal", "rolling"] = "temporal",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    window_days: int = 7,
) -> dict[str, pl.DataFrame]:
    """Split transaction data into train/val/test sets.

    Returns dict with keys 'train', 'val', 'test' (and potentially
    multiple windows for rolling strategy).
    """
    log.info("splitting_data", strategy=strategy, n=len(df))

    if strategy == "random":
        return _random_split(df, train_frac, val_frac, seed)
    elif strategy == "temporal":
        return _temporal_split(df, train_frac, val_frac)
    elif strategy == "rolling":
        return _rolling_split(df, window_days)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


def _random_split(
    df: pl.DataFrame, train_frac: float, val_frac: float, seed: int
) -> dict[str, pl.DataFrame]:
    n = len(df)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return {
        "train": df[indices[:train_end].tolist()],
        "val": df[indices[train_end:val_end].tolist()],
        "test": df[indices[val_end:].tolist()],
    }


def _temporal_split(
    df: pl.DataFrame, train_frac: float, val_frac: float
) -> dict[str, pl.DataFrame]:
    sorted_df = df.sort("timestamp")
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return {
        "train": sorted_df[:train_end],
        "val": sorted_df[train_end:val_end],
        "test": sorted_df[val_end:],
    }


def _rolling_split(
    df: pl.DataFrame, window_days: int
) -> dict[str, pl.DataFrame]:
    """Create rolling-window evaluation folds.

    Each fold uses all data before the window as training, the window as test.
    """
    sorted_df = df.sort("timestamp")
    ts = sorted_df["timestamp"]
    min_ts = ts.min()
    max_ts = ts.max()

    from datetime import timedelta

    results: dict[str, pl.DataFrame] = {}
    window_start = min_ts + timedelta(days=window_days * 10)
    fold = 0

    while window_start + timedelta(days=window_days) <= max_ts:
        window_end = window_start + timedelta(days=window_days)
        train = sorted_df.filter(pl.col("timestamp") < window_start)
        test = sorted_df.filter(
            (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < window_end)
        )
        if len(train) > 100 and len(test) > 10:
            results[f"train_fold_{fold}"] = train
            results[f"test_fold_{fold}"] = test
            fold += 1
        window_start = window_end

    if fold == 0:
        log.warning("rolling_split_no_folds", msg="Falling back to temporal split")
        return _temporal_split(df, 0.7, 0.15)

    log.info("rolling_split_complete", n_folds=fold)
    return results
