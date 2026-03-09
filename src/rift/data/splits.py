from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class DatasetSplit:
    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame


def chronological_split(
    frame: pl.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> DatasetSplit:
    frame = frame.sort("timestamp")
    n_rows = frame.height
    train_end = math.floor(n_rows * train_ratio)
    validation_end = math.floor(n_rows * (train_ratio + validation_ratio))
    return DatasetSplit(
        train=frame.slice(0, train_end),
        validation=frame.slice(train_end, validation_end - train_end),
        test=frame.slice(validation_end),
    )


def random_split(
    frame: pl.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 7,
) -> DatasetSplit:
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
    n_rows = shuffled.height
    train_end = math.floor(n_rows * train_ratio)
    validation_end = math.floor(n_rows * (train_ratio + validation_ratio))
    return DatasetSplit(
        train=shuffled.slice(0, train_end),
        validation=shuffled.slice(train_end, validation_end - train_end),
        test=shuffled.slice(validation_end),
    )
