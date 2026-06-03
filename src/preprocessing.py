from __future__ import annotations

from typing import List

import numpy as np
import polars as pl
import torch

RESPONDERS_TO_DROP = [
    "responder_0",
    "responder_1",
    "responder_2",
    "responder_3",
    "responder_4",
    "responder_5",
    "responder_7",
    "responder_8",
]

FEATURE_COLS = slice(4, 83)


def normalize(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std == 0] = 1
    return (features - mean) / std


def prepare_data(data: pl.LazyFrame | pl.DataFrame) -> np.ndarray:
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    data = data.drop(RESPONDERS_TO_DROP).fill_null(0)
    return data.to_numpy()


def eval_prepare(data: pl.LazyFrame | pl.DataFrame) -> np.ndarray:
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    data = data.sort("date_id", "time_id")
    data = data.drop(RESPONDERS_TO_DROP).fill_null(0)
    return data.to_numpy()


def separate_symbols(data: torch.Tensor) -> List[torch.Tensor]:
    """Group rows by symbol_id (column index 2). Data must be sorted by symbol_id."""
    batches: List[torch.Tensor] = []
    current_batch: List[torch.Tensor] = []
    current_symbol: int | None = None

    for row in data:
        symbol = int(row[2])
        if current_symbol is None or symbol == current_symbol:
            current_batch.append(row)
            current_symbol = symbol
        else:
            batches.append(torch.stack(current_batch))
            current_batch = [row]
            current_symbol = symbol

    if current_batch:
        batches.append(torch.stack(current_batch))
    return batches


def batch_by_timestamp(data: torch.Tensor) -> List[torch.Tensor]:
    """Group rows by (date_id, time_id) pairs. Data must be sorted by date then time."""
    batches: List[torch.Tensor] = []
    current_batch: List[torch.Tensor] = []
    current_key: tuple | None = None

    for row in data:
        key = (int(row[0].item()), int(row[1].item()))
        if current_key is None or key == current_key:
            current_batch.append(row)
            current_key = key
        else:
            batches.append(torch.stack(current_batch))
            current_batch = [row]
            current_key = key

    if current_batch:
        batches.append(torch.stack(current_batch))
    return batches


def expert_data(data: pl.DataFrame) -> List[torch.Tensor]:
    """Prepare per-symbol tensor batches for mixture-of-experts training."""
    data = data.sort("symbol_id")
    data = data.drop(RESPONDERS_TO_DROP).fill_null(0)
    tensor = torch.tensor(data.to_numpy())
    return separate_symbols(tensor)


def create_sequences(data: torch.Tensor, sequence_len: int) -> torch.Tensor:
    """Create sliding-window sequences grouped by the first column (symbol_id)."""
    class_sequences: dict[int, list] = {}
    for row in data:
        label = int(row[0])
        class_sequences.setdefault(label, []).append(row)

    sequences = []
    for rows in class_sequences.values():
        arr = np.array(rows)
        for i in range(len(arr) - sequence_len + 1):
            sequences.append(arr[i : i + sequence_len])

    return torch.tensor(np.array(sequences), dtype=torch.float32)


def prepare_sequences(
    data: pl.LazyFrame | pl.DataFrame, sequence_len: int
) -> torch.Tensor:
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    data = data.sort("symbol_id", "date_id", "time_id")
    data = data.drop(RESPONDERS_TO_DROP).fill_null(0)
    tensor = torch.tensor(data.to_numpy())
    tensor[:, FEATURE_COLS] = normalize(tensor[:, FEATURE_COLS])
    return create_sequences(tensor, sequence_len)
