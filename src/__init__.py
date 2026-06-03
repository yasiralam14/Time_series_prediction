from src.datasets import (
    EmbeddedDataset,
    EmbeddedMLPDataset,
    FullDataset,
    FullDatasetMLP,
)
from src.losses import huber_loss, mae, weighted_huber_loss, weighted_mae, weighted_mse
from src.metrics import weighted_r2
from src.models import ELSTM, FLSTM, MLP, EmbeddedMLP, LSTMModel

__all__ = [
    "EmbeddedDataset",
    "EmbeddedMLPDataset",
    "FullDataset",
    "FullDatasetMLP",
    "MLP",
    "LSTMModel",
    "EmbeddedMLP",
    "ELSTM",
    "FLSTM",
    "weighted_mse",
    "weighted_mae",
    "huber_loss",
    "weighted_huber_loss",
    "mae",
    "weighted_r2",
]


def __getattr__(name):
    """Lazy-import preprocessing and trainer to avoid hard polars dependency at import time."""
    _lazy = {
        "normalize": "src.preprocessing",
        "prepare_data": "src.preprocessing",
        "prepare_sequences": "src.preprocessing",
        "eval_prepare": "src.preprocessing",
        "expert_data": "src.preprocessing",
        "separate_symbols": "src.preprocessing",
        "batch_by_timestamp": "src.preprocessing",
        "Trainer": "src.trainer",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
