"""Central configuration for all experiments."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_root: Path = Path("/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet")
    num_partitions: int = 10
    train_partition: int = 0
    test_partition: int = 1
    num_features: int = 79
    num_symbols: int = 39
    sequence_length: int = 5

    def partition_path(self, partition_id: int) -> Path:
        return self.data_root / f"partition_id={partition_id}" / "part-0.parquet"


@dataclass
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 10
    hidden_size: int = 256
    embedding_dim: int = 10
    num_lstm_layers: int = 1
    online_eval_batches: int = 1600
    device: str = "cuda:0"
    gpu_ids: list[int] = field(default_factory=lambda: [0])


@dataclass
class XGBoostConfig:
    max_depth: int = 6
    learning_rate: float = 0.01
    num_rounds: int = 100
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
