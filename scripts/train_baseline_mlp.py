#!/usr/bin/env python3
"""Train and evaluate the baseline MLP (no symbol information)."""

import argparse
import sys
from pathlib import Path

import polars as pl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.losses import mae
from src.metrics import weighted_r2
from src.models import MLP
from src.preprocessing import eval_prepare, normalize, prepare_data
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Baseline MLP training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-partition", type=int, default=0)
    parser.add_argument("--test-partition", type=int, default=1)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data.data_root = Path(args.data_root)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Training ---
    print("\n=== Loading training data ===")
    train_path = cfg.data.partition_path(args.train_partition)
    train_df = pl.scan_parquet(str(train_path)).collect()
    data = prepare_data(train_df)

    x = torch.tensor(data[:, 4:83])
    x = normalize(x)
    y = torch.tensor(data[:, 83])
    w = torch.tensor(data[:, 3])
    print(f"  Features: {x.shape}  Targets: {y.shape}  Weights: {w.shape}")

    dataset = TensorDataset(x, y, w)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = MLP(input_size=cfg.data.num_features, hidden_size=args.hidden_size, output_size=1)
    trainer = Trainer(model, device, lr=args.lr, loss_fn=mae)

    print("\n=== Training ===")
    trainer.fit(dataloader, epochs=args.epochs)

    # --- Offline evaluation ---
    print("\n=== Offline evaluation ===")
    test_path = cfg.data.partition_path(args.test_partition)
    test_df = pl.scan_parquet(str(test_path)).collect()
    test_data = eval_prepare(test_df)

    x_test = normalize(torch.tensor(test_data[:, 4:83]))
    y_test = torch.tensor(test_data[:, 83])
    w_test = torch.tensor(test_data[:, 3])

    test_dataset = TensorDataset(x_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    preds = trainer.evaluate(test_dataloader)
    r2 = weighted_r2(preds.squeeze(), y_test, w_test)
    print(f"  Weighted R²: {r2.item():.6f}")

    # --- Online evaluation ---
    print("\n=== Online evaluation ===")
    test_tensor = torch.tensor(test_data)
    results = trainer.online_evaluate(
        test_tensor, sequence_size=cfg.data.sequence_length,
        max_batches=cfg.train.online_eval_batches,
    )
    print(f"  Online Weighted R²: {results['weighted_r2']:.6f}")

    return {"offline_r2": r2.item(), "online_r2": results["weighted_r2"]}


if __name__ == "__main__":
    main()
