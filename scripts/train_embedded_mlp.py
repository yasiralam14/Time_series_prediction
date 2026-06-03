#!/usr/bin/env python3
"""Train and evaluate the MLP with symbol embeddings."""

import argparse
import sys
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.datasets import EmbeddedMLPDataset
from src.losses import mae
from src.metrics import weighted_r2
from src.models import EmbeddedMLP
from src.preprocessing import normalize, prepare_data


def main():
    parser = argparse.ArgumentParser(description="Embedded MLP training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=10)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-partition", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data.data_root = Path(args.data_root)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load and prepare data ---
    print("\n=== Loading training data ===")
    train_path = cfg.data.partition_path(args.train_partition)
    train_df = pl.scan_parquet(str(train_path)).collect()
    data = prepare_data(train_df)
    data = torch.tensor(data)
    data[:, 4:83] = normalize(data[:, 4:83])

    dataset = EmbeddedMLPDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Build model ---
    model = EmbeddedMLP(
        num_features=cfg.data.num_features,
        num_symbols=cfg.data.num_symbols,
        embedding_dim=args.embedding_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Train ---
    print("\n=== Training ===")
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for features, targets, weights, symbols in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            symbols = symbols.to(device)

            optimizer.zero_grad()
            preds = model(features, symbols)
            loss = mae(preds.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg = running_loss / len(dataloader)
        print(f"  Epoch [{epoch + 1}/{args.epochs}]  Loss: {avg:.6f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
