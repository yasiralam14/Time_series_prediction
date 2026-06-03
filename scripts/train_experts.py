#!/usr/bin/env python3
"""Train per-symbol expert models (Mixture of Experts approach)."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.datasets import FullDataset, FullDatasetMLP
from src.losses import weighted_mse
from src.metrics import weighted_r2
from src.models import LSTMModel, MLP
from src.preprocessing import (
    batch_by_timestamp,
    eval_prepare,
    expert_data,
    normalize,
    prepare_data,
    prepare_sequences,
)


def train_mlp_experts(cfg: Config, args):
    print("\n=== MLP Experts (one per symbol) ===")
    train_path = cfg.data.partition_path(args.train_partition)
    train_df = pl.scan_parquet(str(train_path)).collect()
    data = prepare_data(train_df)

    symbols = data[:, 2].astype(int)
    symbol_indices = defaultdict(list)
    for idx, sym in enumerate(symbols):
        symbol_indices[sym].append(idx)

    full_dataset = FullDatasetMLP(torch.tensor(data))
    dataloaders = {
        sym: DataLoader(Subset(full_dataset, idxs), batch_size=args.batch_size, shuffle=True)
        for sym, idxs in symbol_indices.items()
    }

    criterion = nn.MSELoss()
    models = {}
    for sym, dl in dataloaders.items():
        model = MLP(cfg.data.num_features, args.hidden_size, 1)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for _ in range(args.epochs):
            for features, targets, weights in dl:
                optimizer.zero_grad()
                preds = model(features[:, 3:])
                loss = criterion(preds.squeeze(), targets.squeeze())
                loss.backward()
                optimizer.step()
        models[sym] = model

    print(f"  Trained {len(models)} expert MLP models")
    return models


def train_lstm_experts(cfg: Config, args):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    print("\n=== LSTM Experts (one per symbol) ===")
    train_path = cfg.data.partition_path(args.train_partition)
    train_df = pl.scan_parquet(str(train_path)).collect()
    symbol_tensors = expert_data(train_df)

    from src.preprocessing import create_sequences

    data_repo = {}
    for sym_data in symbol_tensors:
        sym_id = int(sym_data[0, 2])
        seqs = create_sequences(sym_data, cfg.data.sequence_length)
        if len(seqs) > 0:
            data_repo[sym_id] = seqs

    dataloaders = {
        sym: DataLoader(FullDataset(seqs), batch_size=args.batch_size, shuffle=True)
        for sym, seqs in data_repo.items()
    }

    criterion = nn.MSELoss()
    models = {}
    for sym, dl in dataloaders.items():
        model = LSTMModel(
            input_size=cfg.data.num_features,
            hidden_size=args.hidden_size,
            output_size=1,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for _ in range(args.epochs):
            for features, targets, weights, symbols in dl:
                features = features.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                preds = model(features)
                loss = criterion(preds.squeeze(), targets.squeeze())
                loss.backward()
                optimizer.step()
        models[sym] = model

    print(f"  Trained {len(models)} expert LSTM models")
    return models


def online_evaluate_experts(models, cfg: Config, args, device):
    """Run online evaluation for per-symbol experts with continual updates."""
    print("  Online evaluation...")
    test_path = cfg.data.partition_path(args.test_partition)
    test_df = pl.scan_parquet(str(test_path)).collect()
    data = torch.tensor(eval_prepare(test_df))
    data[:, 4:83] = normalize(data[:, 4:83])
    batches = batch_by_timestamp(data)

    optimizers = {sym: optim.Adam(m.parameters(), lr=args.lr) for sym, m in models.items()}

    seq_size = cfg.data.sequence_length
    limit = min(args.online_batches, len(batches) - seq_size + 1)

    predictions, targets, weights = [], [], []
    for idx in range(limit):
        seq = batches[idx : idx + seq_size]
        seq_tensor = torch.stack(seq).permute(1, 0, 2)

        for i, sample in enumerate(seq_tensor):
            sample = sample.unsqueeze(0)
            feat = sample[:, :, 4:83].to(device)
            y = sample[:, -1, 83]
            w = sample[:, -1, 3]
            sym = int(sample[:, 0, 2])

            if sym not in models:
                continue

            optimizers[sym].zero_grad()
            out = models[sym](feat)
            loss = weighted_mse(out.squeeze(), y.float().to(device), w.float().to(device))
            loss.backward()
            optimizers[sym].step()

            predictions.append(out.squeeze().detach().cpu())
            targets.append(y.squeeze())
            weights.append(w.squeeze())

    preds = torch.stack(predictions)
    tgts = torch.stack(targets)
    wts = torch.stack(weights)
    r2 = weighted_r2(preds, tgts, wts)
    print(f"  Weighted R²: {r2.item():.6f}")
    return r2.item()


def main():
    parser = argparse.ArgumentParser(description="Mixture of Experts training")
    parser.add_argument("--model", choices=["mlp", "lstm", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--online-batches", type=int, default=1600)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-partition", type=int, default=0)
    parser.add_argument("--test-partition", type=int, default=1)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data.data_root = Path(args.data_root)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scores = {}
    if args.model in ("mlp", "both"):
        mlp_models = train_mlp_experts(cfg, args)
        scores["mlp_experts"] = online_evaluate_experts(mlp_models, cfg, args, device)

    if args.model in ("lstm", "both"):
        lstm_models = train_lstm_experts(cfg, args)
        scores["lstm_experts"] = online_evaluate_experts(lstm_models, cfg, args, device)

    print("\n=== Results ===")
    for name, r2 in scores.items():
        print(f"  {name}: {r2:.6f}")


if __name__ == "__main__":
    main()
