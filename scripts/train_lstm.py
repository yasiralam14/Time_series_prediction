#!/usr/bin/env python3
"""Train and evaluate LSTM models (plain and with symbol embeddings)."""

import argparse
import gc
import sys
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.datasets import EmbeddedDataset
from src.losses import mae
from src.metrics import weighted_r2
from src.models import ELSTM, LSTMModel
from src.preprocessing import eval_prepare, prepare_sequences
from src.trainer import Trainer


def train_plain_lstm(cfg: Config, args):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    print("\n=== Plain LSTM ===")
    train_path = cfg.data.partition_path(args.train_partition)
    train_df = pl.scan_parquet(str(train_path)).collect()
    sequences = prepare_sequences(train_df, cfg.data.sequence_length)
    print(f"  Sequences: {sequences.shape}")

    x = sequences[:, :, 4:83]
    y = sequences[:, -1, 83]
    w = sequences[:, -1, 3]

    dataset = TensorDataset(x, y, w)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = LSTMModel(
        input_size=cfg.data.num_features,
        hidden_size=args.hidden_size,
        output_size=1,
        num_layers=args.num_layers,
    )
    trainer = Trainer(model, device, lr=args.lr, loss_fn=mae)

    print("  Training...")
    trainer.fit(dataloader, epochs=args.epochs)

    del train_df, sequences
    gc.collect()

    # --- Online evaluation ---
    print("  Online evaluation...")
    test_path = cfg.data.partition_path(args.test_partition)
    test_df = pl.scan_parquet(str(test_path)).collect()
    test_data = torch.tensor(eval_prepare(test_df))
    results = trainer.online_evaluate(
        test_data, sequence_size=cfg.data.sequence_length,
        max_batches=cfg.train.online_eval_batches,
    )
    print(f"  Weighted R²: {results['weighted_r2']:.6f}")
    return results["weighted_r2"]


def train_embedded_lstm(cfg: Config, args):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    print("\n=== Embedded LSTM (ELSTM) ===")
    train_path = cfg.data.partition_path(args.train_partition)
    train_df = pl.scan_parquet(str(train_path)).collect()
    sequences = prepare_sequences(train_df, cfg.data.sequence_length)
    print(f"  Sequences: {sequences.shape}")

    dataset = EmbeddedDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = ELSTM(
        vocab_size=cfg.data.num_symbols,
        embedding_dim=args.embedding_dim,
        input_size=cfg.data.num_features,
        hidden_size=args.hidden_size,
        output_size=1,
        num_layers=args.num_layers,
    )
    trainer = Trainer(model, device, lr=args.lr, loss_fn=nn.MSELoss())

    print("  Training...")
    trainer.fit(dataloader, epochs=args.epochs, use_symbols=True)

    del train_df, sequences
    gc.collect()

    # --- Online evaluation ---
    print("  Online evaluation...")
    test_path = cfg.data.partition_path(args.test_partition)
    test_df = pl.scan_parquet(str(test_path)).collect()
    test_data = torch.tensor(eval_prepare(test_df))
    results = trainer.online_evaluate(
        test_data, sequence_size=cfg.data.sequence_length,
        max_batches=cfg.train.online_eval_batches,
        use_symbols=True,
    )
    print(f"  Weighted R²: {results['weighted_r2']:.6f}")
    return results["weighted_r2"]


def main():
    parser = argparse.ArgumentParser(description="LSTM model training")
    parser.add_argument("--model", choices=["plain", "embedded", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=10)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-partition", type=int, default=0)
    parser.add_argument("--test-partition", type=int, default=1)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data.data_root = Path(args.data_root)

    print(f"Device: {cfg.train.device if torch.cuda.is_available() else 'cpu'}")

    scores = {}
    if args.model in ("plain", "both"):
        scores["plain_lstm"] = train_plain_lstm(cfg, args)
    if args.model in ("embedded", "both"):
        scores["embedded_lstm"] = train_embedded_lstm(cfg, args)

    print("\n=== Results ===")
    for name, r2 in scores.items():
        print(f"  {name}: {r2:.6f}")


if __name__ == "__main__":
    main()
