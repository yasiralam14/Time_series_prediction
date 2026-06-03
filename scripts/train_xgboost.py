#!/usr/bin/env python3
"""Train and evaluate XGBoost models with offline and online evaluation."""

import argparse
import sys
from pathlib import Path

import polars as pl
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.metrics import weighted_r2
from src.preprocessing import batch_by_timestamp, eval_prepare

FEATURE_COLS = [f"feature_{i:02d}" for i in range(79)]


def standardize_features(df: pl.DataFrame) -> pl.DataFrame:
    epsilon = 1e-6
    cols = df.columns[4:83]
    df = df.fill_null(0)
    std_devs = {col: df[col].std() for col in cols}
    return df.with_columns(
        [
            ((pl.col(col) - pl.col(col).mean()) / (std_devs[col] + epsilon)).alias(col)
            for col in cols
            if std_devs[col] > epsilon
        ]
    )


COLS_TO_DROP = [
    "date_id",
    "time_id",
    "weight",
    "responder_0",
    "responder_1",
    "responder_2",
    "responder_3",
    "responder_4",
    "responder_5",
    "responder_7",
    "responder_8",
]


def main():
    parser = argparse.ArgumentParser(description="XGBoost training")
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--online-batches", type=int, default=10000)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-partition", type=int, default=0)
    parser.add_argument("--test-partition", type=int, default=1)
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data.data_root = Path(args.data_root)

    # --- Training ---
    print("=== Loading training data ===")
    train_path = cfg.data.partition_path(args.train_partition)
    train_df = standardize_features(pl.scan_parquet(str(train_path)).collect())

    X_train = train_df.drop(COLS_TO_DROP)
    y_train = X_train["responder_6"]
    X_train = X_train.drop("responder_6")

    feature_types = ["c" if col == "symbol_id" else "float" for col in X_train.columns]
    dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_types=feature_types)

    params = {
        "objective": cfg.xgboost.objective,
        "max_depth": args.max_depth,
        "eta": args.lr,
        "eval_metric": cfg.xgboost.eval_metric,
    }

    print("\n=== Training ===")
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=args.num_rounds)

    # --- Offline evaluation ---
    print("\n=== Offline evaluation ===")
    test_path = cfg.data.partition_path(args.test_partition)
    test_df = standardize_features(pl.scan_parquet(str(test_path)).collect())

    X_test = test_df.drop(COLS_TO_DROP)
    y_test_pl = X_test["responder_6"]
    X_test = X_test.drop("responder_6")
    w_test_pl = test_df["weight"]

    dtest = xgb.DMatrix(data=X_test)
    preds = torch.tensor(bst.predict(dtest))
    y_test = torch.tensor(y_test_pl.to_numpy())
    w_test = torch.tensor(w_test_pl.fill_null(0).to_numpy())

    r2_offline = weighted_r2(preds, y_test, w_test)
    print(f"  Weighted R²: {r2_offline.item():.6f}")

    # --- Online evaluation ---
    print("\n=== Online evaluation ===")
    test_data = eval_prepare(test_df)
    data_tensor = torch.tensor(test_data)
    batches = batch_by_timestamp(data_tensor)

    predictions, targets, weights = [], [], []
    for idx, batch in enumerate(batches[: args.online_batches]):
        w = batch[:, 3]
        features_with_sym = torch.cat([batch[:, :3], batch[:, 4:]], dim=1)
        features_with_sym[:, 2] = features_with_sym[:, 2].to(torch.int)

        X_batch = features_with_sym[:, 2:82]
        y_batch = features_with_sym[:, 82]

        dm = xgb.DMatrix(data=X_batch, label=y_batch)
        out = bst.predict(dm)
        predictions.append(torch.tensor(out).squeeze())
        bst.update(dm, args.num_rounds)
        targets.append(y_batch.squeeze())
        weights.append(w.squeeze())

    preds_online = torch.cat(predictions)
    tgts = torch.cat(targets)
    wts = torch.cat(weights)
    r2_online = weighted_r2(preds_online, tgts, wts)
    print(f"  Online Weighted R²: {r2_online.item():.6f}")

    print("\n=== Results ===")
    print(f"  Offline: {r2_offline.item():.6f}")
    print(f"  Online:  {r2_online.item():.6f}")


if __name__ == "__main__":
    main()
