#!/usr/bin/env python3
"""Run all experiments and produce a comparative results summary."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

EXPERIMENTS = [
    {
        "name": "Baseline MLP",
        "script": "train_baseline_mlp.py",
        "args": [],
    },
    {
        "name": "Embedded MLP",
        "script": "train_embedded_mlp.py",
        "args": [],
    },
    {
        "name": "Plain LSTM",
        "script": "train_lstm.py",
        "args": ["--model", "plain"],
    },
    {
        "name": "Embedded LSTM",
        "script": "train_lstm.py",
        "args": ["--model", "embedded"],
    },
    {
        "name": "MLP Experts",
        "script": "train_experts.py",
        "args": ["--model", "mlp"],
    },
    {
        "name": "LSTM Experts",
        "script": "train_experts.py",
        "args": ["--model", "lstm"],
    },
    {
        "name": "XGBoost",
        "script": "train_xgboost.py",
        "args": [],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Run specific experiments by name (default: all)",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    selected = args.experiments
    results = {}

    for exp in EXPERIMENTS:
        if selected and exp["name"] not in selected:
            continue

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {exp['name']}")
        print(f"{'='*60}")

        cmd = [sys.executable, str(SCRIPTS_DIR / exp["script"])] + exp["args"]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]

        result = subprocess.run(cmd, capture_output=False)
        results[exp["name"]] = "OK" if result.returncode == 0 else "FAILED"

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        indicator = "[PASS]" if status == "OK" else "[FAIL]"
        print(f"  {indicator} {name}")


if __name__ == "__main__":
    main()
