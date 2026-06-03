# Time Series Prediction for Market Data Forecasting

A comparative study of deep learning and gradient boosting approaches for real-time financial time series prediction, built for the [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) competition.

The project evaluates **seven model architectures** under both standard (offline) and **online learning** evaluation protocols, measuring performance with a competition-aligned **weighted R²** metric.

## Models

| Model | Type | Symbol Info | Temporal |
|---|---|---|---|
| Baseline MLP | Feed-forward | None | No |
| Embedded MLP | Feed-forward | Learned embeddings | No |
| Per-Symbol MLP Experts | Mixture of Experts | Separate model per symbol | No |
| Plain LSTM | Recurrent | None | Sliding window |
| Embedded LSTM (ELSTM) | Recurrent | Learned embeddings | Sliding window |
| Per-Symbol LSTM Experts | Mixture of Experts | Separate model per symbol | Sliding window |
| XGBoost | Gradient Boosting | Categorical feature | No |

## Key Design Decisions

- **Online evaluation**: Models are updated continuously on test data before predicting the next batch, simulating the competition's real-time inference API.
- **Symbol embeddings**: Financial instruments are encoded as learned dense vectors rather than one-hot, allowing the model to discover latent relationships between symbols.
- **Mixture of Experts**: Independent models per symbol capture instrument-specific dynamics that a shared model might average over.
- **Weighted R²**: All evaluation uses sample weights provided by the competition, reflecting the economic significance of each prediction.

## Project Structure

```
.
├── config.py                          # Centralized hyperparameters and paths
├── src/
│   ├── __init__.py
│   ├── datasets.py                    # PyTorch Dataset classes
│   ├── losses.py                      # Weighted MSE, MAE, Huber losses
│   ├── metrics.py                     # Weighted R² implementation
│   ├── models.py                      # MLP, LSTM, ELSTM, FLSTM architectures
│   ├── preprocessing.py               # Data loading, normalization, sequencing
│   └── trainer.py                     # Training/evaluation engine
├── scripts/
│   ├── train_baseline_mlp.py          # Baseline MLP experiment
│   ├── train_embedded_mlp.py          # Symbol-embedded MLP experiment
│   ├── train_lstm.py                  # Plain and embedded LSTM experiments
│   ├── train_experts.py               # Per-symbol expert models
│   ├── train_xgboost.py              # XGBoost with online updates
│   └── run_all_experiments.py         # Run all experiments end-to-end
├── notebooks/
│   ├── CS559_Project.ipynb            # Original exploration notebook
│   └── CS559__Project.pdf             # Project report
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

The training data is expected in Parquet format partitioned as:
```
<data_root>/partition_id=<N>/part-0.parquet
```

## Usage

### Run a single experiment

```bash
# Baseline MLP
python scripts/train_baseline_mlp.py --data-root /path/to/train.parquet --epochs 10

# LSTM models (plain + embedded)
python scripts/train_lstm.py --data-root /path/to/train.parquet --model both

# XGBoost
python scripts/train_xgboost.py --data-root /path/to/train.parquet --num-rounds 100

# Mixture of Experts
python scripts/train_experts.py --data-root /path/to/train.parquet --model lstm
```

### Run all experiments

```bash
python scripts/run_all_experiments.py --data-root /path/to/train.parquet
```

### Configuration

All hyperparameters have sensible defaults in `config.py` and can be overridden via CLI arguments:

```bash
python scripts/train_lstm.py \
    --epochs 20 \
    --batch-size 512 \
    --lr 0.0005 \
    --hidden-size 512 \
    --num-layers 2
```

## Evaluation Protocol

Each model is evaluated in two modes:

1. **Offline**: Train on partition 0, evaluate on partition 1 with frozen weights.
2. **Online**: Train on partition 0, then process partition 1 sequentially — predict each time step, then update model weights before the next step. This mirrors the competition's real-time API where models can adapt to distribution shift.

## Technical Details

- **Normalization**: Zero-mean, unit-variance standardization applied per feature column.
- **Sequence construction**: For LSTM models, data is sorted by `(symbol_id, date_id, time_id)` and sliced into fixed-length sliding windows.
- **Loss functions**: Modular loss library supporting MSE, MAE, Huber, and their weighted variants.
- **Missing values**: Filled with zeros (following competition baseline practice).
