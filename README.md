# Time Series Prediction for Market Data Forecasting

A comparative study of deep learning and gradient boosting approaches for real-time financial time series prediction, built for the [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting) competition.

The project evaluates **seven model architectures** under both standard (offline) and **online learning** evaluation protocols, measuring performance with a competition-aligned **weighted R²** metric. It includes comprehensive exploratory data analysis, feature profiling, and model comparison visualizations.

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

## Key Findings

| Observation | Detail |
|---|---|
| Online learning is essential | Every model shows significant improvement with continual updates, confirming non-stationarity. |
| Symbol information helps | Embeddings and per-symbol experts consistently outperform symbol-agnostic counterparts. |
| XGBoost is strongest | Gradient boosting achieves the best R² in both offline and online settings. |
| Simple experts beat shared LSTMs | Instrument-specific simple models outperform a shared complex model. |

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
│   ├── trainer.py                     # Training/evaluation engine
│   ├── visualization.py               # Plotting utilities (EDA, model comparison)
│   └── feature_analysis.py            # Correlation, missing data, feature profiling
├── scripts/
│   ├── train_baseline_mlp.py          # Baseline MLP experiment
│   ├── train_embedded_mlp.py          # Symbol-embedded MLP experiment
│   ├── train_lstm.py                  # Plain and embedded LSTM experiments
│   ├── train_experts.py               # Per-symbol expert models
│   ├── train_xgboost.py              # XGBoost with online updates
│   ├── run_all_experiments.py         # Run all experiments end-to-end
│   ├── eda.py                         # Full EDA pipeline with saved outputs
│   └── analyze_results.py             # Model comparison charts and reports
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # Guided EDA walkthrough
│   ├── 02_model_comparison.ipynb            # Model results analysis
│   ├── CS559_Project.ipynb                  # Original exploration notebook
│   └── CS559__Project.pdf                   # Project report
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

### Exploratory Data Analysis

```bash
# Generate all EDA plots and statistics
python scripts/eda.py --data-root /path/to/train.parquet --output-dir outputs/eda

# For faster iteration on large data
python scripts/eda.py --data-root /path/to/train.parquet --sample-frac 0.1
```

Or open the interactive notebook:
```
notebooks/01_exploratory_data_analysis.ipynb
```

### Train Models

```bash
# Baseline MLP
python scripts/train_baseline_mlp.py --data-root /path/to/train.parquet --epochs 10

# LSTM models (plain + embedded)
python scripts/train_lstm.py --data-root /path/to/train.parquet --model both

# XGBoost
python scripts/train_xgboost.py --data-root /path/to/train.parquet --num-rounds 100

# Mixture of Experts
python scripts/train_experts.py --data-root /path/to/train.parquet --model lstm

# Run all experiments
python scripts/run_all_experiments.py --data-root /path/to/train.parquet
```

### Analyze Results

```bash
# With actual experiment results
python scripts/analyze_results.py --results-file outputs/results.json

# Demo with example values
python scripts/analyze_results.py --use-example --output-dir outputs/analysis
```

Or open the comparison notebook:
```
notebooks/02_model_comparison.ipynb
```

### Configuration

All hyperparameters have sensible defaults in `config.py` and can be overridden via CLI:

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

## Analysis Pipeline

The EDA pipeline produces:

| Output | Description |
|---|---|
| `summary_statistics.csv` | Extended descriptive stats (mean, std, skew, kurtosis, percentiles) |
| `missing_data_profile.csv` | Per-column missing rates and data types |
| `target_distribution.png` | Weighted and unweighted target histograms |
| `correlation_heatmap.png` | Full 79x79 feature correlation matrix |
| `target_correlations_top25.png` | Top features correlated with the target |
| `highly_correlated_pairs.csv` | Feature pairs with \|r\| >= 0.90 |
| `temporal_patterns.png` | Daily mean, volatility, weight, and sample count trends |
| `symbol_analysis.png` | Per-symbol target and weight statistics |

## Technical Details

- **Normalization**: Zero-mean, unit-variance standardization applied per feature column.
- **Sequence construction**: For LSTM models, data is sorted by `(symbol_id, date_id, time_id)` and sliced into fixed-length sliding windows.
- **Loss functions**: Modular loss library supporting MSE, MAE, Huber, and their weighted variants.
- **Missing values**: Filled with zeros (following competition baseline practice).
- **Visualization**: All plots use a consistent seaborn style with publication-quality DPI.
