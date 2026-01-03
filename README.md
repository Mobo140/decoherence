# Quantum Decoherence Time Prediction

End-to-end baseline project for predicting quantum decoherence times from partial time series of observables using Transformer models.

## 🚨 NEW: Early Warning System

**Версия 2.0** добавляет систему раннего предупреждения о классикализации с multi-task обучением и базовым контролем!

👉 **См. [README_EARLY_WARNING.md](README_EARLY_WARNING.md)** для полного руководства по новой системе.

**Основные возможности:**
- 🔴 Risk Classification: Предсказание вероятности декогеренции в ближайшем окне
- 🔮 Forecasting: Прогнозирование будущих наблюдаемых
- ⏱️ Time Regression: Оценка оставшегося времени
- 🎯 Control Baseline: Демонстрация практической пользы через triggered control

**Быстрый старт:**
```bash
# Запустить полный pipeline для эксперимента E5
bash run_early_warning_pipeline.sh E5
```

---

## Project Overview

This project implements a machine learning pipeline to predict decoherence times of quantum systems from partial observations. The system generates quantum trajectories by solving the Lindblad master equation, then trains various models (MLP, RNN, Transformer) to predict when decoherence occurs.

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Quick Start

1. **Generate dataset for experiment E1**:
   ```bash
   python3 -m src.data.generate_dataset --config configs/experiments/E1.yaml
   ```
   This will create a dataset with train/val/test splits and save split indices to `data/splits/`.

2. **Train and compare multiple models with cross-validation**:
   ```bash
   python3 -m src.train.run_suite --config configs/experiments/E1.yaml \
       --models mlp gru lstm transformer --method cv --n-folds 5
   ```
   Результаты сохраняются в `experiments/`.

3. **Generate comparison report**:
   ```bash
   # Для новой структуры (experiments/exp_*/):
   python3 -m src.reports.make_report --run-dir experiments/exp_YYYYMMDD_HHMMSS \
       --output-dir reports
   
   # Или для старой структуры (experiments/ напрямую):
   python3 -m src.reports.make_report --run-dir experiments \
       --output-dir reports
   ```

4. **Train a single model (hold-out validation)**:
   ```bash
   python3 -m src.train.train --config configs/experiments/E1.yaml --model transformer
   ```

5. **Evaluate the trained model**:
   ```bash
   python3 -m src.train.eval --config configs/experiments/E1.yaml --checkpoint checkpoints/transformer/checkpoint_XX
   ```

## Usage

### 1. Generate Dataset

Generate a dataset for a specific experiment:

```bash
python3 -m src.data.generate_dataset --config configs/experiments/E1.yaml
```

This will:
- Generate trajectories with train/val/test split (default: 80/10/10)
- Save dataset to `data/dataset.npz` (or path specified in config)
- Save split indices to `data/splits/dataset_splits.json`
- Include dataset version and metadata

Available experiments:
- **E1**: Single-qubit, constant γ, σz only
- **E2**: Single-qubit, constant γ, σx,y,z
- **E3**: Single-qubit, time-dependent γ(t), σx,y,z
- **E4**: Two-qubit, simple interaction, multiple channels

### 2. Train Model

Train a model on the generated dataset:

```bash
python3 -m src.train.train --config configs/experiments/E1.yaml --model lstm
```

Available models: `mlp`, `gru`, `lstm`, `transformer`, `two_stage`

The training script will:
- Load the dataset
- Train the model for the specified number of epochs
- Save checkpoints when validation loss improves
- Log training metrics

### 3. Model Comparison Suite

Run comparison of multiple models with cross-validation:

```bash
python3 -m src.train.run_suite --config configs/experiments/E1.yaml \
    --models mlp gru lstm transformer \
    --method cv --n-folds 5 \
    --output-dir experiments
```

Options:
- `--method`: `holdout`, `cv`, or `both`
- `--n-folds`: Number of CV folds (default: 5)
- `--models`: List of models to compare

This will:
- Train each model using K-fold cross-validation
- Save results for each fold
- Aggregate metrics across folds (mean ± std)
- Save experiment config and summary

### 4. Generate Comparison Report

Generate final report with comparison table:

```bash
# Для новой структуры (experiments/exp_*/):
python3 -m src.reports.make_report --run-dir experiments/exp_YYYYMMDD_HHMMSS \
    --output-dir reports --primary-metric mae

# Или для старой структуры (experiments/ напрямую):
python3 -m src.reports.make_report --run-dir experiments \
    --output-dir reports --primary-metric mae
```

This will generate:
- `metrics_summary.csv`: CSV table with all metrics
- `metrics_summary.tex`: LaTeX table ready for paper
- `report.md`: Markdown report with summary

### 5. Evaluate Model

Evaluate a trained model (for final test evaluation):

```bash
python3 -m src.train.eval --config configs/experiments/E1.yaml --checkpoint checkpoints/transformer/checkpoint_19
```

This will:
- Compute metrics on test set (MAE, RMSE, MAPE, R², coverage)
- Generate visualization plots (scatter, error histogram, error vs gamma)
- Save metrics and plots to `figures/`

## Running Tests

```bash
python3 tests/test_decoherence_time.py
```

## Interpreting Metrics

- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (penalizes large errors more)
- **MAPE**: Mean Absolute Percentage Error (%)
- **R²**: Coefficient of determination (1.0 = perfect, 0.0 = no better than mean)
- **Coverage**: Fraction of predictions within 5% of T_max from true value

## Project Structure

```
src/
  physics/         # Quantum simulation modules
  data/            # Dataset generation, splits, transforms
  models/          # ML model architectures
  train/           # Training, evaluation, cross-validation
  reports/         # Report generation (tables, summaries)
  utils/           # Utilities (config, logging, plotting)
configs/           # Configuration files
data/              # Generated datasets
  splits/          # Saved split indices
runs/              # Training runs and experiments
  experiments/     # Model comparison experiments
  <model>/         # Model-specific runs
figures/           # Evaluation plots
reports/           # Generated comparison reports
```

## Key Features

- **Physical Systems**: Single-qubit and two-qubit systems with Lindblad dissipation
- **Models**: MLP, GRU/LSTM, Transformer, Two-stage Transformer
- **Decoherence Criteria**: L1 coherence and purity-based thresholds
- **Reproducibility**: Seed control, saved splits, and configurable experiments
- **Validation**: Hold-out validation and K-fold cross-validation
- **Model Comparison**: Unified interface for comparing multiple models
- **Reporting**: Automatic generation of comparison tables (CSV + LaTeX)

## Validation Protocol

The system follows a strict validation protocol:

1. **Train/Val/Test Split**: Fixed split with saved indices (default: 80/10/10)
2. **Cross-Validation**: K-fold CV for model selection (metrics computed on validation folds)
3. **Test Set**: Used only once for final evaluation of the selected model
4. **No Data Leakage**: Normalization fitted only on training data within each fold/split

## Reproducibility

- All splits are saved with seed and metadata
- Config files are saved with each experiment
- Checkpoints include model state and training history
- Results are aggregated and saved in JSON format
