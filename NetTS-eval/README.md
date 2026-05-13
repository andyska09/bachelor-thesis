# NetTS-eval

Evaluation harness for **NetTS** — five diagnostic benchmarks for network traffic time series forecasting, built on the publicly available [CESNET-TimeSeries24](https://doi.org/10.1038/s41597-025-04603-x) dataset. Developed as part of a bachelor's thesis at CTU FIT by Ondřej Skácel.

> [Dataset (CESNET-TimeSeries24)](https://doi.org/10.1038/s41597-025-04603-x) · [Benchmark construction (NetTS-experiments)](../NetTS-experiments/) · [Croissant metadata](benchmarks/croissant.json)

---

## Motivation

Network traffic forecasting lacks a standardized evaluation framework. Prior work relies on private datasets, incompatible entity subsets, or undifferentiated dataset characteristics — making cross-method comparison unreliable. NetTS addresses this by isolating five distinct behavioral properties of network traffic into separate benchmarks. When a model fails, the cause can be traced to a concrete behavior rather than a generic dataset artifact.

---

## NetTS

NetTS draws series from CESNET-TimeSeries24, a large-scale volumetric dataset collected from the CESNET3 ISP backbone network. The target variable is `n_bytes` (total bytes per interval). Missing intervals are filled with zero (no traffic, not absent data).

**Temporal coverage:** ~280 days per resolution. All series share a common 60/20/20 chronological train/val/test split.

| Resolution | Total timesteps | Train | Validation | Test |
|------------|:--------------:|------:|----------:|-----:|
| Hourly | 6,718 | 4,030 h | 1,344 h | 1,344 h |
| 10-minute | 40,298 | 24,178 steps | 8,060 steps | 8,060 steps |

### Benchmark Types

Each benchmark isolates one characteristic behaviour. Series are selected via a two-stage pipeline: threshold-based scoring (Stage 1) followed by random sampling from qualifiers (Stage 2). Pre-computed score CSVs are available in [NetTS-experiments](../NetTS-experiments/) to skip the expensive Stage 1.

| Benchmark | Behaviour | Aggregation levels | `cesnet-tszoo` name |
|-----------|-----------|:-----------------:|---------------------|
| `seasonal` | Strong daily/weekly periodicity (STL seasonal strength ≥ 0.7) | Institution · Subnet · IP | `NetTS_seasonal` |
| `drift` | Sustained distribution shift visible in the test period | Institution · Subnet · IP | `NetTS_drift` |
| `periodic_spaces` | Structured zero/non-zero traffic cycles (e.g. nights, weekends) | IP only | `NetTS_periodic_spaces` |
| `workers` | Weekday business-hours concentration; silent on weekends | IP only | `NetTS_workers` |
| `random` | Near white-noise, minimal temporal structure | Institution · Subnet · IP | `NetTS_random` |

### Series Counts

**Hourly:**

| Benchmark | Institutions | Subnets | IPs | Total |
|-----------|:-----------:|:-------:|:---:|:-----:|
| `seasonal` | 25 | 25 | 25 | 75 |
| `drift` | 25 | 25 | 25 | 75 |
| `periodic_spaces` | — | — | 25 | 25 |
| `workers` | — | — | 25 | 25 |
| `random` | 13 | 25 | 25 | 63 |
| **Total** | | | | **263** |

**10-minute:**

| Benchmark | Institutions | Subnets | IPs | Total |
|-----------|:-----------:|:-------:|:---:|:-----:|
| `seasonal` | 25 | 25 | 25 | 75 |
| `drift` | 25 | 25 | 25 | 75 |
| `periodic_spaces` | — | — | 25 | 25 |
| `workers` | — | — | 25 | 25 |
| `random` | 19 | 25 | 25 | 69 |
| **Total** | | | | **269** |

Curated series IDs live in `benchmarks/selected_ids/` (tracked in git). Benchmark data files (CSV/Parquet) are generated from CESNET-TimeSeries24 and are not tracked — see [Setup](#setup).

---

## Evaluation Protocol

**Method:** Rolling-origin (sliding window) forecast over the 20% test split. Each window uses the preceding context as input and predicts the next fixed-horizon block.

**Primary metric:** MASE, normalised per series by its own seasonal-naive in-sample error (seasonal period m = 24 for hourly, m = 144 for 10-minute). Lower is better. Secondary metric: RMSSE.

Rankings use **average rank** (lower = better) computed per series across all models. The overall leaderboard pools per-benchmark average ranks across all four configurations. Seasonal Naive is the reference baseline.

### Configurations

| Configuration | Resolution | Context (C) | Horizon (H) |
|--------------|:----------:|:-----------:|:-----------:|
| `h_168_24` | Hourly | 168 steps (7 days) | 24 steps (1 day) |
| `h_744_168` | Hourly | 744 steps (31 days) | 168 steps (7 days) |
| `m_288_144` | 10-min | 288 steps (2 days) | 144 steps (1 day) |
| `m_1008_144` | 10-min | 1,008 steps (7 days) | 144 steps (1 day) |

---

## Baseline Results

### Overall (pooled across 4 configurations and 5 benchmarks)

| Rank | Model | Type | Drift | Seasonal | Periodic Spaces | Workers | Random | Avg Rank |
|:----:|-------|------|------:|--------:|----------------:|--------:|-------:|---------:|
| 1 | **TimesFM** | Foundation | 2.67 | 2.68 | 3.14 | 3.50 | 2.77 | **2.95** |
| 2 | **Chronos** | Foundation | 2.15 | 1.88 | 4.42 | 4.58 | 3.02 | **3.21** |
| 3 | Moirai | Foundation | 3.39 | 4.22 | 4.63 | 4.18 | 3.21 | 3.93 |
| 4 | GRU | Deep learning | 5.75 | 6.32 | 4.72 | 3.03 | 4.76 | 4.91 |
| 5 | GRU-FCN | Deep learning | 5.78 | 5.17 | 4.84 | 4.05 | 5.48 | 5.06 |
| 6 | LSTM | Deep learning | 6.44 | 6.54 | 4.95 | 3.44 | 5.53 | 5.38 |
| 7 | TTM | Foundation | 5.49 | 4.96 | 6.44 | 6.48 | 5.61 | 5.80 |
| 8 | SARIMA | Statistical | 7.42 | 7.24 | 6.91 | 8.27 | 7.82 | 7.53 |
| 9 | Seasonal Naive *(baseline)* | Statistical | 7.82 | 7.86 | 7.20 | 8.14 | 8.71 | 7.94 |
| 10 | Prophet | Statistical | 8.10 | 8.13 | 7.75 | 9.33 | 8.10 | 8.28 |

Foundation models (TimesFM, Chronos, Moirai) lead on Seasonal, Drift, and Random benchmarks. Deep learning models (GRU, LSTM) outperform on Workers, where the binary weekday/weekend structure is learnable from scratch, but drop sharply on Periodic Spaces at 10-minute resolution.

### Per-Benchmark Results

<details>
<summary><b>Seasonal</b> — strong daily and weekly periodic patterns</summary>

| Rank | Model | Type | Avg Rank (pooled) |
|:----:|-------|------|:-----------------:|
| 1 | **Chronos** | Foundation | **1.88** |
| 2 | TimesFM | Foundation | 2.68 |
| 3 | Moirai | Foundation | 4.22 |
| 4 | TTM | Foundation | 4.96 |
| 5 | GRU-FCN | Deep learning | 5.17 |
| 6 | GRU | Deep learning | 6.32 |
| 7 | LSTM | Deep learning | 6.54 |
| 8 | SARIMA | Statistical | 7.24 |
| 9 | Seasonal Naive *(baseline)* | Statistical | 7.86 |
| 10 | Prophet | Statistical | 8.13 |

</details>

<details>
<summary><b>Drift</b> — non-stationary trend changes over time</summary>

| Rank | Model | Type | Avg Rank (pooled) |
|:----:|-------|------|:-----------------:|
| 1 | **Chronos** | Foundation | **2.15** |
| 2 | TimesFM | Foundation | 2.67 |
| 3 | Moirai | Foundation | 3.39 |
| 4 | TTM | Foundation | 5.49 |
| 5 | GRU | Deep learning | 5.75 |
| 6 | GRU-FCN | Deep learning | 5.78 |
| 7 | LSTM | Deep learning | 6.44 |
| 8 | SARIMA | Statistical | 7.42 |
| 9 | Seasonal Naive *(baseline)* | Statistical | 7.82 |
| 10 | Prophet | Statistical | 8.10 |

</details>

<details>
<summary><b>Periodic Spaces</b> — regular zero-traffic periods (e.g. nights, weekends)</summary>

| Rank | Model | Type | Avg Rank (pooled) |
|:----:|-------|------|:-----------------:|
| 1 | **TimesFM** | Foundation | **3.14** |
| 2 | Chronos | Foundation | 4.42 |
| 3 | Moirai | Foundation | 4.63 |
| 4 | GRU | Deep learning | 4.72 |
| 5 | GRU-FCN | Deep learning | 4.84 |
| 6 | LSTM | Deep learning | 4.95 |
| 7 | TTM | Foundation | 6.44 |
| 8 | SARIMA | Statistical | 6.91 |
| 9 | Seasonal Naive *(baseline)* | Statistical | 7.20 |
| 10 | Prophet | Statistical | 7.75 |

</details>

<details>
<summary><b>Workers</b> — workstation patterns: active on workdays, silent on weekends</summary>

| Rank | Model | Type | Avg Rank (pooled) |
|:----:|-------|------|:-----------------:|
| 1 | **GRU** | Deep learning | **3.03** |
| 2 | LSTM | Deep learning | 3.44 |
| 3 | TimesFM | Foundation | 3.50 |
| 4 | GRU-FCN | Deep learning | 4.05 |
| 5 | Moirai | Foundation | 4.18 |
| 6 | Chronos | Foundation | 4.58 |
| 7 | TTM | Foundation | 6.48 |
| 8 | Seasonal Naive *(baseline)* | Statistical | 8.14 |
| 9 | SARIMA | Statistical | 8.27 |
| 10 | Prophet | Statistical | 9.33 |

</details>

<details>
<summary><b>Random</b> — near white-noise, no discernible structure</summary>

| Rank | Model | Type | Avg Rank (pooled) |
|:----:|-------|------|:-----------------:|
| 1 | **TimesFM** | Foundation | **2.77** |
| 2 | Chronos | Foundation | 3.02 |
| 3 | Moirai | Foundation | 3.21 |
| 4 | GRU | Deep learning | 4.76 |
| 5 | GRU-FCN | Deep learning | 5.48 |
| 6 | LSTM | Deep learning | 5.53 |
| 7 | TTM | Foundation | 5.61 |
| 8 | SARIMA | Statistical | 7.82 |
| 9 | Prophet | Statistical | 8.10 |
| 10 | Seasonal Naive *(baseline)* | Statistical | 8.71 |

</details>

### Per-Configuration Results

<details>
<summary><b>h_168_24</b> — 168-step context / 24-step horizon (hourly)</summary>

| Rank | Model | Drift | Seasonal | Periodic Spaces | Workers | Random | Avg Rank |
|:----:|-------|------:|--------:|----------------:|--------:|-------:|---------:|
| 1 | **TimesFM** | 2.77 | 2.67 | 3.92 | 3.48 | 3.11 | **3.19** |
| 2 | Chronos | 2.33 | 2.00 | 5.04 | 5.00 | 3.89 | 3.65 |
| 3 | TTM | 3.59 | 4.12 | 4.00 | 3.88 | 3.68 | 3.85 |
| 4 | Moirai | 3.63 | 3.97 | 5.44 | 4.60 | 3.30 | 4.19 |
| 5 | GRU | 5.35 | 5.88 | 3.36 | 2.72 | 4.24 | 4.31 |
| 6 | LSTM | 6.67 | 6.44 | 3.04 | 2.44 | 5.40 | 4.80 |
| 7 | GRU-FCN | 5.07 | 3.80 | 5.68 | 6.56 | 5.37 | 5.29 |
| 8 | Seasonal Naive | 8.47 | 8.32 | 7.28 | 7.76 | 8.83 | 8.13 |
| 9 | Prophet | 8.43 | 9.05 | 8.20 | 8.76 | 8.19 | 8.53 |
| 10 | SARIMA | 8.71 | 8.75 | 9.04 | 9.80 | 9.00 | 9.06 |

</details>

<details>
<summary><b>h_744_168</b> — 744-step context / 168-step horizon (hourly)</summary>

| Rank | Model | Drift | Seasonal | Periodic Spaces | Workers | Random | Avg Rank |
|:----:|-------|------:|--------:|----------------:|--------:|-------:|---------:|
| 1 | **TimesFM** | 2.33 | 1.85 | 3.30 | 3.92 | 2.52 | **2.79** |
| 2 | Chronos | 2.29 | 2.19 | 4.04 | 4.92 | 2.40 | 3.17 |
| 3 | Moirai | 2.47 | 2.51 | 4.32 | 3.96 | 2.81 | 3.21 |
| 4 | GRU-FCN | 7.19 | 7.55 | 3.04 | 1.88 | 7.32 | 5.39 |
| 5 | GRU | 7.41 | 8.08 | 4.12 | 2.28 | 6.48 | 5.67 |
| 6 | LSTM | 7.97 | 8.12 | 5.08 | 4.12 | 6.17 | 6.29 |
| 7 | SARIMA | 6.15 | 5.68 | 7.02 | 7.44 | 6.75 | 6.61 |
| 8 | TTM | 6.16 | 5.56 | 7.96 | 8.32 | 5.63 | 6.73 |
| 9 | Seasonal Naive | 5.33 | 6.25 | 7.26 | 8.40 | 7.46 | 6.94 |
| 10 | Prophet | 7.69 | 7.21 | 8.86 | 9.76 | 7.46 | 8.20 |

</details>

<details>
<summary><b>m_288_144</b> — 288-step context / 144-step horizon (10-minute)</summary>

| Rank | Model | Drift | Seasonal | Periodic Spaces | Workers | Random | Avg Rank |
|:----:|-------|------:|--------:|----------------:|--------:|-------:|---------:|
| 1 | **TimesFM** | 3.04 | 3.57 | 1.58 | 2.12 | 2.78 | **2.62** |
| 2 | Chronos | 2.21 | 2.16 | 3.68 | 3.80 | 2.96 | 2.96 |
| 3 | Moirai | 3.77 | 5.37 | 3.72 | 3.12 | 3.28 | 3.85 |
| 4 | GRU-FCN | 5.44 | 4.80 | 5.72 | 4.04 | 4.61 | 4.92 |
| 5 | GRU | 5.05 | 5.64 | 8.28 | 5.52 | 4.33 | 5.77 |
| 6 | LSTM | 5.29 | 5.09 | 8.36 | 5.48 | 5.19 | 5.88 |
| 7 | TTM | 5.95 | 5.09 | 6.12 | 6.36 | 6.39 | 5.98 |
| 8 | SARIMA | 7.44 | 6.96 | 5.34 | 7.68 | 7.83 | 7.05 |
| 9 | Prophet | 8.07 | 7.95 | 5.46 | 8.92 | 8.29 | 7.74 |
| 10 | Seasonal Naive | 8.73 | 8.36 | 6.74 | 7.96 | 9.35 | 8.23 |

</details>

<details>
<summary><b>m_1008_144</b> — 1008-step context / 144-step horizon (10-minute)</summary>

| Rank | Model | Drift | Seasonal | Periodic Spaces | Workers | Random | Avg Rank |
|:----:|-------|------:|--------:|----------------:|--------:|-------:|---------:|
| 1 | **Chronos** | 1.75 | 1.17 | 4.92 | 4.60 | 2.83 | **3.05** |
| 2 | TimesFM | 2.53 | 2.63 | 3.74 | 4.48 | 2.65 | 3.21 |
| 3 | GRU | 5.17 | 5.67 | 3.12 | 1.60 | 3.99 | 3.91 |
| 4 | Moirai | 3.69 | 5.03 | 5.04 | 5.04 | 3.43 | 4.45 |
| 5 | LSTM | 5.83 | 6.51 | 3.32 | 1.72 | 5.35 | 4.54 |
| 6 | GRU-FCN | 5.41 | 4.55 | 4.92 | 3.72 | 4.64 | 4.65 |
| 7 | TTM | 6.28 | 5.05 | 7.68 | 7.36 | 6.72 | 6.62 |
| 8 | SARIMA | 7.40 | 7.57 | 6.26 | 8.16 | 7.72 | 7.42 |
| 9 | Seasonal Naive | 8.73 | 8.51 | 7.50 | 8.44 | 9.19 | 8.47 |
| 10 | Prophet | 8.20 | 8.32 | 8.50 | 9.88 | 8.48 | 8.68 |

</details>

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| Seasonal Naive | Statistical baseline | Repeats last seasonal cycle (period 24 h / 144×10 min) |
| SARIMA | Statistical | Statsmodels SARIMAX, refits each window |
| Prophet | Statistical | Facebook Prophet with holiday calendar |
| TimesFM | Foundation (zero-shot) | Google TimesFM 2.5 200M, batched inference |
| TTM | Foundation (zero-shot) | IBM TTM, revision selected per context length |
| Chronos | Foundation (zero-shot) | Amazon Chronos 2, median quantile, batched |
| Moirai | Foundation (zero-shot) | Salesforce Moirai 2.0-R-small via GluonTS |
| GRU | Deep learning | Bidirectional GRU (tsai/fastai), log1p + per-window instance normalisation |
| LSTM | Deep learning | Bidirectional LSTM (tsai/fastai), same pipeline as GRU |
| GRU-FCN | Deep learning | GRU + FCN hybrid (tsai/fastai), same pipeline |

---

## Setup

```bash
conda create -n nettseval python=3.12 -y
conda activate nettseval
pip install -r requirements.txt
```

### Export Benchmark Data

Download CESNET-TimeSeries24 to `../data` (see [cesnet-tszoo docs](https://cesnet.github.io/cesnet-tszoo/)), then export the benchmark CSVs:

```bash
python save_benchmarks.py
```

---

## Reproducing Results

```bash
# Single model, single benchmark
python run_evaluation.py --model timesfm --benchmark seasonal

# All benchmarks
python run_evaluation.py --model chronos --benchmark all

# Specify configuration explicitly
python run_evaluation.py --model ttm --benchmark seasonal --freq h --lookback 744 --horizon 168

# Multiple repeated runs for variance estimation
python run_evaluation.py --model gru --benchmark all --n-runs 3
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | One or more model names (required) | — |
| `--benchmark` | One or more benchmark types, or `all` | `all` |
| `--freq` | Resolution: `h`=hourly, `m`=10min | `h` |
| `--lookback` | Context window size | h=168, m=288 |
| `--horizon` | Forecast horizon | h=24, m=144 |
| `--n-runs` | Repeated evaluation runs | 1 |
| `--save-predictions` | Save per-window prediction CSVs | off |
| `--config` | JSON file for model hyperparameter overrides | — |

### Generate Analysis Graphs and Tables

```bash
python -m result_exploration.generate
python -m result_exploration.generate --metrics rmse mase
```

Output is written to `result_exploration/graphs/`. Result directory paths per model are configured in `result_exploration/analysis.py` under `EXPERIMENT_CONFIGS`. The full analysis notebook (ranks, skill scores, significance tests) is at `result_exploration/results.ipynb`.

---

## Adding New Models

1. Create `nettseval/models/newmodel.py`, inherit from `BaseModel`
2. Implement `fit(train_data, target_column)`, `predict(horizon)`, `get_params()`, `get_name()`
3. Optionally implement `tune(train, val, target_column, ts_id=None)` → `dict | None` for per-series hyperparameter fitting
4. Optionally set `supports_batch = True` and implement `predict_batch(contexts, target_column, horizon)` for batched inference
5. Export in `nettseval/models/__init__.py`
6. Add to `MODEL_ORDER` in `nettseval/constants.py` and to `create_model()` in `run_evaluation.py`

See [adding_a_model.md](adding_a_model.md) for a full guide with code templates.

---

## Repository Structure

```
nettseval/                   # Main Python package
  models/                   # Model wrappers (BaseModel interface)
  benchmarks/               # BenchmarkSource, BenchmarkLoader
  evaluation/               # Evaluator, metrics registry, result storage
  utils/                    # EvaluationConfig dataclass
  constants.py              # MODEL_ORDER, DEFAULT_EVAL_METRICS, SEASONAL_PERIOD
run_evaluation.py            # Main CLI entry point
save_benchmarks.py           # Export NetTS from CESNET-TimeSeries24
benchmarks/                 # Benchmark CSVs/Parquet + selected_ids/ (IDs tracked in git)
results/                    # Evaluation output (gitignored)
result_exploration/         # Analysis pipeline (analysis.py, plots.py, generate.py)
  results.ipynb             # Full results notebook (ranks, skill scores, significance tests)
  predictions.ipynb         # Prediction visualization notebook
  bench_ts_sparsity/        # Benchmark time series sparsity analysis
metacentrum/                # PBS cluster job submission scripts
```
