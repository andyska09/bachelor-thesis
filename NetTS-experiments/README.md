# NetTS: Diagnostic Benchmarks for Network Traffic Forecasting

Benchmark construction code for the NetTS suite, developed as part of a bachelor's thesis at CTU FIT by Ondřej Skácel.

## Overview

Network traffic forecasting is an active but fragmented research area. Existing evaluations rely on datasets that are often private, cover only a small subset of available entities, or are incomparable across studies — making it difficult or impossible to systematically compare forecasting approaches across the domain.

This repository provides **NetTS**, a suite of five diagnostic benchmarks built on the publicly available [CESNET-TimeSeries24](https://github.com/CESNET/cesnet-tszoo) dataset. Unlike general-purpose time series benchmarks, each NetTS benchmark isolates a specific behavioral property of network traffic. This design supports **explainable evaluation**: when a model underperforms on a given benchmark, the failure can be attributed to a concrete property (e.g., inability to track distribution shift, or to handle structured periodic absences), rather than to an undifferentiated mix of dataset characteristics.

To our knowledge, no comparable public benchmark suite exists for network traffic forecasting.

## Benchmarks

Each benchmark selects series that exhibit a well-defined property. Series are drawn from three aggregation levels: Institutions (283 series), Institution Subnets (548 series), and IP Addresses (up to ~275k series).

| Benchmark | Property | Levels |
|-----------|----------|--------|
| **SEASON** | Strong daily and/or weekly seasonality (STL seasonal strength ≥ 0.7) | Institutions, Subnets, IPs |
| **DRIFT** | Sustained distribution shift between train and test periods (≥ 30% MA deviation for ≥ 14 consecutive days, visible in test split) | Institutions, Subnets, IPs |
| **PERIODIC_SPACES** | Periodic intermittent traffic — structured zero/non-zero on/off cycles (binary ACF peak ≥ 0.6) | IPs only |
| **WORKERS** | Working-hours pattern — activity concentrated on weekday business hours (WHR ≥ 0.6) | IPs only |
| **RANDOM** | No exploitable temporal structure — near white noise (max |ACF| ≤ 0.1 across lags 1–168) | Institutions, Subnets, IPs |

Each benchmark selects **25 series per level** via a two-stage pipeline: score all candidates on the target metric, apply a threshold to form a qualifying pool, then randomly sample (`random_state=42`). This prevents selection bias from cherry-picking top-ranked series.

All benchmarks are provided at two temporal resolutions: **hourly** (`AGG_1_HOUR`) and **10-minute** (`AGG_10_MINUTES`).

## Dataset

CESNET-TimeSeries24 is a large-scale network volumetric dataset from a high-speed ISP backbone. It contains time series of `n_bytes` (and other features) aggregated at multiple resolutions across Institutions, Institution Subnets, and individual IP addresses.

Data must be placed at `../data/tszoo/databases/CESNET-TimeSeries24/` relative to this repository, or the path configured in `utils.py`. The dataset is available from the [cesnet-tszoo](https://github.com/CESNET/cesnet-tszoo) library.

## Setup

```bash
conda create -n netmts-env python=3.11
conda activate netmts-env
pip install -r requirements.txt
```

## Reproducing the Benchmarks

Benchmark construction is a two-stage pipeline.

**Stage 1 — Scoring** (computationally expensive; run once per aggregation level):

```bash
# Score one benchmark and source
python -m scoring.runner --aggregation hourly --benchmark SEASON --source institutions

# Score all sources for one benchmark
python -m scoring.runner --aggregation hourly --benchmark SEASON

# Score everything at 10-minute resolution
python -m scoring.runner --aggregation 10min
```

Pre-computed score CSVs are provided in `scores/` so Stage 1 can be skipped.

**Stage 2 — Selection** (instant; reads pre-computed scores):

```bash
python selection.py --aggregation hourly --benchmark SEASON --source institutions subnets ips
python selection.py --aggregation hourly --benchmark DRIFT  --source institutions subnets ips
python selection.py --aggregation hourly --benchmark PERIODIC_SPACES --source ips
python selection.py --aggregation hourly --benchmark WORKERS --source ips
python selection.py --aggregation hourly --benchmark RANDOM  --source institutions subnets ips
```

Final selected series IDs are saved to `selected_ids/` and are also pre-committed to this repository.

**From a notebook or script:**

```python
from scoring.runner import score_source
from selection import select_source
import pandas as pd

# Run scoring directly
scores_df = score_source("SEASON", "institutions", aggregation="hourly")

# Run selection directly
ids, pool_size, scores_df = select_source("SEASON", "institutions", aggregation="hourly")

# Or read pre-computed scores
scores_df = pd.read_csv("scores/hourly/SEASON_institutions.csv")
```

**HPC (for `ips_full`, ~275k series):**

Scoring the full IP dataset is too slow for a single machine. PBS job submission for MetaCentrum HPC is provided:

```bash
python -m metacentrum.submit -b SEASON -s ips_full -a hourly
python -m metacentrum.submit -b SEASON -s ips_full --dry-run   # preview qsub command
```

The runner supports resumable chunked processing for `ips_full` — partial results are saved and resumed on restart.

## Repository Structure

```
config.py                    # All benchmark parameters and aggregation configs
utils.py                     # Data loading, sparsity helpers, plotting helpers

scoring/
├── __init__.py              # SCORE_FUNCTIONS registry (benchmark name → score_fn)
├── runner.py                # Generic: load data → score → save CSV
├── seasonality.py           # Scoring for SEASON
├── drift.py                 # Scoring for DRIFT
├── periodic_spaces.py       # Scoring for PERIODIC_SPACES
├── workstations.py          # Scoring for WORKERS
└── random.py                # Scoring for RANDOM

scores/                      # Pre-computed score CSVs (hourly/ and 10min/)
selection.py                 # Threshold + random sample from scores
selected_ids/                # Final selected series IDs (hourly/ and 10min/)

benchmarks/
├── hourly/                  # Verification notebooks for hourly benchmarks
└── 10min/                   # Verification notebooks for 10-minute benchmarks

exploration/                 # Data exploration and analysis notebooks
sensitivity_analysis/        # Threshold sensitivity sweep notebooks
metacentrum/                 # PBS job scripts for HPC cluster (ips_full scoring)
```

## Key Design Decisions

**No selection bias.** Each benchmark defines a scoring metric and threshold, then randomly samples from the qualifying pool. Series are never sorted and top-N picked, which would systematically bias toward extreme examples.

**log1p transform before spectral analysis.** Raw network traffic is bursty with variance proportional to level. Applying `log1p(x)` before STL decomposition stabilizes variance and handles zeros safely (`log1p(0) = 0`). This is equivalent to multiplicative decomposition.

**No data leakage.** Series are characterized on training data or the full series. Benchmarks do not use test-set appearance for selection. The DRIFT benchmark additionally requires that detected drift overlaps with the test portion (last 20%) to ensure the property is visible at evaluation time.

**Aggregation scaling.** All parameters defined in hourly units (window sizes, lag counts, duration thresholds) are scaled by a factor of 6 for 10-minute resolution via `scale = agg_params["daily_period"] / 24`. This ensures methodological consistency across resolutions.

**Train/val/test split:** 60% / 20% / 20%.
