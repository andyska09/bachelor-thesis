# Benchmarking Network Traffic Forecasting

Bachelor's thesis attachments — Ondřej Skácel, CTU FIT, 2026. Supervisor: Ing. Josef Koumar.

This thesis presents five diagnostic benchmarks from the CESNET-TimeSeries24 dataset, each isolating one structural property of network traffic (seasonality, data drift, periodic intermittent activity, workstation-driven schedules, or the absence of temporal structure). Nine forecasting models from statistical, deep learning, and zero-shot foundation model families are compared using a framework implemented as part of this work.

## Contents

```
.
├── NetTS-experiments/           Benchmark construction
│   ├── scoring/                 Score functions (one per benchmark type)
│   ├── scores/                  Pre-computed score CSVs (hourly/ and 10min/)
│   ├── selected_ids/            Final selected series IDs
│   ├── selection.py             Threshold + random sampling from scores
│   ├── config.py                All benchmark parameters and aggregation configs
│   ├── sensitivity_analysis/    Threshold sensitivity sweep notebooks
│   ├── exploration/             Data exploration notebooks
│   └── benchmarks/              Verification notebooks
│
├── NetTS-eval/                  Evaluation framework
│   ├── nettseval/               Python package (models, evaluation, metrics, benchmarks)
│   ├── run_evaluation.py        Main CLI entry point
│   ├── save_benchmarks.py       Export benchmark data from CESNET-TimeSeries24
│   ├── benchmarks/              Benchmark data + selected series IDs (LFS-tracked)
│   ├── results/                 Evaluation output (LFS-tracked)
│   └── result_exploration/      Analysis pipeline (tables, plots, notebooks)
│
├── text/                        Thesis text
├── LICENSE                      MIT License
└── README.md
```

**NetTS-experiments** scores each series on a target property, applies thresholds, and randomly samples to produce curated series IDs. 

**NetTS-eval** consumes those IDs, exports benchmark data from CESNET-TimeSeries24, runs forecasting models, and produces evaluation results.

See [NetTS-experiments/README.md](NetTS-experiments/README.md) and [NetTS-eval/README.md](NetTS-eval/README.md) for setup instructions, CLI usage, and detailed documentation.
