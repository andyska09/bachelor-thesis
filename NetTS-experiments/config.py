from cesnet_tszoo.utils.enums import AgreggationType

RANDOM_SEED = 42
FEATURE = "n_bytes"

AGGREGATIONS = {
    "hourly": {
        "enum": AgreggationType.AGG_1_HOUR,
        "time_range": range(0, 6718),
        "daily_period": 24,
        "weekly_period": 168,
    },
    "10min": {
        "enum": AgreggationType.AGG_10_MINUTES,
        "time_range": range(0, 40298),
        "daily_period": 144,
        "weekly_period": 1008,
    },
}

BENCHMARKS = {
    "SEASON": {
        "levels": ["institutions", "subnets", "ips"],
        "sparsity_min": 0.4,
        "threshold": 0.7,
        "metric_col": "max_strength",
        "metric_op": ">=",
        "sample_size": 25,
    },
    "DRIFT": {
        "levels": ["institutions", "subnets", "ips"],
        "sparsity_min": 0.9,
        "threshold": 0.30,
        "metric_col": "max_rel_dev",
        "metric_op": "custom",  # has_drift & drift_in_test
        "sample_size": 25,
        "short_window_factor": 168,  # in hourly units, scaled by agg
        "long_window_factor": 744,
        "min_drift_days": 14,
        "train_val_frac": 0.8,
    },
    "DRIFT_SWEEP": {
        "levels": ["ips_full"],
        "sparsity_min": 0.9,
        "threshold": 0.30,
        "metric_col": "max_rel_dev",
        "metric_op": "custom",
        "sample_size": 25,
        "short_window_factor": 168,
        "long_window_factor": 744,
        "min_drift_days": 14,
        "train_val_frac": 0.8,
        "dev_sweep": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        "persistence_sweep_days": [7, 14, 21, 28],
    },
    "PERIODIC_SPACES": {
        "levels": ["ips"],
        "sparsity_min": 0.05,
        "sparsity_max": 0.95,
        "threshold": 0.6,
        "metric_col": "max_acf",
        "metric_op": ">=",
        "sample_size": 25,
        "max_lag_factor": 336,  # in hourly units
        "peak_prominence": 0.1,
    },
    "WORKERS": {
        "levels": ["ips"],
        "sparsity_min": 0.23,
        "threshold": 0.6,
        "metric_col": "whr",
        "metric_op": ">=",
        "sample_size": 25,
        "needs_calendar": True,
    },
    "RANDOM": {
        "levels": ["institutions", "subnets", "ips"],
        "sparsity_min": 0.4,
        "threshold": 0.1,
        "metric_col": "max_acf",
        "metric_op": "<=",
        "sample_size": 25,
        "max_lag_factor": 168,  # in hourly units
    },
}
