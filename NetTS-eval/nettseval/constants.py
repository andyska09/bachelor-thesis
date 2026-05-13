"""Single source of truth for constants shared across nettseval modules."""

MODEL_ORDER = ["timesfm", "chronos", "moirai", "gru", "gru_fcn", "lstm", "ttm", "sarima", "seasonal_naive", "prophet"]

DEFAULT_EVAL_METRICS = ["mase", "rmsse"]

SCALED_METRICS = {"mase", "rmsse"}

FREQ_DIR = {"h": "hourly", "d": "daily", "m": "10min"}

SEASONAL_PERIOD = {"h": 24, "d": 1, "m": 144}

PANDAS_FREQ = {"h": "h", "d": "D", "m": "10min"}
