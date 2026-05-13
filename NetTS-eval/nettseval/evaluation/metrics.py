from typing import Callable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from nettseval.constants import SCALED_METRICS


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, 1e-10, denominator)

    return 200 * np.mean(numerator / denominator)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def seasonal_error(y_hist: np.ndarray, m: int) -> float:
    if len(y_hist) <= m:
        return np.nan
    return float(np.mean(np.abs(y_hist[m:] - y_hist[:-m])))


def squared_seasonal_error(y_hist: np.ndarray, m: int) -> float:
    if len(y_hist) <= m:
        return np.nan
    return float(np.mean((y_hist[m:] - y_hist[:-m]) ** 2))


def calculate_mase_windowed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_hist_base: np.ndarray,
    horizon: int,
    n_windows: int,
    m: int,
) -> float:
    n = len(y_true)
    values = []
    for w in range(n_windows):
        start = w * horizon
        end = min(start + horizon, n)
        y_hist = np.concatenate([y_hist_base, y_true[:start]])
        denom = seasonal_error(y_hist, m)
        if np.isnan(denom) or denom == 0:
            continue
        values.append(float(np.mean(np.abs(y_true[start:end] - y_pred[start:end])) / denom))
    return float(np.nanmean(values)) if values else float("nan")


def calculate_rmsse_windowed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_hist_base: np.ndarray,
    horizon: int,
    n_windows: int,
    m: int,
) -> float:
    n = len(y_true)
    values = []
    for w in range(n_windows):
        start = w * horizon
        end = min(start + horizon, n)
        y_hist = np.concatenate([y_hist_base, y_true[:start]])
        denom = squared_seasonal_error(y_hist, m)
        if np.isnan(denom) or denom == 0:
            continue
        values.append(float(np.sqrt(np.mean((y_true[start:end] - y_pred[start:end]) ** 2) / denom)))
    return float(np.nanmean(values)) if values else float("nan")


_REGISTRY: dict[str, Callable] = {
    "rmse": calculate_rmse,
    "smape": calculate_smape,
    "r2": calculate_r2,
    "mae": calculate_mae,
    "mase": calculate_mase_windowed,
    "rmsse": calculate_rmsse_windowed,
}

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[str],
    y_hist_base: np.ndarray,
    horizon: int,
    n_windows: int,
    m: int,
) -> dict[str, float]:
    result = {}
    for name in metrics:
        fn = _REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown metric: {name!r}. Available: {sorted(_REGISTRY)}")
        if name in SCALED_METRICS:
            result[name] = fn(y_true, y_pred, y_hist_base, horizon, n_windows, m)
        else:
            result[name] = fn(y_true, y_pred)
    return result
