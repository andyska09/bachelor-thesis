from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from nettseval.utils.config import EvaluationConfig


class SeriesResult(TypedDict):
    ts_id: int
    n_windows: int
    tuned_params: dict | None
    datetimes: np.ndarray
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: dict[str, float]
    tune_time_per_100pts: float
    prediction_time_per_100pts: float


def _compute_timing(series: list[SeriesResult]) -> dict[str, float]:
    return {
        "tune_time_per_100pts": float(np.mean([s["tune_time_per_100pts"] for s in series])),
        "prediction_time_per_100pts": float(np.mean([s["prediction_time_per_100pts"] for s in series])),
    }


def _compute_stats(series: list[SeriesResult], keys: list[str]) -> dict[str, float]:
    result = {}
    for key in keys:
        values = [r["metrics"][key] for r in series]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values, ddof=1))
    return result


@dataclass
class BenchmarkSourceResult:
    bench_type: str
    agg_level: str
    model_name: str
    model_params: dict
    per_series_results: list[SeriesResult]
    config: EvaluationConfig

    @property
    def aggregate_metrics(self) -> dict:
        if not self.per_series_results:
            return {}
        result = _compute_stats(self.per_series_results, self.config.metrics)
        result.update(_compute_timing(self.per_series_results))
        return result

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.per_series_results:
            row = {"id": r["ts_id"], "agg_level": self.agg_level}
            row.update(r["metrics"])
            rows.append(row)
        return pd.DataFrame(rows)

    def to_predictions_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.per_series_results:
            for dt, pred, act in zip(r["datetimes"], r["predictions"], r["actuals"]):
                rows.append(
                    {
                        "id": r["ts_id"],
                        "agg_level": self.agg_level,
                        "datetime": dt,
                        "prediction": pred,
                        "actual": act,
                    }
                )
        return pd.DataFrame(rows)


def save_benchmark_results(results: list[BenchmarkSourceResult], run_dir: Path, save_predictions: bool = False) -> None:
    """Save per-series metrics (and optionally predictions) for one benchmark type to run_dir."""
    if not results:
        return
    bench_type = results[0].bench_type
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([r.to_dataframe() for r in results], ignore_index=True).to_csv(
        run_dir / f"{bench_type}_metrics.csv", index=False
    )
    if save_predictions:
        pd.concat([r.to_predictions_dataframe() for r in results], ignore_index=True).to_csv(
            run_dir / f"{bench_type}_predictions.csv", index=False
        )


def save_run_summary(results: list[BenchmarkSourceResult], run_dir: Path) -> None:
    """Save mean/std of metrics per bench_type and agg_level for a single run to run_dir/summary.csv."""
    rows = [
        {"bench_type": r.bench_type, "agg_level": r.agg_level, **r.aggregate_metrics}
        for r in results
        if r.per_series_results
    ]
    if rows:
        metric_keys = results[0].config.metrics
        all_series = [s for r in results for s in r.per_series_results]
        rows.append({
            "bench_type": "all",
            "agg_level": "all",
            **_compute_stats(all_series, metric_keys),
            **_compute_timing(all_series),
        })
        pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)


def save_summary(all_run_results: list[list[BenchmarkSourceResult]], model_dir: Path) -> None:
    """Save mean/std of metrics per bench_type aggregated across all runs to model_dir/summary.csv."""
    all_flat = [r for run in all_run_results for r in run if r.per_series_results]
    if not all_flat:
        return

    metric_keys = all_flat[0].config.metrics
    by_bench: dict[str, list[SeriesResult]] = {}
    for r in all_flat:
        by_bench.setdefault(r.bench_type, []).extend(r.per_series_results)

    rows = [
        {
            "bench_type": bt,
            **_compute_stats(series, metric_keys),
            **_compute_timing(series),
        }
        for bt, series in sorted(by_bench.items())
    ]
    all_series = [s for series in by_bench.values() for s in series]
    rows.append({
        "bench_type": "all",
        **_compute_stats(all_series, metric_keys),
        **_compute_timing(all_series),
    })
    pd.DataFrame(rows).to_csv(model_dir / "summary.csv", index=False)
