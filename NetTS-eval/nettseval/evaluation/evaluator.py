import math
import time

import numpy as np
import pandas as pd

from nettseval.benchmarks.benchmark import BenchmarkSource
from nettseval.evaluation.metrics import calculate_metrics
from nettseval.evaluation.results import BenchmarkSourceResult, SeriesResult
from nettseval.models.base import BaseModel
from nettseval.utils.config import EvaluationConfig


class Evaluator:
    def __init__(self, benchmark: BenchmarkSource, model: BaseModel, config: EvaluationConfig):
        self.benchmark = benchmark
        self.model = model
        self.config = config

    def evaluate(self) -> BenchmarkSourceResult:
        context_size = self.model.get_required_context_size() or self.config.context_size
        model_params = self.model.get_params()
        results = []
        n_failed = 0

        for ts_id in self.benchmark.ts_ids:
            print(f"    {ts_id}")
            try:
                results.append(self._evaluate_series(ts_id, context_size))
            except Exception as e:
                n_failed += 1
                print(f"    ERROR {ts_id}: {e}")

        if n_failed:
            print(f"  WARNING: {n_failed}/{len(self.benchmark.ts_ids)} series failed")

        return BenchmarkSourceResult(
            bench_type=self.benchmark.bench_type,
            agg_level=self.benchmark.source_name,
            model_name=self.model.get_name(),
            model_params=model_params,
            per_series_results=results,
            config=self.config,
        )

    def _build_context(self, ts_id: int, context_size: int) -> tuple:
        train = self.benchmark.get_series(ts_id, "train")
        val = self.benchmark.get_series(ts_id, "val")
        test = self.benchmark.get_series(ts_id, "test")
        context = pd.concat([train, val], ignore_index=True).iloc[-context_size:].reset_index(drop=True)
        return train, val, test, context

    def _build_result(
        self,
        ts_id: int,
        n_windows: int,
        all_predictions: list,
        all_actuals: list,
        all_datetimes: list,
        tune_time_per_100pts: float,
        prediction_time_per_100pts: float,
        tuned_params: dict | None = None,
        y_hist_base: np.ndarray | None = None,
    ) -> SeriesResult:
        actuals = np.array(all_actuals)
        predictions = np.array(all_predictions)
        return SeriesResult(
            ts_id=ts_id,
            n_windows=n_windows,
            tuned_params=tuned_params,
            datetimes=np.array(all_datetimes),
            predictions=predictions,
            actuals=actuals,
            metrics=calculate_metrics(
                actuals,
                predictions,
                self.config.metrics,
                y_hist_base=y_hist_base,
                horizon=self.config.forecast_horizon,
                n_windows=n_windows,
                m=self.config.mase_m,
            ),
            tune_time_per_100pts=tune_time_per_100pts,
            prediction_time_per_100pts=prediction_time_per_100pts,
        )

    def _evaluate_series(self, ts_id: int, context_size: int) -> SeriesResult:
        train, val, test, context = self._build_context(ts_id, context_size)
        n_train_val = len(train) + len(val)
        t_tune = time.perf_counter()
        tuned_params = self.model.tune(train, val, self.config.target_column, ts_id=ts_id)
        tune_time_per_100pts = (time.perf_counter() - t_tune) / n_train_val * 100

        target_col = self.config.target_column
        horizon = self.config.forecast_horizon
        test_len = len(test)
        n_windows = math.ceil(test_len / horizon)

        contexts = [
            pd.concat([context, test.iloc[: w * horizon]], ignore_index=True)
            .iloc[-context_size:]
            .reset_index(drop=True)
            for w in range(n_windows)
        ]

        t0 = time.perf_counter()
        if self.model.supports_batch:
            raw_preds = self.model.predict_batch(contexts, target_col, horizon)
        else:
            raw_preds = []
            for w, ctx in enumerate(contexts):
                self.model.fit(ctx, target_col)
                raw_preds.append(self.model.predict(min(horizon, test_len - w * horizon)))
        prediction_time_per_100pts = (time.perf_counter() - t0) / test_len * 100

        all_predictions, all_actuals, all_datetimes = [], [], []
        for w, preds in enumerate(raw_preds):
            start = w * horizon
            end = min(start + horizon, test_len)
            actuals_df = test.iloc[start:end].reset_index(drop=True)
            all_predictions.extend(np.asarray(preds)[: end - start])
            all_actuals.extend(actuals_df[target_col].values)
            all_datetimes.extend(actuals_df["datetime"].values)

        return self._build_result(
            ts_id,
            n_windows,
            all_predictions,
            all_actuals,
            all_datetimes,
            tune_time_per_100pts,
            prediction_time_per_100pts,
            tuned_params,
            y_hist_base=pd.concat([train, val], ignore_index=True)[target_col].values,
        )
