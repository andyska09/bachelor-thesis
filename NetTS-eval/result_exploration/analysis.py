import math
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import rankdata

from nettseval.constants import MODEL_ORDER, SCALED_METRICS, FREQ_DIR, SEASONAL_PERIOD
from nettseval.evaluation.metrics import calculate_metrics

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR.parent / "results"
GRAPHS_DIR = _SCRIPT_DIR / "graphs"
BENCH_BASE_DIR = _SCRIPT_DIR.parent / "benchmarks"

EXPERIMENT_CONFIGS: dict[str, dict[str, Path]] = {
    "h_168_24": {
        # "zero": RESULTS_DIR / "20260428_202615_zero_h_168_24",
        # "mean": RESULTS_DIR / "20260428_202620_mean_h_168_24",
        "seasonal_naive": RESULTS_DIR / "20260428_113720_seasonal_naive_h_168_24",
        "sarima": RESULTS_DIR / "20260426_200640_sarima_h_168_24",
        "prophet": RESULTS_DIR / "20260426_200640_prophet_h_168_24",
        "timesfm": RESULTS_DIR / "20260426_200640_timesfm_h_168_24",
        "ttm": RESULTS_DIR / "20260426_201223_ttm_h_168_24",
        "chronos": RESULTS_DIR / "20260426_200553_chronos_h_168_24",
        "moirai": RESULTS_DIR / "20260426_202404_moirai_h_168_24",
        "gru": RESULTS_DIR / "20260426_200957_gru_h_168_24",
        "lstm": RESULTS_DIR / "20260426_202318_lstm_h_168_24",
        "gru_fcn": RESULTS_DIR / "20260426_202146_gru_fcn_h_168_24",
    },
    "h_744_168": {
        # "zero": RESULTS_DIR / "20260428_202658_zero_h_744_168",
        # "mean": RESULTS_DIR / "20260428_202700_mean_h_744_168",
        "seasonal_naive": RESULTS_DIR / "20260426_200640_seasonal_naive_h_744_168",
        "sarima": RESULTS_DIR / "20260324_225714_sarima_h_744_168",
        "prophet": RESULTS_DIR / "20260428_114207_prophet_h_744_168",
        "timesfm": RESULTS_DIR / "20260426_203757_timesfm_h_744_168",
        "ttm": RESULTS_DIR / "20260426_203757_ttm_h_744_168",
        "chronos": RESULTS_DIR / "20260426_203825_chronos_h_744_168",
        "moirai": RESULTS_DIR / "20260426_203757_moirai_h_744_168",
        "gru": RESULTS_DIR / "20260426_204352_gru_h_744_168",
        "lstm": RESULTS_DIR / "20260426_204900_lstm_h_744_168",
        "gru_fcn": RESULTS_DIR / "20260426_204341_gru_fcn_h_744_168",
    },
    "m_288_144": {
        # "zero": RESULTS_DIR / "20260428_202754_zero_m_288_144",
        # "mean": RESULTS_DIR / "20260428_202807_mean_m_288_144",
        "seasonal_naive": RESULTS_DIR / "20260426_200823_seasonal_naive_m_288_144",
        "sarima": RESULTS_DIR / "20260426_200824_sarima_m_288_144",
        "prophet": RESULTS_DIR / "20260426_200823_prophet_m_288_144",
        "timesfm": RESULTS_DIR / "20260426_204458_timesfm_m_288_144",
        "ttm": RESULTS_DIR / "20260426_205122_ttm_m_288_144",
        "chronos": RESULTS_DIR / "20260426_205531_chronos_m_288_144",
        "moirai": RESULTS_DIR / "20260426_210023_moirai_m_288_144",
        "gru": RESULTS_DIR / "20260426_210023_gru_m_288_144",
        "lstm": RESULTS_DIR / "20260426_205905_lstm_m_288_144",
        "gru_fcn": RESULTS_DIR / "20260426_205908_gru_fcn_m_288_144",
    },
    "m_1008_144": {
        # "zero": RESULTS_DIR / "20260428_202924_zero_m_1008_144",
        # "mean": RESULTS_DIR / "20260428_202937_mean_m_1008_144",
        "seasonal_naive": RESULTS_DIR / "20260428_113904_seasonal_naive_m_1008_144",
        "sarima": RESULTS_DIR / "20260428_115305_sarima_m_1008_144",
        "prophet": RESULTS_DIR / "20260426_201346_prophet_m_1008_144",
        "timesfm": RESULTS_DIR / "20260426_210812_timesfm_m_1008_144",
        "ttm": RESULTS_DIR / "20260426_211057_ttm_m_1008_144",
        "chronos": RESULTS_DIR / "20260426_213125_chronos_m_1008_144",
        "moirai": RESULTS_DIR / "20260426_214426_moirai_m_1008_144",
        "gru": RESULTS_DIR / "20260426_214707_gru_m_1008_144",
        "lstm": RESULTS_DIR / "20260426_215602_lstm_m_1008_144",
        "gru_fcn": RESULTS_DIR / "20260426_221256_gru_fcn_m_1008_144",
    },
}

CONFIG_META = {
    "h_168_24": {"lookback": 168, "horizon": 24, "freq": "h", "label": "168/24 (hourly)"},
    "h_744_168": {"lookback": 744, "horizon": 168, "freq": "h", "label": "744/168 (hourly)"},
    "m_288_144": {"lookback": 288, "horizon": 144, "freq": "m", "label": "288/144 (10min)"},
    "m_1008_144": {"lookback": 1008, "horizon": 144, "freq": "m", "label": "1008/144 (10min)"},
}

CANONICAL_RUNS = EXPERIMENT_CONFIGS["h_168_24"]

BENCHMARKS = ["drift", "seasonal", "periodic_spaces", "workers", "random"]
AGG_LEVEL_ORDER = ["institutions", "subnets", "ips"]
METRICS = ["rmse", "mae", "mase", "rmsse", "smape", "r2"]
HIGHER_IS_BETTER = {"r2"}

BENCH_LABELS = {
    "drift": "Drift",
    "seasonal": "Seasonal",
    "periodic_spaces": "Periodic Spaces",
    "workers": "Workers",
    "random": "Random",
}

MODEL_LABELS = {
    "zero": "Zero",
    "mean": "Mean",
    "seasonal_naive": "Seasonal Naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "timesfm": "TimesFM",
    "ttm": "TTM",
    "chronos": "Chronos-2",
    "moirai": "Moirai",
    "gru": "GRU",
    "lstm": "LSTM",
    "gru_fcn": "GRU-FCN",
}

ID_COL = {"ips": "id_ip", "institutions": "id_institution", "subnets": "id_institution_subnet"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _compute_from_predictions(pred_csv, metrics, bench, horizon, bench_dir, m=24):
    pdf = pd.read_csv(pred_csv)
    needs_hist = bool(set(metrics) & SCALED_METRICS)
    bench_data = {}
    val_end = None
    if needs_hist:
        meta = json.loads((bench_dir / bench / "metadata.json").read_text())
        val_end = meta["val_end"]
    records = []
    for (series_id, agg_level), grp in pdf.groupby(["id", "agg_level"]):
        grp_sorted = grp.sort_values("datetime")
        actual = grp_sorted["actual"].fillna(0).values
        predicted = grp_sorted["prediction"].values
        n_windows = math.ceil(len(actual) / horizon)
        y_hist = np.array([])
        if needs_hist:
            if agg_level not in bench_data:
                bench_csv = bench_dir / bench / f"{agg_level}.csv"
                bench_data[agg_level] = pd.read_csv(bench_csv) if bench_csv.exists() else None
            bdf = bench_data[agg_level]
            if bdf is not None:
                id_col = ID_COL[agg_level]
                series_rows = bdf[bdf[id_col].astype(int) == int(series_id)].sort_values("datetime")
                y_hist = series_rows["n_bytes"].fillna(0).values[:val_end]
        row = {"id": series_id, "agg_level": agg_level}
        row.update(
            calculate_metrics(actual, predicted, metrics, y_hist_base=y_hist, horizon=horizon, n_windows=n_windows, m=m)
        )
        records.append(row)
    return pd.DataFrame(records)


def load_all_results(runs=None, metrics=None):
    if runs is None:
        runs = CANONICAL_RUNS
    if metrics is None:
        metrics = list(METRICS)
    records = []
    for model, run_dir in runs.items():
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        run_config = json.loads(config_path.read_text()) if config_path.exists() else {}
        horizon = run_config.get("horizon", 24)
        freq = run_config.get("freq", "h")
        bench_dir = BENCH_BASE_DIR / FREQ_DIR.get(freq, "hourly")
        m = SEASONAL_PERIOD.get(freq, 24)
        run_subdirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
        for bench in BENCHMARKS:
            frames = []
            for rd in run_subdirs:
                metrics_csv = rd / f"{bench}_metrics.csv"
                pred_csv = rd / f"{bench}_predictions.csv"
                mdf = pd.read_csv(metrics_csv) if metrics_csv.exists() else None
                missing = [mt for mt in metrics if mdf is None or mt not in mdf.columns]
                if not missing:
                    frames.append(mdf[["id", "agg_level"] + metrics])
                elif pred_csv.exists():
                    computed = _compute_from_predictions(pred_csv, missing, bench, horizon, bench_dir, m=m)
                    if mdf is not None:
                        merged = mdf.merge(computed, on=["id", "agg_level"], how="left")
                        frames.append(merged[["id", "agg_level"] + metrics])
                        merged.to_csv(metrics_csv, index=False)
                    else:
                        available = ["id", "agg_level"] + [mt for mt in metrics if mt in computed.columns]
                        frames.append(computed[available])
                elif mdf is not None:
                    available = ["id", "agg_level"] + [mt for mt in metrics if mt in mdf.columns]
                    frames.append(mdf[available])
            if not frames:
                continue
            avg = pd.concat(frames).groupby(["id", "agg_level"], as_index=False).mean(numeric_only=True)
            avg["model"] = model
            avg["benchmark"] = bench
            records.append(avg)
    if not records:
        raise RuntimeError("No results found")
    df = pd.concat(records, ignore_index=True)
    present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    df["model"] = pd.Categorical(df["model"], categories=present, ordered=True)
    df["benchmark"] = pd.Categorical(df["benchmark"], categories=BENCHMARKS, ordered=True)
    df["agg_level"] = pd.Categorical(df["agg_level"], categories=AGG_LEVEL_ORDER, ordered=True)
    return df


def load_all_results_multi_config(configs=None, metrics=None):
    if configs is None:
        configs = EXPERIMENT_CONFIGS
    frames = []
    for config_name, runs in configs.items():
        if not runs:
            continue
        df = load_all_results(runs=runs, metrics=metrics)
        df["config"] = config_name
        df["horizon_label"] = CONFIG_META[config_name]["label"]
        frames.append(df)
    if not frames:
        raise RuntimeError("No results found in any config")
    return pd.concat(frames, ignore_index=True)


def load_timing_data(runs=None):
    if runs is None:
        runs = CANONICAL_RUNS
    records = []
    for model, run_dir in runs.items():
        summary = run_dir / "summary.csv"
        if not summary.exists():
            continue
        sdf = pd.read_csv(summary)
        row = sdf[sdf["bench_type"] == "all"]
        if row.empty:
            row = sdf.mean(numeric_only=True).to_frame().T
        records.append(
            {
                "model": model,
                "tune_time": row["tune_time_per_100pts"].values[0] if "tune_time_per_100pts" in row.columns else 0,
                "predict_time": (
                    row["prediction_time_per_100pts"].values[0] if "prediction_time_per_100pts" in row.columns else 0
                ),
            }
        )
    df = pd.DataFrame(records)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ranking & aggregation
# ---------------------------------------------------------------------------


def compute_average_rank(df, metric="mase"):
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    pivot = df.pivot_table(index=["id", "agg_level", "benchmark"], columns="model", values=metric)
    pivot = pivot[[m for m in models if m in pivot.columns]]

    def rank_row(row):
        valid_mask = row.notna()
        if valid_mask.sum() == 0:
            return row
        ranks = pd.Series(np.nan, index=row.index)
        ranks[valid_mask] = rankdata(row[valid_mask], method="average")
        return ranks

    ranks_df = pivot.apply(rank_row, axis=1)

    bench_avg = {}
    for bench in BENCHMARKS:
        mask = ranks_df.index.get_level_values("benchmark") == bench
        bench_rows = ranks_df[mask]
        bench_avg[bench] = bench_rows.mean()

    bench_avg_df = pd.DataFrame(bench_avg, index=pivot.columns)

    result = bench_avg_df.copy()
    result.index = result.index.astype(str)
    result.index.name = "model"
    result = result.reset_index()
    result["avg_rank"] = bench_avg_df.mean(axis=1).values
    result = result.sort_values("avg_rank").reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def compute_overall_ranking(df):
    metric_cols = [
        c for c in df.columns if c not in {"id", "agg_level", "model", "benchmark", "config", "horizon_label"}
    ]
    rows = []
    for model in [m for m in MODEL_ORDER if m in df["model"].unique()]:
        mdf = df[df["model"] == model]
        row = {"model": model}
        for metric in metric_cols:
            row[f"{metric}_mean"] = mdf[metric].mean()
            row[f"{metric}_median"] = mdf[metric].median()
            row[f"{metric}_std"] = mdf[metric].std()
        rows.append(row)
    sort_col = "mase_mean" if "mase" in metric_cols else f"{metric_cols[0]}_mean"
    ranking = pd.DataFrame(rows).sort_values(sort_col).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking


def compute_per_benchmark_pivot(df, metric="mase", aggfunc="mean"):
    return df.pivot_table(
        index="model",
        columns="benchmark",
        values=metric,
        aggfunc=aggfunc,
        observed=True,
    ).reindex(index=[m for m in MODEL_ORDER if m in df["model"].unique()])


def compute_pairwise_wins(df, metric="mase"):
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    pivot = df.pivot_table(index=["id", "agg_level", "benchmark"], columns="model", values=metric)
    series = {m: pivot[m] for m in models if m in pivot.columns}
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for ma in models:
        for mb in models:
            if ma == mb:
                matrix.loc[ma, mb] = np.nan
                continue
            if ma not in series or mb not in series:
                matrix.loc[ma, mb] = np.nan
                continue
            shared = series[ma].notna() & series[mb].notna()
            if shared.sum() == 0:
                matrix.loc[ma, mb] = np.nan
                continue
            a, b = series[ma][shared], series[mb][shared]
            if metric in HIGHER_IS_BETTER:
                wins = (a > b).sum() + 0.5 * (a == b).sum()
            else:
                wins = (a < b).sum() + 0.5 * (a == b).sum()
            matrix.loc[ma, mb] = 100.0 * wins / shared.sum()
    return matrix


def compute_pairwise_skill(df, metric="mase", clip_lo=1e-2, clip_hi=100.0):
    from scipy.stats import gmean as _gmean

    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    pivot = df.pivot_table(index=["id", "agg_level", "benchmark"], columns="model", values=metric)
    series = {m: pivot[m] for m in models if m in pivot.columns}
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for mj in models:
        for mk in models:
            if mj == mk:
                matrix.loc[mj, mk] = np.nan
                continue
            if mj not in series or mk not in series:
                matrix.loc[mj, mk] = np.nan
                continue
            shared = series[mj].notna() & series[mk].notna() & (series[mk] > 0)
            if shared.sum() == 0:
                matrix.loc[mj, mk] = np.nan
                continue
            ej, ek = series[mj][shared].values, series[mk][shared].values
            ratios = np.clip(ej / ek, clip_lo, clip_hi)
            matrix.loc[mj, mk] = 1.0 - _gmean(ratios)
    return matrix


def compute_series_wins(df, metric="mase"):
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    pivot = df.pivot_table(index=["id", "agg_level", "benchmark"], columns="model", values=metric)
    winner = pivot.idxmax(axis=1) if metric in HIGHER_IS_BETTER else pivot.idxmin(axis=1)
    rows = {}
    for model in models:
        row = {}
        for bench in BENCHMARKS:
            mask = winner.index.get_level_values("benchmark") == bench
            bw = winner[mask].dropna()
            total = len(bw)
            wins = int((bw == model).sum())
            row[BENCH_LABELS[bench]] = f"{wins}/{total}" if total > 0 else "-"
        rows[MODEL_LABELS.get(model, model)] = row
    return pd.DataFrame(rows).T


def compute_model_agreement(df, metric="mase"):
    pivot = df.pivot_table(index=["id", "agg_level", "benchmark"], columns="model", values=metric)
    return pivot.corr(method="spearman")


# ---------------------------------------------------------------------------
# Prediction browser (Plotly -- for notebook use)
# ---------------------------------------------------------------------------


def find_predictions(model, bench, runs=None):
    if runs is None:
        runs = CANONICAL_RUNS
    if model in runs:
        run_dir = runs[model]
        for rd in sorted(run_dir.iterdir()):
            if rd.is_dir() and rd.name.startswith("run_"):
                pred = rd / f"{bench}_predictions.csv"
                if pred.exists():
                    return pred
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.endswith(f"_{model}"):
            continue
        for rd in sorted(d.iterdir()):
            if rd.is_dir() and rd.name.startswith("run_"):
                pred = rd / f"{bench}_predictions.csv"
                if pred.exists():
                    return pred
    return None


def list_available_predictions():
    available = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        model = d.name.split("_")[-1]
        for rd in d.iterdir():
            if not rd.is_dir() or not rd.name.startswith("run_"):
                continue
            for pred in rd.glob("*_predictions.csv"):
                bench = pred.stem.replace("_predictions", "")
                available.append(
                    {
                        "model": model,
                        "benchmark": bench,
                        "path": pred,
                        "run": d.name,
                    }
                )
    return pd.DataFrame(available)


def plot_predictions(bench, model, series_id, agg_level="ips", runs=None, freq="h"):
    pred_path = find_predictions(model, bench, runs)
    if pred_path is None:
        raise FileNotFoundError(
            f"No predictions for {model}/{bench}. " f"Use list_available_predictions() to see what's available."
        )

    pdf = pd.read_csv(pred_path, parse_dates=["datetime"])
    test = (
        pdf[(pdf["id"] == series_id) & (pdf["agg_level"] == agg_level)]
        .sort_values("datetime")
        .drop_duplicates("datetime")
    )
    if test.empty:
        raise ValueError(f"No predictions for series {series_id} ({agg_level}) " f"in {model}/{bench}")

    bench_dir = BENCH_BASE_DIR / FREQ_DIR.get(freq, "hourly")
    id_col = ID_COL.get(agg_level, "id_ip")
    meta = json.loads((bench_dir / bench / "metadata.json").read_text())
    full = pd.read_csv(bench_dir / bench / f"{agg_level}.csv", parse_dates=["datetime"])
    full["datetime"] = full["datetime"].dt.tz_localize(None)
    series = full[full[id_col] == series_id].sort_values("datetime").reset_index(drop=True)

    train = series.iloc[: meta["train_end"]]
    val = series.iloc[meta["train_end"] : meta["val_end"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train["datetime"],
            y=train["n_bytes"],
            name="Train",
            line=dict(color="#94a3b8", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=val["datetime"],
            y=val["n_bytes"],
            name="Val",
            line=dict(color="#a78bfa", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test["datetime"],
            y=test["actual"],
            name="Actual",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test["datetime"],
            y=test["prediction"],
            name="Prediction",
            line=dict(color="#ea580c", width=2),
        )
    )
    fig.update_layout(
        title=(
            f"{MODEL_LABELS.get(model, model)} | "
            f"{BENCH_LABELS.get(bench, bench)} | "
            f"id={series_id} ({agg_level})"
        ),
        hovermode="x unified",
        height=500,
        xaxis_title="Datetime",
        yaxis_title="n_bytes",
    )
    return fig
