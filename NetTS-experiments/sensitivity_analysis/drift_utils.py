import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import BENCHMARKS, AGGREGATIONS
from utils import load_dataset, SOURCE_MAP
from scoring.drift import compute_rel_dev
from scoring.runner import scores_path

BENCHMARK = "DRIFT_SWEEP"
BENCH_PARAMS = BENCHMARKS[BENCHMARK]


def load_sweep(aggregation, sources, sparsity_min=None):
    if sparsity_min is None:
        sparsity_min = BENCH_PARAMS["sparsity_min"]
    out = {}
    for source, label in sources:
        _, id_col = SOURCE_MAP[source]
        path = scores_path(aggregation, BENCHMARK, source)
        df = pd.read_csv(path)
        if df[id_col].dtype == object:
            df[id_col] = df[id_col].str.strip('() ,"').astype(int)
        n_total = len(df)
        df = df[df["ratio_active"] >= sparsity_min].reset_index(drop=True)
        print(f"  {label}: {len(df)}/{n_total} pass sparsity >= {sparsity_min}")
        out[label] = df
    return out


def sweep_dev(dfs_by_label, persistence_days, dev_sweep=None):
    if dev_sweep is None:
        dev_sweep = BENCH_PARAMS["dev_sweep"]
    rows = []
    for t in dev_sweep:
        key = f"t{int(round(t * 100))}p{persistence_days}"
        row = {"threshold": round(t, 2)}
        for label, df in dfs_by_label.items():
            qual = df[f"{key}_d"]
            row[f"{label} pool"] = int(qual.sum())
            row[f"{label} events"] = int(df.loc[qual, f"{key}_n"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_persist(dfs_by_label, threshold, persist_sweep=None):
    if persist_sweep is None:
        persist_sweep = BENCH_PARAMS["persistence_sweep_days"]
    rows = []
    for p in persist_sweep:
        key = f"t{int(round(threshold * 100))}p{p}"
        row = {"persistence_days": p}
        for label, df in dfs_by_label.items():
            qual = df[f"{key}_d"]
            row[f"{label} pool"] = int(qual.sum())
            row[f"{label} events"] = int(df.loc[qual, f"{key}_n"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_drift_sweep(sweep_df, x_col, title_prefix, chosen_val, sample_size, xlabel, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    pool_cols = [c for c in sweep_df.columns if c.endswith(" pool")]
    labels = [c.replace(" pool", "") for c in pool_cols]

    for i, label in enumerate(labels):
        ax1.plot(sweep_df[x_col], sweep_df[f"{label} pool"],
                 marker="o", color=f"C{i}", linewidth=2, label=label)
        ax2.plot(sweep_df[x_col], sweep_df[f"{label} events"],
                 marker="s", color=f"C{i}", linewidth=2, label=label)

    ax1.axvline(chosen_val, color="red", linestyle="--", label=f"chosen = {chosen_val}")
    ax1.axhline(sample_size, color="gray", linestyle=":", alpha=0.7, label=f"sample size = {sample_size}")
    ax1.set_ylabel("Pool size (drift_in_test)")
    ax1.set_yscale("log")
    ax1.set_title(f"{title_prefix} — Pool size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axvline(chosen_val, color="red", linestyle="--", label=f"chosen = {chosen_val}")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Total drift events in pool")
    ax2.set_yscale("log")
    ax2.set_title(f"{title_prefix} — Drift events captured")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_sp_survival(dfs_by_label, persistence_days, threshold, restrict_to_test=False, save_path=None):
    suffix = "_test" if restrict_to_test else ""
    n = len(dfs_by_label)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (label, df) in zip(axes, dfs_by_label.items()):
        n_total = len(df)
        for i, p in enumerate(persistence_days):
            vals = np.sort(df[f"S_p{p}{suffix}"].dropna().values)
            xs = np.concatenate([[0.0], vals])
            ys = np.concatenate([[1.0], 1 - np.arange(1, len(vals) + 1) / len(vals)])
            ax.step(xs, ys, where="post", linewidth=2, color=f"C{i}",
                    label=f"p = {p} d")
        ax.axvline(threshold, color="red", linestyle="--", alpha=0.7,
                   label=f"τ = {threshold}")
        ax.set_title(f"{label} (n={n_total})")
        ax.set_xlabel("τ")
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    metric = "S_p^test" if restrict_to_test else "S_p"
    axes[0].set_ylabel(f"P({metric} ≥ τ)")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def event_timestamps_for_ids(source, aggregation, ts_ids, threshold, persistence_days):
    if not ts_ids:
        return []
    agg_params = AGGREGATIONS[aggregation]
    bench_params = {**BENCH_PARAMS, "threshold": threshold, "min_drift_days": persistence_days}
    scale = agg_params["daily_period"] / 24
    min_drift_steps = int(persistence_days * 24 * scale)

    data = load_dataset(source,
                        aggregation=agg_params["enum"],
                        time_range=agg_params["time_range"],
                        ts_ids=ts_ids)
    df = data["df"]
    id_col = data["id_col"]

    events = []
    for ts_id, grp in df.groupby(id_col):
        rel_dev, _, _ = compute_rel_dev(grp, agg_params, bench_params)
        if rel_dev is None:
            continue
        mask = rel_dev > threshold
        sustained = mask.rolling(min_drift_steps, min_periods=min_drift_steps).mean() == 1.0
        starts = sustained & ~sustained.shift(1, fill_value=False)
        for idx in starts[starts].index:
            dt = pd.Timestamp(grp.loc[idx, "datetime"])
            iso = dt.isocalendar()
            events.append({"ts_id": int(ts_id), "date": dt,
                           "year": int(iso[0]), "week": int(iso[1])})
    return events


def calendar_week_analysis(sources, aggregation, threshold=None, min_drift_days=None, save_path=None):
    if threshold is None:
        threshold = BENCH_PARAMS["threshold"]
    if min_drift_days is None:
        min_drift_days = BENCH_PARAMS["min_drift_days"]

    selected = pd.read_csv(f"selected_ids/{aggregation}/DRIFT.csv")
    selected_by_level = {level: grp["ts_id"].astype(int).tolist()
                         for level, grp in selected.groupby("level")}

    all_events = []
    n_total_selected = 0
    for source, label in sources:
        ids = selected_by_level.get(label, [])
        n_total_selected += len(ids)
        print(f"  {label}: {len(ids)} selected (from selected_ids/{aggregation}/DRIFT.csv) — loading their data...")
        evts = event_timestamps_for_ids(source, aggregation, ids, threshold, min_drift_days)
        for e in evts:
            e["level"] = label
        all_events.extend(evts)

    events_df = pd.DataFrame(all_events)
    if events_df.empty:
        print("  No drift events found among selected series.")
        return events_df

    unique_weeks = events_df.groupby(["year", "week"]).ngroups
    span_weeks = int(np.ceil((events_df["date"].max() - events_df["date"].min()).days / 7)) + 1

    print(f"\n  Total drift events across selected series: {len(events_df)}")
    print(f"  Distinct series with >=1 event: {events_df['ts_id'].nunique()}")
    print(f"  Distinct calendar weeks with >=1 event: {unique_weeks}")
    print(f"  Date range: {events_df['date'].min().date()} to {events_df['date'].max().date()} ({span_weeks} weeks span)")

    week_counts = events_df.groupby(["year", "week"]).size().reset_index(name="events")
    week_counts["label"] = week_counts.apply(lambda r: f"{int(r['year'])}-W{int(r['week']):02d}", axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(week_counts)), week_counts["events"], color="C0", edgecolor="black", alpha=0.7)
    ax.set_xticks(range(len(week_counts)))
    ax.set_xticklabels(week_counts["label"], rotation=45, ha="right")
    ax.set_xlabel("Calendar week")
    ax.set_ylabel("Drift events")
    agg_label = aggregation.replace("10min", "10-Minute").replace("hourly", "Hourly")
    ax.set_title(f"{agg_label} — Drift events per calendar week ({n_total_selected} selected series)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"\n  Events per level:")
    for source, label in sources:
        sub = events_df[events_df["level"] == label]
        if sub.empty:
            print(f"    {label}: 0 events")
            continue
        n_weeks = sub.groupby(["year", "week"]).ngroups
        print(f"    {label}: {len(sub)} events across {n_weeks} distinct weeks")

    return events_df
