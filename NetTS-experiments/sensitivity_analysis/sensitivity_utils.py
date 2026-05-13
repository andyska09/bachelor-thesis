import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scoring.runner import scores_path


def load_scores(aggregation, benchmark, sources):
    scores = {}
    for source, label in sources:
        path = scores_path(aggregation, benchmark, source)
        scores[label] = pd.read_csv(path)
        print(f"  {label}: {len(scores[label])} series")
    return scores


def sweep_thresholds(scores_dict, metric_col, thresholds, metric_op=">="):
    rows = []
    for t in thresholds:
        row = {"threshold": round(t, 2)}
        for label, df in scores_dict.items():
            vals = df[metric_col].dropna()
            pool = vals[vals >= t] if metric_op == ">=" else vals[vals <= t]
            row[f"{label} pool"] = len(pool)
            row[f"{label} median"] = (
                round(pool.median(), 3) if len(pool) > 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sweep(
    sweep_df,
    title_prefix,
    threshold,
    sample_size,
    xlabel,
    metric_col,
    save_path=None,
    log_y=False,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    pool_cols = [c for c in sweep_df.columns if c.endswith(" pool")]
    labels = [c.replace(" pool", "") for c in pool_cols]

    for i, label in enumerate(labels):
        ax1.plot(
            sweep_df["threshold"],
            sweep_df[f"{label} pool"],
            marker="o",
            color=f"C{i}",
            linewidth=2,
            label=label,
        )
        ax2.plot(
            sweep_df["threshold"],
            sweep_df[f"{label} median"],
            marker="s",
            color=f"C{i}",
            linewidth=2,
            label=label,
        )

    ax1.axvline(threshold, color="red", linestyle="--", label=f"chosen = {threshold}")
    ax1.axhline(
        sample_size,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"sample size = {sample_size}",
    )
    ax1.set_ylabel("Pool size")
    ax1.set_title(f"{title_prefix} — Pool size vs. threshold")
    if log_y:
        ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axvline(threshold, color="red", linestyle="--", label=f"chosen = {threshold}")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(f"Median {metric_col} in pool")
    ax2.set_title(f"{title_prefix} — Median {metric_col} vs. threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def print_sparsity_check(df, metric_col, threshold, sparsity_min):
    pool = df[df[metric_col] >= threshold]
    both = pool[pool["ratio_active"] >= sparsity_min]
    removed = len(pool) - len(both)
    pct = removed / len(pool) * 100 if len(pool) else 0.0

    print(f"Total series:                          {len(df):>7,}")
    print(f"{metric_col} >= {threshold} alone:                       {len(pool):>7,}")
    print(
        f"{metric_col} >= {threshold} AND sparsity >= {sparsity_min}:        {len(both):>7,}"
    )
    print(
        f"Removed by sparsity (on top of {metric_col}):   {removed:>7,}  ({pct:.1f}% of pool)"
    )
    print()
    if pct < 5:
        print(
            f"Sparsity is essentially non-binding: it removes <5% of {metric_col} qualifiers."
        )
        print(
            f"{metric_col} does the heavy lifting; sparsity acts as a soft guardrail."
        )
    else:
        print(f"Sparsity is the dominant filter: it shrinks the {metric_col}-only pool")
        print(f"from {len(pool):,} down to {len(both):,} ({pct:.1f}% removed).")
