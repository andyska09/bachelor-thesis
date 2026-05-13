import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from result_exploration.analysis import (
    BENCHMARKS, AGG_LEVEL_ORDER,
    BENCH_LABELS, MODEL_LABELS, GRAPHS_DIR,
    compute_pairwise_wins, compute_model_agreement, compute_average_rank,
)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

MASE_CLIP = 4.0

METRIC_CONFIG = {
    "mase": {"label": "MASE", "ref_line": 1.0, "clip": MASE_CLIP},
    "rmse": {"label": "RMSE", "ref_line": None, "clip": None},
    "mae": {"label": "MAE", "ref_line": None, "clip": None},
    "smape": {"label": "SMAPE (%)", "ref_line": None, "clip": None},
    "rmsse": {"label": "RMSSE", "ref_line": 1.0, "clip": MASE_CLIP},
    "r2": {"label": "R²", "ref_line": 0.0, "clip": None},
}


def _output_dir(metric, base_dir=None):
    d = (base_dir or GRAPHS_DIR) / metric
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cfg(metric):
    return METRIC_CONFIG[metric]


def _model_labels(models):
    return [MODEL_LABELS.get(m, m) for m in models]


def _bench_labels(benchmarks):
    return [BENCH_LABELS.get(b, b) for b in benchmarks]


def _prepare(df, metric):
    cfg = _cfg(metric)
    sub = df.copy()
    if cfg["clip"] is not None:
        sub[metric] = sub[metric].clip(upper=cfg["clip"])
    return sub


def _add_ref_line(ax, metric):
    cfg = _cfg(metric)
    if cfg["ref_line"] is not None:
        ax.axhline(cfg["ref_line"], color="red", linewidth=0.8,
                    linestyle="--", alpha=0.7)


def _set_ylim(ax, metric):
    cfg = _cfg(metric)
    if cfg["clip"] is not None:
        ax.set_ylim(0, cfg["clip"])


# ---------------------------------------------------------------------------
# 1. Boxplots by model, faceted by benchmark
# ---------------------------------------------------------------------------

def plot_boxplots_by_model(df, metric="mase", base_dir=None):
    sub = _prepare(df, metric)
    cfg = _cfg(metric)
    benchmarks = [b for b in BENCHMARKS if b in sub["benchmark"].cat.categories]
    models = list(sub["model"].cat.categories)
    palette = sns.color_palette("tab10", len(models))

    fig, axes = plt.subplots(
        1, len(benchmarks), figsize=(6 * len(benchmarks), 7), sharey=True
    )
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, bench in zip(axes, benchmarks):
        bsub = sub[sub["benchmark"] == bench]
        sns.boxplot(
            data=bsub, x="model", y=metric, hue="model",
            order=models, hue_order=models, palette=palette,
            showfliers=True,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
            linewidth=0.8, legend=False, ax=ax,
        )
        _add_ref_line(ax, metric)
        _set_ylim(ax, metric)
        ax.set_title(BENCH_LABELS.get(bench, bench), fontsize=10,
                      fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(cfg["label"] if ax is axes[0] else "")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(_model_labels(models), rotation=45, ha="right",
                           fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{cfg['label']} Distribution by Model per Benchmark",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = _output_dir(metric, base_dir) / "boxplots_by_model.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 2. Boxplots by benchmark, faceted by model
# ---------------------------------------------------------------------------

def plot_boxplots_by_benchmark(df, metric="mase", base_dir=None):
    sub = _prepare(df, metric)
    cfg = _cfg(metric)
    benchmarks = [b for b in BENCHMARKS if b in sub["benchmark"].cat.categories]
    models = list(sub["model"].cat.categories)
    palette = sns.color_palette("Set2", len(benchmarks))

    fig, axes = plt.subplots(
        1, len(models), figsize=(5 * len(models), 7), sharey=True
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        msub = sub[sub["model"] == model]
        sns.boxplot(
            data=msub, x="benchmark", y=metric, hue="benchmark",
            order=benchmarks, hue_order=benchmarks, palette=palette,
            showfliers=True,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
            linewidth=0.8, legend=False, ax=ax,
        )
        _add_ref_line(ax, metric)
        _set_ylim(ax, metric)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10,
                      fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(cfg["label"] if ax is axes[0] else "")
        ax.set_xticks(range(len(benchmarks)))
        ax.set_xticklabels(_bench_labels(benchmarks), rotation=45,
                           ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{cfg['label']} Distribution by Benchmark per Model",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = _output_dir(metric, base_dir) / "boxplots_by_benchmark.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Series wins bar chart
# ---------------------------------------------------------------------------

def plot_series_wins(df, metric="mase", base_dir=None):
    cfg = _cfg(metric)
    models = list(df["model"].cat.categories)
    benchmarks = [b for b in BENCHMARKS if b in df["benchmark"].cat.categories]

    pivot = df.pivot_table(
        index=["id", "agg_level", "benchmark"], columns="model", values=metric
    )
    winner = pivot.idxmin(axis=1).reset_index()
    winner.columns = [*pivot.index.names, "winner"]

    records = []
    for bench in benchmarks:
        bw = winner[winner["benchmark"] == bench]["winner"]
        total = len(bw.dropna())
        for model in models:
            records.append({
                "benchmark": bench, "model": model,
                "wins": int((bw == model).sum()), "total": total,
            })
    wins_df = pd.DataFrame(records)

    palette = sns.color_palette("tab10", len(models))
    color_map = dict(zip(models, palette))

    fig, axes = plt.subplots(
        1, len(benchmarks), figsize=(4.5 * len(benchmarks), 5), sharey=False
    )
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, bench in zip(axes, benchmarks):
        bsub = wins_df[wins_df["benchmark"] == bench]
        bars = ax.bar(
            range(len(models)), bsub["wins"].values,
            color=[color_map[m] for m in bsub["model"]],
        )
        total = bsub["total"].iloc[0] if not bsub.empty else 0
        for bar, (_, row) in zip(bars, bsub.iterrows()):
            if row["wins"] > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(int(row["wins"])),
                    ha="center", va="bottom", fontsize=8,
                )
        ax.set_title(
            f"{BENCH_LABELS.get(bench, bench)}\n(n={total})",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("Series wins" if ax is axes[0] else "")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(_model_labels(models), rotation=45, ha="right",
                           fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Series Wins per Model per Benchmark ({cfg['label']})\n"
        f"(win = lowest {cfg['label']} on that series)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = _output_dir(metric, base_dir) / "series_wins.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 4. Pairwise win rate heatmap
# ---------------------------------------------------------------------------

def plot_pairwise_heatmap(df, metric="mase", base_dir=None, title=None, save=True):
    cfg = _cfg(metric)
    win_matrix = compute_pairwise_wins(df, metric)

    labels = _model_labels(win_matrix.index)
    display = win_matrix.copy()
    display.index = labels
    display.columns = labels

    n = len(labels)
    cell_size = 0.85
    side = max(cell_size * n + 2.5, 7)
    fig, ax = plt.subplots(figsize=(side, side))

    sns.heatmap(
        display, annot=True, fmt=".0f", annot_kws={"size": 14},
        cmap="Blues", vmin=0, vmax=100,
        linewidths=0.8, linecolor="white", square=True, ax=ax,
        cbar_kws={"label": "Win rate (%)", "shrink": 0.7},
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    desc = title or (
        f"Pairwise Win Rate (%) \u2014 {cfg['label']}\n"
        f"(cell[A,B] = % of series where row A < column B)"
    )
    print(desc)
    ax.set_title("")
    fig.tight_layout()
    if save:
        out = _output_dir(metric, base_dir) / "pairwise_heatmap.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
    return fig


# ---------------------------------------------------------------------------
# 5. Aggregation level breakdown
# ---------------------------------------------------------------------------

def plot_agg_level_breakdown(df, metric="mase", base_dir=None, title=None, save=True):
    sub = _prepare(df, metric)
    cfg = _cfg(metric)
    benchmarks = [b for b in BENCHMARKS if b in sub["benchmark"].cat.categories]
    multi = [
        b for b in benchmarks
        if sub[sub["benchmark"] == b]["agg_level"].nunique() > 1
    ]
    if not multi:
        return None

    models = list(sub["model"].cat.categories)
    agg_levels = [
        a for a in AGG_LEVEL_ORDER if a in sub["agg_level"].cat.categories
    ]
    palette = sns.color_palette("Set2", len(agg_levels))

    fig, axes = plt.subplots(
        1, len(multi), figsize=(4.5 * len(multi), 5), sharey=True
    )
    if len(multi) == 1:
        axes = [axes]

    for ax, bench in zip(axes, multi):
        bsub = sub[sub["benchmark"] == bench]
        sns.boxplot(
            data=bsub, x="model", y=metric, hue="agg_level",
            order=models, hue_order=agg_levels, palette=palette,
            showfliers=True,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
            linewidth=0.8, ax=ax,
        )
        _add_ref_line(ax, metric)
        ax.set_title(BENCH_LABELS.get(bench, bench), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(cfg["label"] if ax is axes[0] else "")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(_model_labels(models), rotation=45, ha="right",
                           fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[-1]:
            ax.legend(title="Agg level", fontsize=7, title_fontsize=7,
                      loc="upper right")
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

    fig.suptitle(
        title or f"{cfg['label']} by Aggregation Level per Benchmark",
        fontsize=12,
    )
    fig.tight_layout()
    if save:
        out = _output_dir(metric, base_dir) / "agg_level_breakdown.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
    return fig


# ---------------------------------------------------------------------------
# 6. Model agreement heatmap
# ---------------------------------------------------------------------------

def plot_model_agreement(df, metric="mase", base_dir=None):
    cfg = _cfg(metric)
    corr = compute_model_agreement(df, metric)

    labels = _model_labels(corr.index)
    display = corr.copy()
    display.index = labels
    display.columns = labels

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        display, annot=True, fmt=".2f",
        cmap="Reds", vmin=0, vmax=1,
        linewidths=0.5, ax=ax,
    )
    print(
        f"Spearman Correlation of Per-Series {cfg['label']}\n"
        f"(high = models fail on the same series)"
    )
    ax.set_title("")
    fig.tight_layout()
    out = _output_dir(metric, base_dir) / "model_agreement.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 7. Radar chart (rank per benchmark)
# ---------------------------------------------------------------------------

def plot_radar_chart(df, metric="mase", base_dir=None):
    cfg = _cfg(metric)
    models = list(df["model"].cat.categories)
    benchmarks = [b for b in BENCHMARKS if b in df["benchmark"].cat.categories]

    avg_rank = compute_average_rank(df, metric).set_index("model")
    ranks = avg_rank[[b for b in benchmarks if b in avg_rank.columns]].reindex(models)

    angles = np.linspace(0, 2 * np.pi, len(benchmarks), endpoint=False).tolist()
    angles += angles[:1]
    labels = _bench_labels(benchmarks)

    palette = [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#F0E442", "#000000", "#8B4513",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, model in enumerate(models):
        if model not in ranks.index:
            continue
        values = ranks.loc[model].tolist()
        values += values[:1]
        ax.plot(
            angles, values,
            marker=markers[i % len(markers)], linewidth=1.8, markersize=5,
            label=MODEL_LABELS.get(model, model), color=palette[i % len(palette)],
        )
        ax.fill(angles, values, alpha=0.03, color=palette[i % len(palette)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, len(models) + 0.5)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels([str(i) for i in range(1, len(models) + 1)], fontsize=7)
    ax.set_title(
        f"Model Ranking by Benchmark ({cfg['label']})\n"
        f"(closer to center = better)",
        fontsize=12, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.tight_layout()
    out = _output_dir(metric, base_dir) / "radar_chart.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 8. RMSE histograms – overall and per benchmark
# ---------------------------------------------------------------------------

def plot_histogram_overall(df, metric="rmse", base_dir=None):
    sub = df.copy()
    cfg = _cfg(metric)
    models = list(sub["model"].cat.categories)
    palette = sns.color_palette("tab10", len(models))

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()

    for i, model in enumerate(models):
        ax = axes_flat[i]
        values = sub[sub["model"] == model][metric].dropna()
        sns.histplot(values, bins=30, kde=True, color=palette[i], alpha=0.7, ax=ax)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10, fontweight="bold")
        ax.set_xlabel(cfg["label"])
        ax.set_ylabel("Count")
        ax.text(0.97, 0.97, f"n={len(values)}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(models), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Distribution of {cfg['label']} Across All Benchmarks",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = _output_dir(metric, base_dir) / "histogram_overall.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_histogram_per_benchmark(df, metric="rmse", base_dir=None):
    sub = df.copy()
    cfg = _cfg(metric)
    benchmarks = [b for b in BENCHMARKS if b in sub["benchmark"].cat.categories]
    models = list(sub["model"].cat.categories)
    palette = sns.color_palette("tab10", len(models))

    fig, axes = plt.subplots(
        len(benchmarks), len(models),
        figsize=(4 * len(models), 3.5 * len(benchmarks)),
        sharey="row",
    )
    if len(benchmarks) == 1:
        axes = [axes]

    for row_idx, bench in enumerate(benchmarks):
        bsub = sub[sub["benchmark"] == bench]
        row_axes = axes[row_idx]
        for col_idx, model in enumerate(models):
            ax = row_axes[col_idx]
            values = bsub[bsub["model"] == model][metric].dropna()
            sns.histplot(values, bins=20, kde=True, color=palette[col_idx], alpha=0.7, ax=ax)
            if row_idx == 0:
                ax.set_title(MODEL_LABELS.get(model, model), fontsize=8, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{BENCH_LABELS.get(bench, bench)}\nCount", fontsize=8)
            else:
                ax.set_ylabel("")
            ax.set_xlabel(cfg["label"] if row_idx == len(benchmarks) - 1 else "")
            ax.text(0.97, 0.97, f"n={len(values)}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Distribution of {cfg['label']} per Benchmark and Model",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = _output_dir(metric, base_dir) / "histogram_per_benchmark.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 9. Speed comparison (metric-agnostic)
# ---------------------------------------------------------------------------

def plot_speed_comparison(timing_df, base_dir=None, title=None, save=True):
    models = list(timing_df["model"].values)
    labels = _model_labels(models)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.35

    tune = timing_df["tune_time"].values.astype(float)
    pred = timing_df["predict_time"].values.astype(float)

    tune_plot = np.where(tune > 0, tune, np.nan)
    pred_plot = np.where(pred > 0, pred, np.nan)

    ax.bar(x - width / 2, tune_plot, width, label="Tune time",
           color="#6366f1", alpha=0.8)
    ax.bar(x + width / 2, pred_plot, width, label="Predict time",
           color="#f97316", alpha=0.8)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time per 100 data points (s, log scale)")
    ax.set_title(title or "Inference Speed Comparison", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save:
        out_dir = base_dir or GRAPHS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "speed_comparison.pdf"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
    return fig
