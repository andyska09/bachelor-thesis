#!/usr/bin/env python3
"""
Generate all analysis graphs and tables.

Usage (from project root):
    python -m result_exploration.generate
    python -m result_exploration.generate --metrics rmse mase
"""

import argparse

from result_exploration.analysis import (
    load_all_results,
    load_timing_data,
    compute_average_rank,
    compute_overall_ranking,
    compute_per_benchmark_pivot,
    compute_series_wins,
    EXPERIMENT_CONFIGS,
    METRICS,
    GRAPHS_DIR,
)
from result_exploration.plots import (
    plot_boxplots_by_model,
    plot_boxplots_by_benchmark,
    plot_series_wins,
    plot_pairwise_heatmap,
    plot_agg_level_breakdown,
    plot_model_agreement,
    plot_radar_chart,
    plot_speed_comparison,
    plot_histogram_overall,
    plot_histogram_per_benchmark,
)

def generate_tables(df, base_dir, metrics):
    tables_dir = base_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ranking = compute_overall_ranking(df)
    ranking.to_csv(tables_dir / "overall_ranking.csv", index=False)
    print(f"Saved: {tables_dir / 'overall_ranking.csv'}")

    for metric in metrics:
        pivot = compute_per_benchmark_pivot(df, metric)
        pivot.to_csv(tables_dir / f"per_benchmark_{metric}.csv")
        print(f"Saved: {tables_dir / f'per_benchmark_{metric}.csv'}")

        pivot_median = compute_per_benchmark_pivot(df, metric, aggfunc="median")
        pivot_median.to_csv(tables_dir / f"per_benchmark_{metric}_median.csv")
        print(f"Saved: {tables_dir / f'per_benchmark_{metric}_median.csv'}")

        sw = compute_series_wins(df, metric)
        sw.to_csv(tables_dir / f"series_wins_{metric}.csv")
        print(f"Saved: {tables_dir / f'series_wins_{metric}.csv'}")

    avg_rank = compute_average_rank(df, "mase")
    avg_rank.to_csv(tables_dir / "average_rank_mase.csv", index=False)
    print(f"Saved: {tables_dir / 'average_rank_mase.csv'}")


def generate_plots(df, base_dir, metrics):
    for metric in metrics:
        print(f"\n  [{metric.upper()}]")
        plot_boxplots_by_model(df, metric, base_dir)
        plot_boxplots_by_benchmark(df, metric, base_dir)
        plot_series_wins(df, metric, base_dir)
        plot_pairwise_heatmap(df, metric, base_dir)
        plot_agg_level_breakdown(df, metric, base_dir)
        plot_model_agreement(df, metric, base_dir)
        plot_radar_chart(df, metric, base_dir)
        plot_histogram_overall(df, metric, base_dir)
        plot_histogram_per_benchmark(df, metric, base_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS, metavar="METRIC")
    args = parser.parse_args()
    metrics = args.metrics

    for config_name, runs in EXPERIMENT_CONFIGS.items():
        if not runs:
            print(f"\nSkipping {config_name} (no runs configured)")
            continue

        print(f"\n=== {config_name} ===")
        base_dir = GRAPHS_DIR / config_name

        print("Loading results...")
        df = load_all_results(runs=runs)
        print(f"Loaded {len(df)} series-level records, {df['model'].nunique()} models.")

        print("\n--- Tables ---")
        generate_tables(df, base_dir, metrics=metrics)

        print("\n--- Per-metric plots ---")
        generate_plots(df, base_dir, metrics=metrics)

        print("\n--- Speed comparison ---")
        timing = load_timing_data(runs=runs)
        plot_speed_comparison(timing, base_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
