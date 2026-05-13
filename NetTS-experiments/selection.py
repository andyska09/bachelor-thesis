import argparse
import os

import pandas as pd

from config import AGGREGATIONS, BENCHMARKS, RANDOM_SEED
from utils import SOURCE_MAP
from scoring.runner import scores_path

SELECTED_DIR = "selected_ids"


def _filter_pool(scores_df, bench_params):
    col = bench_params["metric_col"]
    op = bench_params["metric_op"]
    thr = bench_params["threshold"]

    if op == ">=":
        return scores_df[scores_df[col] >= thr].copy()
    elif op == "<=":
        return scores_df[scores_df[col] <= thr].copy()
    elif op == "custom":
        return scores_df[scores_df["has_drift"] & scores_df["drift_in_test"]].copy()
    else:
        raise ValueError(f"Unknown metric_op: {op}")


def select_source(benchmark, source, aggregation="hourly"):
    bench_params = BENCHMARKS[benchmark]
    sample_size = bench_params["sample_size"]
    _, id_col = SOURCE_MAP[source]

    path = scores_path(aggregation, benchmark, source)
    scores_df = pd.read_csv(path)
    if scores_df[id_col].dtype == object:
        scores_df[id_col] = scores_df[id_col].str.strip('() ,"').astype(int)
    total = len(scores_df)

    sparsity_min = bench_params["sparsity_min"]
    sparsity_max = bench_params.get("sparsity_max", None)
    if sparsity_max is not None:
        active = scores_df[
            (scores_df["ratio_active"] >= sparsity_min)
            & (scores_df["ratio_active"] <= sparsity_max)
        ]
    else:
        active = scores_df[scores_df["ratio_active"] >= sparsity_min]
    print(f"  {source}: {len(active)}/{total} pass sparsity", end="")

    pool = _filter_pool(active, bench_params)
    pool = pool.sort_values(id_col).reset_index(
        drop=True
    )  # to make random sampling consistent with previous approach
    print(f", pool={len(pool)}", end="")

    if len(pool) == 0:
        print(" — empty pool!")
        return [], len(pool), scores_df

    if len(pool) < sample_size:
        print(f" — pool exhausted, taking all {len(pool)}")
        ids = sorted(pool[id_col].astype(int).tolist())
        return ids, len(pool), scores_df

    sampled = pool.sample(n=sample_size, random_state=RANDOM_SEED)
    ids = sorted(sampled[id_col].astype(int).tolist())
    print(f" — selected {len(ids)}")
    return ids, len(pool), scores_df


def save_selection(selected, benchmark, aggregation):
    out_dir = os.path.join(SELECTED_DIR, aggregation)
    os.makedirs(out_dir, exist_ok=True)

    level_names = {
        "institutions": "Institutions",
        "subnets": "Subnets",
        "ips": "IPs",
        "ips_full": "IPs",
    }
    rows = []
    for source, ids in selected.items():
        level = level_names.get(source, source)
        for ts_id in ids:
            rows.append({"level": level, "ts_id": ts_id})

    out_path = os.path.join(out_dir, f"{benchmark}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(rows)} series)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Select benchmark series from pre-computed scores"
    )
    parser.add_argument(
        "--aggregation", "-a", required=True, choices=list(AGGREGATIONS)
    )
    parser.add_argument("--benchmark", "-b", required=True, choices=list(BENCHMARKS))
    parser.add_argument(
        "--source", "-s", required=True, nargs="+", choices=list(SOURCE_MAP)
    )
    args = parser.parse_args()

    bench = args.benchmark
    bench_params = BENCHMARKS[bench]
    selected = {}
    pool_sizes = {}

    print(f"Selecting: {bench} / {args.aggregation}")
    print(
        f"Threshold: {bench_params['metric_col']} {bench_params['metric_op']} {bench_params['threshold']}"
    )
    print(f"Sample size: {bench_params['sample_size']} | Seed: {RANDOM_SEED}")
    print()

    for source in args.source:
        path = scores_path(args.aggregation, bench, source)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {source}")
            print(
                f"    Run: python -m scoring.runner -a {args.aggregation} -b {bench} -s {source}"
            )
            continue
        ids, pool_size, _ = select_source(bench, source, args.aggregation)
        selected[source] = ids
        pool_sizes[source] = pool_size

    print()
    for source, ids in selected.items():
        print(f"  {source} ({len(ids)}): {ids}")
        print(f"    pool: {pool_sizes[source]}")

    save_selection(selected, bench, args.aggregation)


if __name__ == "__main__":
    main()
