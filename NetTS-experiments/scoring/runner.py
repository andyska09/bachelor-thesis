import argparse
import glob
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
from cesnet_tszoo.configs import TimeBasedConfig
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import DatasetType

from config import AGGREGATIONS, BENCHMARKS, RANDOM_SEED, FEATURE
from utils import load_dataset, SOURCE_MAP
from scoring import SCORE_FUNCTIONS

SCORES_DIR = "scores"
CHUNK_SIZE = 6000


def scores_path(aggregation, benchmark, source):
    return os.path.join(SCORES_DIR, aggregation, f"{benchmark}_{source}.csv")


def _load_non_work_days(dataset_obj, agg_params):
    wh = dataset_obj.get_additional_data("weekends_and_holidays").copy()
    wh["day"] = pd.to_datetime(wh["Date"]).dt.normalize()
    return set(wh["day"])


def _score_one(args):
    grp, score_fn, id_col, agg_params, bench_params, extra_kwargs = args
    return score_fn(grp, id_col, agg_params, bench_params, **extra_kwargs)


def score_source(
    benchmark,
    source,
    aggregation="hourly",
    data_root="../data/",
    shard=None,
    num_shards=None,
    workers=1,
):
    bench_params = BENCHMARKS[benchmark]
    agg_params = AGGREGATIONS[aggregation]
    score_fn = SCORE_FUNCTIONS[benchmark]
    _, id_col = SOURCE_MAP[source]

    out_path = scores_path(aggregation, benchmark, source)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if source == "ips_full":
        return _score_chunked(
            benchmark,
            source,
            aggregation,
            data_root,
            shard=shard,
            num_shards=num_shards,
            workers=workers,
        )

    data = load_dataset(
        source,
        aggregation=agg_params["enum"],
        data_root=data_root,
        time_range=agg_params["time_range"],
    )
    df = data["df"]
    ids = data["ids"]

    extra_kwargs = {}
    if bench_params.get("needs_calendar"):
        non_work_days = _load_non_work_days(data["dataset"], agg_params)
        extra_kwargs["non_work_days"] = non_work_days

    tasks = [
        (
            df[df[id_col] == ts_id],
            score_fn,
            id_col,
            agg_params,
            bench_params,
            extra_kwargs,
        )
        for ts_id in ids
    ]

    if workers > 1:
        with Pool(workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(_score_one, tasks),
                    total=len(tasks),
                    desc=f"{benchmark}/{source}",
                )
            )
    else:
        rows = [_score_one(t) for t in tqdm(tasks, desc=f"{benchmark}/{source}")]

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(scores_df)} series)")
    return scores_df


def _score_chunked(
    benchmark, source, aggregation, data_root, shard=None, num_shards=None, workers=1
):
    bench_params = BENCHMARKS[benchmark]
    agg_params = AGGREGATIONS[aggregation]
    score_fn = SCORE_FUNCTIONS[benchmark]
    source_type, id_col = SOURCE_MAP[source]

    out_path = scores_path(aggregation, benchmark, source)
    if shard is not None:
        partial_path = out_path.replace(".csv", f"_partial_s{shard}.csv")
    else:
        partial_path = out_path.replace(".csv", "_partial.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dataset_obj = CESNET_TimeSeries24.get_dataset(
        data_root=data_root,
        source_type=source_type,
        aggregation=agg_params["enum"],
        dataset_type=DatasetType.TIME_BASED,
    )
    indices = dataset_obj.get_available_ts_indices()
    all_ids = indices[indices.dtype.names[0]].tolist()
    print(f"Total IDs: {len(all_ids)}")

    if shard is not None and num_shards is not None:
        shard_size = len(all_ids) // num_shards
        start = shard * shard_size
        end = start + shard_size if shard < num_shards - 1 else len(all_ids)
        all_ids = all_ids[start:end]
        print(
            f"Shard {shard}/{num_shards}: IDs {start}–{end - 1} ({len(all_ids)} series)"
        )

    extra_kwargs = {}
    time_format = None
    if bench_params.get("needs_calendar"):
        config = TimeBasedConfig(
            ts_ids=[all_ids[0]],
            train_time_period=agg_params["time_range"],
            features_to_take=[FEATURE],
        )
        dataset_obj.set_dataset_config_and_initialize(config)
        non_work_days = _load_non_work_days(dataset_obj, agg_params)
        extra_kwargs["non_work_days"] = non_work_days
        time_format = "datetime"

    done_ids = set()
    if os.path.exists(partial_path):
        done_ids.update(pd.read_csv(partial_path)[id_col].tolist())
    original_partial = out_path.replace(".csv", "_partial.csv")
    if original_partial != partial_path and os.path.exists(original_partial):
        done_ids.update(pd.read_csv(original_partial)[id_col].tolist())
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done.")

    remaining = [i for i in all_ids if i not in done_ids]
    print(f"Remaining: {len(remaining)}")

    chunks = [
        remaining[i : i + CHUNK_SIZE] for i in range(0, len(remaining), CHUNK_SIZE)
    ]
    for idx, chunk_ids in enumerate(chunks):
        print(f"Chunk {idx + 1}/{len(chunks)} ({len(chunk_ids)} series)...", flush=True)
        try:
            kwargs = dict(
                ts_ids=chunk_ids,
                train_time_period=agg_params["time_range"],
                features_to_take=[FEATURE],
            )
            if time_format:
                kwargs["time_format"] = time_format
            config = TimeBasedConfig(**kwargs)
            dataset_obj.set_dataset_config_and_initialize(config)
            df_chunk = dataset_obj.get_all_df()
        except Exception as e:
            print(f"  ERROR: {e} — skipping chunk.")
            continue

        groups = dict(list(df_chunk.groupby(id_col, sort=False)))
        tasks = []
        for ts_id in chunk_ids:
            grp = groups.get(ts_id)
            if grp is None or grp.empty:
                continue
            tasks.append(
                (grp, score_fn, id_col, agg_params, bench_params, extra_kwargs)
            )

        if workers > 1:
            with Pool(workers) as pool:
                rows = list(
                    tqdm(
                        pool.imap(_score_one, tasks),
                        total=len(tasks),
                        desc="  scoring",
                        leave=False,
                    )
                )
        else:
            rows = [_score_one(t) for t in tqdm(tasks, desc="  scoring", leave=False)]

        if rows:
            chunk_df = pd.DataFrame(rows)
            header = (
                not os.path.exists(partial_path) or os.path.getsize(partial_path) == 0
            )
            chunk_df.to_csv(partial_path, mode="a", header=header, index=False)
            print(f"  +{len(rows)} rows saved.")
        del df_chunk

    if shard is not None:
        print(f"Shard {shard} done. Partial saved to {partial_path}")
        return pd.DataFrame()

    if os.path.exists(partial_path):
        final = pd.read_csv(partial_path)
        final.to_csv(out_path, index=False)
        print(f"Done. Saved {out_path} ({len(final)} series)")
        return final

    return pd.DataFrame()


def merge_shards(benchmark, source, aggregation="hourly"):
    _, id_col = SOURCE_MAP[source]
    out_path = scores_path(aggregation, benchmark, source)
    pattern = out_path.replace(".csv", "_partial_s*.csv")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        print(f"No shard files found matching {pattern}")
        return pd.DataFrame()

    original_partial = out_path.replace(".csv", "_partial.csv")
    all_files = shard_files
    if os.path.exists(original_partial):
        all_files = [original_partial] + shard_files
    dfs = [pd.read_csv(f) for f in all_files]
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=[id_col])
    merged.to_csv(out_path, index=False)
    print(f"Merged {len(all_files)} files → {out_path} ({len(merged)} series)")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Score time series for benchmarks")
    parser.add_argument(
        "--aggregation", "-a", default="hourly", choices=list(AGGREGATIONS)
    )
    parser.add_argument(
        "--benchmark", "-b", default="all", choices=["all"] + list(BENCHMARKS)
    )
    parser.add_argument(
        "--source", "-s", default="all", choices=["all"] + list(SOURCE_MAP)
    )
    parser.add_argument("--data-root", default="../data/")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=1)
    args = parser.parse_args()

    if args.merge:
        benchmarks = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
        sources = [args.source] if args.source != "all" else ["ips_full"]
        for bench in benchmarks:
            for source in sources:
                merge_shards(bench, source, args.aggregation)
        return

    benchmarks = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]

    for bench in benchmarks:
        bench_params = BENCHMARKS[bench]
        sources = bench_params["levels"] if args.source == "all" else [args.source]
        for source in sources:
            print(f"\n{'='*60}")
            print(f"Scoring: {bench} / {source} / {args.aggregation}")
            print(f"{'='*60}")
            score_source(
                bench,
                source,
                args.aggregation,
                args.data_root,
                shard=args.shard,
                num_shards=args.num_shards,
                workers=args.workers,
            )


if __name__ == "__main__":
    main()
