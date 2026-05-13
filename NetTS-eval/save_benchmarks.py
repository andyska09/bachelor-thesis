import argparse
from pathlib import Path

import numpy as np

np.bool = np.bool_  # cesnet_tszoo compat fix for NumPy >=1.24

import pandas as pd
from cesnet_tszoo.configs.time_based_config import TimeBasedConfig
from cesnet_tszoo.utils.enums import AgreggationType, SourceType

from nettseval.benchmarks.loader import BenchmarkLoader
from nettseval.constants import FREQ_DIR

DATA_ROOT = "../data"
SELECTED_IDS_DIR = Path("benchmarks/selected_ids")
BENCHMARKS_DIR = Path("benchmarks")

BENCH_FILE_MAP = {
    "DRIFT": "drift",
    "SEASON": "seasonal",
    "RANDOM": "random",
    "PERIODIC_SPACES": "periodic_spaces",
    "WORKERS": "workers",
}

LEVEL_SOURCE_MAP = {
    "Institutions": (SourceType.INSTITUTIONS, "institutions"),
    "Subnets": (SourceType.INSTITUTION_SUBNETS, "subnets"),
    "IPs": (SourceType.IP_ADDRESSES_FULL, "ips"),
}

FREQ_AGGREGATION = {
    "h": AgreggationType.AGG_1_HOUR,
    "m": AgreggationType.AGG_10_MINUTES,
}


def load_selected_ids(bench_stem: str, freq_dir: str) -> dict:
    df = pd.read_csv(SELECTED_IDS_DIR / freq_dir / f"{bench_stem}.csv")
    result = {}
    for level, group in df.groupby("level"):
        if level in LEVEL_SOURCE_MAP:
            source_type, source_name = LEVEL_SOURCE_MAP[level]
            result[source_name] = (source_type, group["ts_id"].tolist())
    return result


def create_config(ts_ids: list) -> TimeBasedConfig:
    return TimeBasedConfig(
        ts_ids=ts_ids,
        train_time_period=1.0,
        features_to_take=["n_bytes"],
        default_values=0,
        include_time=True,
        include_ts_id=True,
        time_format="datetime",
    )


def main():
    parser = argparse.ArgumentParser(description="Export benchmarks from cesnet-tszoo")
    freq_choices = list(FREQ_AGGREGATION.keys())
    parser.add_argument("--freq", choices=freq_choices, nargs="+", default=freq_choices)
    parser.add_argument("--benchmark", choices=list(BENCH_FILE_MAP.values()) + ["all"], nargs="+", default=["all"])
    args = parser.parse_args()

    if "all" in args.benchmark:
        bench_items = list(BENCH_FILE_MAP.items())
    else:
        bench_items = [(stem, btype) for stem, btype in BENCH_FILE_MAP.items() if btype in args.benchmark]

    for freq_key in args.freq:
        freq_dir = FREQ_DIR[freq_key]
        aggregation = FREQ_AGGREGATION[freq_key]

        ids_dir = SELECTED_IDS_DIR / freq_dir
        if not ids_dir.exists() or not any(ids_dir.iterdir()):
            print(f"No selected IDs for {freq_dir} in {ids_dir}, skipping")
            continue

        for bench_stem, bench_type in bench_items:
            ids_file = ids_dir / f"{bench_stem}.csv"
            if not ids_file.exists():
                print(f"No IDs file {ids_file}, skipping {bench_type} for {freq_dir}")
                continue

            sources = load_selected_ids(bench_stem, freq_dir)
            for source_name, (source_type, ts_ids) in sources.items():
                print(f"[{freq_dir}] Loading {bench_type}/{source_name} ({len(ts_ids)} series)...")
                config = create_config(ts_ids)
                dataset = BenchmarkLoader.load_bench(
                    source=source_type,
                    aggregation=aggregation,
                    config=config,
                    data_root=DATA_ROOT,
                )
                BenchmarkLoader.save_dataset(
                    dataset,
                    bench_type,
                    source_name,
                    freq=freq_dir,
                    benchmarks_dir=BENCHMARKS_DIR,
                )
                print(f"[{freq_dir}] Saved {bench_type}/{source_name}")

    croissant_path = BenchmarkLoader.save_croissant(BENCHMARKS_DIR)
    print(f"Saved unified Croissant metadata to {croissant_path}")


if __name__ == "__main__":
    main()
