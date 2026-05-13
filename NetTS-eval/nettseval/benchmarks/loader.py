import hashlib
import json
from pathlib import Path

import mlcroissant as mlc
import pandas as pd
from cesnet_tszoo.configs.time_based_config import TimeBasedConfig
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.datasets.time_based_cesnet_dataset import TimeBasedCesnetDataset
from cesnet_tszoo.utils.enums import AgreggationType, DatasetType, SourceType

from nettseval.benchmarks.benchmark import Benchmark, BenchmarkSource

NETTS_BENCH_CITATION = "Anonymous. (2026). NetTS: Behavioral Diagnostic Benchmarks for Network Traffic Forecasting"

CESNET_CITATION = (
    "Koumar, J., Hynek, K., Čejka, T., & Šiška, P. (2025). "
    "CESNET-TimeSeries24: Time Series Dataset for Network Traffic Anomaly Detection and Forecasting. "
    "Scientific Data, 12, 338. https://doi.org/10.1038/s41597-025-04603-x"
)


class BenchmarkLoader:
    DEFAULT_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"

    @staticmethod
    def load_bench(
        source: SourceType,
        aggregation: AgreggationType,
        config: TimeBasedConfig,
        data_root: str,
    ) -> TimeBasedCesnetDataset:
        dataset = CESNET_TimeSeries24.get_dataset(
            data_root,
            source_type=source,
            aggregation=aggregation,
            dataset_type=DatasetType.TIME_BASED,
        )
        dataset.set_dataset_config_and_initialize(config)
        return dataset

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def build_croissant(benchmarks_dir: Path = None) -> dict:
        if benchmarks_dir is None:
            benchmarks_dir = BenchmarkLoader.DEFAULT_BENCHMARKS_DIR

        distribution = []
        record_sets = []
        total_series = 0

        for freq_dir in sorted(p.name for p in benchmarks_dir.iterdir() if p.is_dir() and p.name != "selected_ids"):
            freq_path = benchmarks_dir / freq_dir
            for bench_type in sorted(p.name for p in freq_path.iterdir() if p.is_dir()):
                metadata_path = freq_path / bench_type / "metadata.json"
                if not metadata_path.exists():
                    continue
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)

                for source_name, source_info in metadata["sources"].items():
                    ts_id_col = source_info["ts_id_col"]
                    n_series = len(source_info["ts_ids"])
                    total_series += n_series
                    prefix = f"{freq_dir}-{bench_type}-{source_name}"
                    csv_path = freq_path / bench_type / f"{source_name}.csv"
                    parquet_path = freq_path / bench_type / f"{source_name}.parquet"
                    content_prefix = f"{freq_dir}/{bench_type}"
                    csv_id = f"{prefix}-csv"

                    distribution.extend(
                        [
                            mlc.FileObject(
                                id=csv_id,
                                name=f"{content_prefix}/{source_name}.csv",
                                content_url=f"{content_prefix}/{source_name}.csv",
                                encoding_formats=["text/csv"],
                                sha256=BenchmarkLoader._sha256(csv_path) if csv_path.exists() else "0" * 64,
                            ),
                            mlc.FileObject(
                                id=f"{prefix}-parquet",
                                name=f"{content_prefix}/{source_name}.parquet",
                                content_url=f"{content_prefix}/{source_name}.parquet",
                                encoding_formats=["application/x-parquet"],
                                sha256=(BenchmarkLoader._sha256(parquet_path) if parquet_path.exists() else "0" * 64),
                            ),
                        ]
                    )

                    record_sets.append(
                        mlc.RecordSet(
                            id=prefix,
                            name=prefix,
                            description=(
                                f"{freq_dir}/{bench_type} — {source_name}-level time series ({n_series} series)"
                            ),
                            key=[f"{prefix}/datetime", f"{prefix}/{ts_id_col}"],
                            fields=[
                                mlc.Field(
                                    id=f"{prefix}/datetime",
                                    name="datetime",
                                    data_types=[mlc.DataType.DATETIME],
                                    source=mlc.Source(file_object=csv_id, extract=mlc.Extract(column="datetime")),
                                ),
                                mlc.Field(
                                    id=f"{prefix}/{ts_id_col}",
                                    name=ts_id_col,
                                    data_types=[mlc.DataType.INTEGER],
                                    source=mlc.Source(file_object=csv_id, extract=mlc.Extract(column=ts_id_col)),
                                ),
                                mlc.Field(
                                    id=f"{prefix}/n_bytes",
                                    name="n_bytes",
                                    data_types=[mlc.DataType.INTEGER],
                                    source=mlc.Source(file_object=csv_id, extract=mlc.Extract(column="n_bytes")),
                                ),
                            ],
                        )
                    )

        metadata = mlc.Metadata(
            name="NetTS-bench",
            description=(
                "NetTS-bench: Network Traffic Time Series Benchmark suite for evaluating forecasting models "
                "on CESNET-TimeSeries24 network traffic data. Contains 5 benchmark types (seasonal, drift, "
                "periodic_spaces, workers, random) at 2 temporal resolutions (hourly, 10-minute) across "
                f"3 aggregation levels (IPs, institutions, subnets). {total_series} total time series. "
                "Data split: 60% train, 20% validation, 20% test. Target variable: n_bytes."
            ),
            license=["https://creativecommons.org/licenses/by/4.0/"],
            cite_as=NETTS_BENCH_CITATION,
            date_published="2026",
            version="1.0.0",
            distribution=distribution,
            record_sets=record_sets,
        )

        result = metadata.to_json()
        result["isBasedOn"] = {
            "@type": "sc:Dataset",
            "name": "CESNET-TimeSeries24",
            "url": "https://zenodo.org/records/13382427",
            "citeAs": CESNET_CITATION,
        }
        return result

    @staticmethod
    def save_croissant(benchmarks_dir: Path = None) -> Path:
        if benchmarks_dir is None:
            benchmarks_dir = BenchmarkLoader.DEFAULT_BENCHMARKS_DIR
        croissant_data = BenchmarkLoader.build_croissant(benchmarks_dir)
        out_path = benchmarks_dir / "croissant.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(croissant_data, f, indent=2, ensure_ascii=False)
        return out_path

    @staticmethod
    def save_dataset(
        dataset: TimeBasedCesnetDataset,
        bench_type: str,
        source_name: str,
        freq: str = "hourly",
        benchmarks_dir: Path = None,
    ) -> Path:
        if benchmarks_dir is None:
            benchmarks_dir = BenchmarkLoader.DEFAULT_BENCHMARKS_DIR

        bench_dir = Path(benchmarks_dir) / freq / bench_type
        bench_dir.mkdir(parents=True, exist_ok=True)

        all_df = dataset.get_all_df(as_single_dataframe=True, workers="config")
        all_df.to_csv(bench_dir / f"{source_name}.csv", index=False)
        all_df.to_parquet(bench_dir / f"{source_name}.parquet", index=False)

        config = dataset.dataset_config
        total = len(config.all_time_period)
        train_end = int(total * 0.6)
        val_end = train_end + (total - train_end) // 2

        metadata_path = bench_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {"sources": {}}

        metadata["aggregation"] = str(config.aggregation)
        metadata["train_end"] = train_end
        metadata["val_end"] = val_end

        metadata["sources"][source_name] = {
            "ts_id_col": dataset.metadata.ts_id_name,
            "ts_ids": config.ts_ids.tolist(),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return bench_dir

    @staticmethod
    def _split_df(
        data_df: pd.DataFrame, ts_id_col: str, train_end: int, val_end: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_parts, val_parts, test_parts = [], [], []
        for _, group in data_df.groupby(ts_id_col, sort=False):
            train_parts.append(group.iloc[:train_end])
            val_parts.append(group.iloc[train_end:val_end])
            test_parts.append(group.iloc[val_end:])
        return (
            pd.concat(train_parts, ignore_index=True),
            pd.concat(val_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True),
        )

    @staticmethod
    def _load_source(bench_dir: Path, source_name: str, bench_type: str, metadata: dict) -> BenchmarkSource:
        csv_file = bench_dir / f"{source_name}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {csv_file}")
        data_df = pd.read_csv(csv_file, parse_dates=["datetime"])

        source_meta = metadata["sources"][source_name]
        train_df, val_df, test_df = BenchmarkLoader._split_df(
            data_df,
            source_meta["ts_id_col"],
            metadata["train_end"],
            metadata["val_end"],
        )

        return BenchmarkSource(
            source_name=source_name,
            bench_type=bench_type,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            ts_id_column=source_meta["ts_id_col"],
            ts_ids=source_meta["ts_ids"],
            target_column="n_bytes",
            aggregation=metadata["aggregation"],
        )

    @staticmethod
    def list_sources(bench_type: str, freq_dir: str = "hourly") -> list[str]:
        bench_dir = BenchmarkLoader.DEFAULT_BENCHMARKS_DIR / freq_dir / bench_type
        if not bench_dir.exists():
            return []
        return sorted(p.stem for p in bench_dir.glob("*.csv"))

    @staticmethod
    def load(bench_type: str, freq_dir: str = "hourly") -> Benchmark:
        bench_dir = BenchmarkLoader.DEFAULT_BENCHMARKS_DIR / freq_dir / bench_type
        metadata_path = bench_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {bench_dir}")
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        sources = {}
        for source_name in metadata["sources"]:
            try:
                sources[source_name] = BenchmarkLoader._load_source(bench_dir, source_name, bench_type, metadata)
            except FileNotFoundError as e:
                print(f"Skipping {bench_type}/{source_name}: {e}")

        return Benchmark(bench_type=bench_type, freq=freq_dir, sources=sources)
