from dataclasses import dataclass

import pandas as pd


@dataclass
class BenchmarkSource:
    source_name: str
    bench_type: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    ts_id_column: str
    ts_ids: list[int]
    target_column: str
    aggregation: str

    def get_series(self, ts_id: int, split: str = "train") -> pd.DataFrame:
        if split == "train":
            df = self.train_df
        elif split == "val":
            df = self.val_df
        elif split == "test":
            df = self.test_df
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        return df[df[self.ts_id_column] == ts_id].copy()

    @property
    def n_series(self) -> int:
        return len(self.ts_ids)


@dataclass
class Benchmark:
    bench_type: str
    freq: str
    sources: dict[str, BenchmarkSource]
