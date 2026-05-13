"""Shared utilities for benchmark rebuild notebooks.

Contains data loading, sparsity metrics, filtering/sampling,
and common visualization helpers used across all benchmark categories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from cesnet_tszoo.datasets import CESNET_TimeSeries24
from cesnet_tszoo.utils.enums import SourceType, AgreggationType, DatasetType
from cesnet_tszoo.configs import TimeBasedConfig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Maps short names to (SourceType, id column name)
SOURCE_MAP = {
    "institutions": (SourceType.INSTITUTIONS, "id_institution"),
    "subnets": (SourceType.INSTITUTION_SUBNETS, "id_institution_subnet"),
    "ips": (SourceType.IP_ADDRESSES_SAMPLE, "id_ip"),
    "ips_full": (SourceType.IP_ADDRESSES_FULL, "id_ip"),
}


def load_dataset(
    source: str,
    aggregation=AgreggationType.AGG_1_HOUR,
    data_root: str = "../data/",
    time_range: range = range(0, 6718),
    ts_ids=1.0,
):
    """Load a single source type dataset.

    Parameters
    ----------
    source : one of 'institutions', 'subnets', 'ips'
    aggregation : AgreggationType (default AGG_1_HOUR)

    Returns dict with keys: 'dataset', 'df', 'id_col', 'ids'.
    """
    source_type, id_col = SOURCE_MAP[source]
    config = TimeBasedConfig(
        ts_ids=ts_ids,
        train_time_period=time_range,
        features_to_take=["n_bytes"],
        time_format="datetime",
        all_batch_size=2048,
    )
    dataset = CESNET_TimeSeries24.get_dataset(
        data_root=data_root,
        source_type=source_type,
        aggregation=aggregation,
        dataset_type=DatasetType.TIME_BASED,
    )
    dataset.set_dataset_config_and_initialize(config)
    df = dataset.get_all_df()
    ids = df[id_col].unique().tolist()
    print(f"{source} ({aggregation.name}): {len(ids)} series loaded")
    return {
        "dataset": dataset,
        "df": df,
        "id_col": id_col,
        "ids": ids,
    }


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------


def ratio_active(y: pd.Series, eps: float = 0.0) -> float:
    """Fraction of non-zero (> eps) points in the series."""
    y = y.astype(float)
    return float(((y > eps) & (~y.isna())).sum() / len(y)) if len(y) > 0 else 0.0


def filter_by_sparsity(
    df: pd.DataFrame, id_col: str, min_active: float, feature: str = "n_bytes"
) -> list:
    """Return list of IDs where ratio of active points >= min_active."""
    ids = []
    for ts_id, grp in df.groupby(id_col):
        if ratio_active(grp[feature]) >= min_active:
            ids.append(ts_id)
    return ids


def filter_by_sparsity_band(
    df: pd.DataFrame,
    id_col: str,
    min_active: float,
    max_active: float,
    feature: str = "n_bytes",
) -> list:
    """Return list of IDs where activity ratio is in [min_active, max_active]."""
    ids = []
    for ts_id, grp in df.groupby(id_col):
        r = ratio_active(grp[feature])
        if min_active <= r <= max_active:
            ids.append(ts_id)
    return ids


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------


def seasonality_strength(y: pd.Series, period: int) -> float:
    """Compute the strength of the seasonal component via STL decomposition.

    Expects an already-transformed series (e.g. log1p applied by the caller).
    Returns a value in [0, 1] where higher = stronger seasonality.
    Reference: https://otexts.com/fpp3/stlfeatures.html
    """
    y = y.astype(float).copy()
    if y.var() == 0:
        return 0.0

    stl = STL(y, period=period).fit()
    remainder_var = stl.resid.var()
    seasonal_plus_resid_var = (stl.resid + stl.seasonal).var()

    if seasonal_plus_resid_var == 0:
        return 0.0

    return max(0, 1 - remainder_var / seasonal_plus_resid_var)


# ---------------------------------------------------------------------------
# Filtering & sampling
# ---------------------------------------------------------------------------


def filter_and_sample(
    scores_df: pd.DataFrame, metric_col: str, threshold: float, n: int, seed: int
) -> pd.DataFrame:
    """Filter rows where metric_col >= threshold, then randomly sample n rows.

    Prints pool size and warns if the pool is smaller than n.
    """
    pool = scores_df[scores_df[metric_col] >= threshold].copy()
    print(f"  Pool size ({metric_col} >= {threshold}): {len(pool)} series")

    if len(pool) < n:
        print(f"  WARNING: pool has only {len(pool)} series, less than requested {n}.")
        print(f"  Consider lowering the threshold. Taking all available.")
        return pool

    sampled = pool.sample(n=n, random_state=seed)
    return sampled


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_pool_distribution(
    scores_dfs: list[tuple[pd.DataFrame, str]],
    metric_col: str,
    threshold: float,
    xlabel: str = "Score",
    metric_op: str = ">=",
    save_path=None,
    log_y: bool = False,
):
    """Plot histograms of a metric for multiple aggregation levels.

    Parameters
    ----------
    scores_dfs : list of (DataFrame, label) tuples
    metric_col : column name to histogram
    threshold : vertical line to draw
    xlabel : x-axis label
    metric_op : comparison operator for pool count ('>=' or '<=')
    log_y : use log scale for y-axis
    """
    n = len(scores_dfs)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (df_s, label) in zip(axes, scores_dfs):
        ax.hist(df_s[metric_col], bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(
            threshold, color="red", linestyle="--", label=f"threshold={threshold}"
        )
        xmin, xmax = ax.get_xlim()
        if metric_op == "<=":
            n_pass = (df_s[metric_col] <= threshold).sum()
            ax.axvspan(threshold, xmax, color="red", alpha=0.08, label="dropped")
        else:
            n_pass = (df_s[metric_col] >= threshold).sum()
            ax.axvspan(xmin, threshold, color="red", alpha=0.08, label="dropped")
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"{label} (pool={n_pass}/{len(df_s)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        if log_y:
            ax.set_yscale("log")
        ax.legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def print_summary(
    benchmark_name: str,
    selected: dict[str, list],
    pool_sizes: dict[str, int],
    threshold: float,
    threshold_name: str,
    sample_size: int,
    seed: int,
    extra_info: str = "",
):
    """Print a standardized summary of selected series for a benchmark."""
    print(f"\n{benchmark_name} BENCHMARK - Selected Series")
    print("=" * 55)
    print(f"Threshold: {threshold_name} >= {threshold}")
    print(f"Selection: Random sample of {sample_size} from qualifying pool")
    if extra_info:
        print(extra_info)
    print(f"Random seed: {seed}")
    print()
    for level, ids in selected.items():
        pool = pool_sizes.get(level, "?")
        print(f"{level} ({len(ids)}): {sorted(ids)}")
        print(f"  Pool size: {pool}")
