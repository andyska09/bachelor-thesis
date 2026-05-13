import numpy as np
from utils import ratio_active, seasonality_strength


def score_seasonality(grp, id_col, agg_params, bench_params):
    ts_id = grp[id_col].iloc[0]
    y = grp["n_bytes"]
    ra = ratio_active(y)

    if (
        ra < bench_params["sparsity_min"]
    ):  # just for speeding the screening of the dataset
        return {
            id_col: ts_id,
            "ratio_active": ra,
            "strength_daily": np.nan,
            "strength_weekly": np.nan,
            "max_strength": np.nan,
        }

    y_log = np.log1p(y.astype(float))
    s_daily = seasonality_strength(y_log, period=agg_params["daily_period"])
    s_weekly = seasonality_strength(y_log, period=agg_params["weekly_period"])

    return {
        id_col: ts_id,
        "ratio_active": ra,
        "strength_daily": s_daily,
        "strength_weekly": s_weekly,
        "max_strength": max(s_daily, s_weekly),
    }
