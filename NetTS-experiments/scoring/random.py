import numpy as np
from statsmodels.tsa.stattools import acf
from utils import ratio_active


def score_random(grp, id_col, agg_params, bench_params):
    ts_id = grp[id_col].iloc[0]
    y = grp["n_bytes"].astype(float).values
    ra = ratio_active(grp["n_bytes"])

    scale = agg_params["daily_period"] / 24
    max_lag = int(bench_params["max_lag_factor"] * scale)

    if len(y) < max_lag + 2 or np.std(y) == 0:
        return {id_col: ts_id, "ratio_active": ra, "max_acf": np.inf}

    acf_vals = acf(y, nlags=max_lag, fft=True)
    max_acf = float(np.max(np.abs(acf_vals[1:])))

    return {id_col: ts_id, "ratio_active": ra, "max_acf": max_acf}
