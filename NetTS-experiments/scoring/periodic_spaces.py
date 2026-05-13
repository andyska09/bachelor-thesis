import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from utils import ratio_active


def score_periodic_spaces(grp, id_col, agg_params, bench_params):
    ts_id = grp[id_col].iloc[0]
    y = grp["n_bytes"].astype(float).values
    ra = ratio_active(grp["n_bytes"])

    scale = agg_params["daily_period"] / 24
    max_lag = int(bench_params["max_lag_factor"] * scale)
    prominence = bench_params["peak_prominence"]

    binary = (y == 0).astype(float)
    acf_values = acf(binary, nlags=max_lag, fft=True)

    peaks, _ = find_peaks(acf_values, prominence=prominence)

    if len(peaks) == 0:
        return {id_col: ts_id, "ratio_active": ra, "max_acf": 0.0, "dominant_lag": 0}

    peak_acfs = acf_values[peaks]
    best_idx = np.argmax(peak_acfs)

    return {
        id_col: ts_id,
        "ratio_active": ra,
        "max_acf": float(peak_acfs[best_idx]),
        "dominant_lag": int(peaks[best_idx]),
    }
