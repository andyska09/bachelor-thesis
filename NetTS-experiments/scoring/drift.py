from utils import ratio_active


def compute_rel_dev(grp, agg_params, bench_params):
    y = grp["n_bytes"]
    ra = ratio_active(y)

    scale = agg_params["daily_period"] / 24  # 1 for hourly, 6 for 10min
    short_window = int(bench_params["short_window_factor"] * scale)
    long_window = int(bench_params["long_window_factor"] * scale)

    x = y.astype(float)
    ma_short = x.rolling(window=short_window, min_periods=short_window).mean()
    ma_long = x.rolling(window=long_window, min_periods=long_window).mean()

    valid = ma_long.dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return None, None, ra

    idx = valid.index
    rel_dev = (ma_short.loc[idx] - valid).abs() / valid
    test_start_idx = x.index[int(len(x) * bench_params["train_val_frac"])]
    return rel_dev, test_start_idx, ra


def _eval_drift(rel_dev, test_start_idx, threshold, min_drift_steps):
    mask = rel_dev > threshold
    sustained = mask.rolling(min_drift_steps, min_periods=min_drift_steps).mean() == 1.0
    n_events = int((sustained & ~sustained.shift(1, fill_value=False)).sum())
    has_drift = bool(sustained.any())
    drift_in_test = bool(sustained[sustained.index >= test_start_idx].any())
    return has_drift, drift_in_test, n_events, sustained


def score_drift(grp, id_col, agg_params, bench_params):
    ts_id = grp[id_col].iloc[0]
    rel_dev, test_start_idx, ra = compute_rel_dev(grp, agg_params, bench_params)

    if rel_dev is None:
        return {
            id_col: ts_id,
            "ratio_active": ra,
            "has_drift": False,
            "drift_in_test": False,
            "max_rel_dev": 0.0,
            "drift_fraction": 0.0,
            "n_drift_events": 0,
        }

    scale = agg_params["daily_period"] / 24
    min_drift_steps = int(bench_params["min_drift_days"] * 24 * scale)
    has_drift, drift_in_test, n_events, sustained = _eval_drift(
        rel_dev, test_start_idx, bench_params["threshold"], min_drift_steps
    )

    return {
        id_col: ts_id,
        "ratio_active": ra,
        "has_drift": has_drift,
        "drift_in_test": drift_in_test,
        "max_rel_dev": float(rel_dev.max()),
        "drift_fraction": float(sustained.sum() / len(grp)),
        "n_drift_events": n_events,
    }


def score_drift_sweep(grp, id_col, agg_params, bench_params):
    ts_id = grp[id_col].iloc[0]
    rel_dev, test_start_idx, ra = compute_rel_dev(grp, agg_params, bench_params)

    row = {
        id_col: ts_id,
        "ratio_active": ra,
        "max_rel_dev": float(rel_dev.max()) if rel_dev is not None else 0.0,
    }

    scale = agg_params["daily_period"] / 24
    for t in bench_params["dev_sweep"]:
        for p_days in bench_params["persistence_sweep_days"]:
            key = f"t{int(round(t * 100))}p{p_days}"
            if rel_dev is None:
                row[f"{key}_d"] = False
                row[f"{key}_n"] = 0
            else:
                min_drift_steps = int(p_days * 24 * scale)
                _, drift_in_test, n_events, _ = _eval_drift(
                    rel_dev, test_start_idx, t, min_drift_steps
                )
                row[f"{key}_d"] = drift_in_test
                row[f"{key}_n"] = n_events

    for p_days in bench_params["persistence_sweep_days"]:
        p_steps = int(p_days * 24 * scale)
        if rel_dev is None or len(rel_dev) < p_steps:
            row[f"S_p{p_days}"] = 0.0
            row[f"S_p{p_days}_test"] = 0.0
        else:
            rmin = rel_dev.rolling(p_steps, min_periods=p_steps).min()
            row[f"S_p{p_days}"] = float(rmin.max())
            rmin_test = rmin[rmin.index >= test_start_idx]
            row[f"S_p{p_days}_test"] = (
                float(rmin_test.max()) if not rmin_test.empty else 0.0
            )
    return row
