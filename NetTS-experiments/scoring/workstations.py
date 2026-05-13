import numpy as np
from utils import ratio_active


def score_workstations(grp, id_col, agg_params, bench_params, non_work_days=None):
    ts_id = grp[id_col].iloc[0]
    y = grp["n_bytes"]
    ra = ratio_active(y)

    grp = grp.copy()
    grp["day"] = grp["datetime"].dt.normalize()
    grp["hour"] = grp["datetime"].dt.hour

    is_active = grp["n_bytes"] > 0
    total_active = is_active.sum()

    if total_active == 0:
        return {
            id_col: ts_id,
            "ratio_active": 0.0,
            "whr": 0.0,
            "total_active_intervals": 0,
            "work_active_intervals": 0,
        }

    if non_work_days is None:
        non_work_days = set()

    mask_work = (
        (~grp["day"].isin(non_work_days)) & (grp["hour"].between(8, 18)) & is_active
    )

    return {
        id_col: ts_id,
        "ratio_active": float(total_active / len(grp)),
        "whr": float(mask_work.sum() / total_active),
        "total_active_intervals": int(total_active),
        "work_active_intervals": int(mask_work.sum()),
    }
