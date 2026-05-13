import numpy as np
import pandas as pd

from nettseval.constants import SEASONAL_PERIOD
from nettseval.models.base import BaseModel


class SeasonalNaiveModel(BaseModel):

    def __init__(self, freq: str = "h", **kwargs):
        self.freq = freq
        self.m = SEASONAL_PERIOD[freq]
        self._last_season: np.ndarray | None = None

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        values = train_data[target_column].values
        self._last_season = values[-self.m :]

    def predict(self, horizon: int) -> pd.Series:
        reps = (horizon + self.m - 1) // self.m
        forecast = np.tile(self._last_season, reps)[:horizon]
        return pd.Series(np.maximum(forecast, 0)).reset_index(drop=True)

    def get_params(self) -> dict:
        return {"freq": self.freq, "m": self.m}

    def get_name(self) -> str:
        return "seasonal_naive"
