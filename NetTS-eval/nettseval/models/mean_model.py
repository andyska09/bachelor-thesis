import numpy as np
import pandas as pd

from nettseval.models.base import BaseModel


class MeanModel(BaseModel):

    def __init__(self, **kwargs):
        self._mean: float = 0.0

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        self._mean = train_data[target_column].mean()

    def predict(self, horizon: int) -> pd.Series:
        return pd.Series(np.full(horizon, self._mean)).reset_index(drop=True)

    def get_params(self) -> dict:
        return {"mean": self._mean}

    def get_name(self) -> str:
        return "mean"
