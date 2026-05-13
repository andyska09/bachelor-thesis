import numpy as np
import pandas as pd

from nettseval.models.base import BaseModel


class ZeroModel(BaseModel):

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        pass

    def predict(self, horizon: int) -> pd.Series:
        return pd.Series(np.zeros(horizon)).reset_index(drop=True)

    def get_params(self) -> dict:
        return {}

    def get_name(self) -> str:
        return "zero"
