import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from nettseval.models.base import BaseModel

_SARIMA_SEASONAL_ORDER = {"h": 24, "d": 1, "m": 6}


class SARIMAModel(BaseModel):
    def __init__(
        self,
        freq: str = "h",
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = None,
        log_transform: bool = False,
        **sarimax_params,
    ):
        self.freq = freq
        s = _SARIMA_SEASONAL_ORDER.get(freq, 24)
        self.order = order
        self.seasonal_order = seasonal_order if seasonal_order is not None else (1, 1, 1, s)
        self.log_transform = log_transform
        self.sarimax_params = sarimax_params
        self.fitted_model = None

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        y = train_data[target_column].values
        if self.log_transform:
            y = np.log1p(y)
        try:
            self.fitted_model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                # initialization="approximate_diffuse",
                **self.sarimax_params,
            ).fit(disp=False)
        except np.linalg.LinAlgError:
            self.fitted_model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                # initialization="approximate_diffuse",
                enforce_stationarity=False,
                enforce_invertibility=False,
                **self.sarimax_params,
            ).fit(disp=False)

    def predict(self, horizon: int) -> pd.Series:
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before making predictions")
        forecast = self.fitted_model.forecast(steps=horizon)
        if self.log_transform:
            forecast = np.expm1(forecast)
        forecast = np.maximum(forecast, 0)
        return pd.Series(forecast).reset_index(drop=True)

    def get_params(self) -> dict:
        params = {
            "freq": self.freq,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "log_transform": self.log_transform,
        }
        params.update(self.sarimax_params)
        return params

    def get_name(self) -> str:
        return "sarima"
