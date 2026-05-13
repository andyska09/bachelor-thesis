import logging
import pandas as pd
from prophet import Prophet

from nettseval.constants import PANDAS_FREQ
from nettseval.models.base import BaseModel


class ProphetModel(BaseModel):
    def __init__(
        self,
        freq: str = "h",
        weekly_seasonality: bool | int | str = "auto",
        daily_seasonality: bool | int | str = "auto",
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        n_changepoints: int = 25,
        country_holidays: str = None,
    ):
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        self.freq = PANDAS_FREQ.get(freq, freq)
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.n_changepoints = n_changepoints
        self.country_holidays = country_holidays
        self.model = None

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        df = pd.DataFrame(
            {
                "ds": pd.to_datetime(train_data["datetime"]).dt.tz_localize(None),
                "y": train_data[target_column].values,
            }
        )

        self.model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=False,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            n_changepoints=self.n_changepoints,
        )
        if self.country_holidays:
            self.model.add_country_holidays(country_name=self.country_holidays)
        self.model.fit(df)

    def predict(self, horizon: int) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model must be fitted before making predictions")

        future = self.model.make_future_dataframe(periods=horizon, freq=self.freq)
        forecast = self.model.predict(future)
        predictions = forecast["yhat"].iloc[-horizon:].values
        predictions = pd.Series(predictions).clip(lower=0)

        return predictions.reset_index(drop=True)

    def get_params(self) -> dict:
        return {
            "freq": self.freq,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "seasonality_mode": self.seasonality_mode,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "n_changepoints": self.n_changepoints,
            "country_holidays": self.country_holidays,
        }

    def get_name(self) -> str:
        return "prophet"
