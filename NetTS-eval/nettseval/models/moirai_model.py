import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from nettseval.constants import PANDAS_FREQ
from nettseval.models.base import BaseModel


class MoraiModel(BaseModel):
    supports_batch = True

    def __init__(self, freq: str = "h"):
        self._gluonts_freq = PANDAS_FREQ.get(freq, freq)
        self._module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        self._predictor = None
        self._context = None
        self._target_column = None

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        self._context = train_data[["datetime", target_column]].copy()
        self._target_column = target_column

    def predict(self, horizon: int) -> pd.Series:
        if self._context is None:
            raise RuntimeError("Model must be fitted before making predictions")
        return self.predict_batch([self._context], self._target_column, horizon)[0]

    def predict_batch(self, contexts: list[pd.DataFrame], target_column: str, horizon: int) -> list[pd.Series]:
        if self._predictor is None:
            self._predictor = Moirai2Forecast(
                module=self._module,
                prediction_length=horizon,
                context_length=len(contexts[0]),
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).create_predictor(batch_size=64)
        dfs = {}
        for i, ctx in enumerate(contexts):
            df = ctx[[target_column]].copy()
            start = ctx["datetime"].iloc[0]
            df.index = pd.date_range(start, periods=len(df), freq=self._gluonts_freq)
            dfs[str(i)] = df
        ds = PandasDataset(dfs, freq=self._gluonts_freq, target=target_column)
        forecasts = list(self._predictor.predict(ds))
        return [pd.Series(f.quantile(0.5)).clip(lower=0).reset_index(drop=True) for f in forecasts]

    def get_params(self) -> dict:
        return {"model": "Salesforce/moirai-2.0-R-small"}

    def get_name(self) -> str:
        return "moirai"
