import pandas as pd
from chronos import Chronos2Pipeline

from nettseval.constants import PANDAS_FREQ
from nettseval.models.base import BaseModel


class ChronosModel(BaseModel):
    supports_batch = True

    def __init__(self, device: str = "auto", freq: str = "h"):
        self.device = device
        self._pandas_freq = PANDAS_FREQ.get(freq, freq)
        self._context_df = None
        self._target_column = None

        self._pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        self._context_df = train_data[["datetime", target_column]].copy()
        self._target_column = target_column

    def predict(self, horizon: int) -> pd.Series:
        if self._context_df is None:
            raise RuntimeError("Model must be fitted before making predictions")
        return self.predict_batch([self._context_df], self._target_column, horizon)[0]

    def predict_batch(self, contexts: list[pd.DataFrame], target_column: str, horizon: int) -> list[pd.Series]:
        frames = []
        for i, ctx in enumerate(contexts):
            start = ctx["datetime"].iloc[0]
            timestamps = pd.date_range(start, periods=len(ctx), freq=self._pandas_freq)
            df = pd.DataFrame({
                "timestamp": timestamps,
                "target": ctx[target_column].values,
                "item_id": str(i),
            })
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)

        pred_df = self._pipeline.predict_df(
            combined,
            prediction_length=horizon,
            quantile_levels=[0.5],
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
        )

        median_col = next(c for c in pred_df.columns if str(c) == "0.5")
        return [
            pd.Series(pred_df[pred_df["item_id"] == str(i)][median_col].values).clip(lower=0).reset_index(drop=True)
            for i in range(len(contexts))
        ]

    def get_params(self) -> dict:
        return {"model": "amazon/chronos-2", "device": self.device}

    def get_name(self) -> str:
        return "chronos"
