import pandas as pd
import torch
import timesfm

from nettseval.models.base import BaseModel


class TimesFMModel(BaseModel):
    supports_batch = True

    def __init__(self):
        self._context = None

        torch.set_float32_matmul_precision("high")

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                per_core_batch_size=64,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        self._context = train_data[target_column].values.astype(float)

    def predict(self, horizon: int) -> pd.Series:
        if self._context is None:
            raise RuntimeError("Model must be fitted before making predictions")

        point_forecast, _ = self.model.forecast(
            inputs=[self._context],
            horizon=horizon,
        )

        return pd.Series(point_forecast[0][:horizon]).clip(lower=0).reset_index(drop=True)

    def predict_batch(self, contexts: list, target_column: str, horizon: int) -> list:
        inputs = [ctx[target_column].values.astype(float) for ctx in contexts]
        point_forecasts, _ = self.model.forecast(inputs=inputs, horizon=horizon)
        return [pd.Series(pf[:horizon]).clip(lower=0).reset_index(drop=True) for pf in point_forecasts]

    def get_params(self) -> dict:
        return {
            "model": "timesfm-2.5-200m-pytorch",
        }

    def get_name(self) -> str:
        return "timesfm"
