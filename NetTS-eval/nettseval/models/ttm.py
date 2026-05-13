import pandas as pd
import torch
from tsfm_public import TinyTimeMixerForPrediction

from nettseval.models.base import BaseModel

_FREQ_TOKEN_MAP = {
    "oov": 0,
    "min": 1,
    "2min": 2,
    "5min": 3,
    "10min": 4,
    "m": 4,
    "15min": 5,
    "30min": 6,
    "h": 7,
    "H": 7,
    "d": 8,
    "D": 8,
    "W": 9,
}


_MODEL_MAP = {
    (168, 24):   ("ibm-granite/granite-timeseries-ttm-r2", "180-60-ft-l1-r2.1"),
    (744, 168):  ("ibm-research/ttm-research-r2",          "1024-192-ft-r2"),
    (288, 144):  ("ibm-research/ttm-research-r2",          "512-192-ft-r2"),
    (1008, 144): ("ibm-research/ttm-research-r2",          "1024-192-ft-r2"),
}


class TTMModel(BaseModel):
    def __init__(self, freq: str = "h", lookback: int = 168, horizon: int = 24):
        self._context = None
        self._freq_token = _FREQ_TOKEN_MAP.get(freq, 0)

        default = ("ibm-granite/granite-timeseries-ttm-r2", "180-60-ft-l1-r2.1")
        model_id, revision = _MODEL_MAP.get((lookback, horizon), default)
        self._model_id = model_id
        self._revision = revision

        self.model = TinyTimeMixerForPrediction.from_pretrained(
            model_id,
            revision=revision,
            num_input_channels=1,
            device_map="auto",
        )
        self.device = next(self.model.parameters()).device
        self.model.eval()

    def get_required_context_size(self) -> int:
        return self.model.config.context_length

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        self._context = train_data[target_column].values.astype(float)

    def predict(self, horizon: int) -> pd.Series:
        if self._context is None:
            raise RuntimeError("Model must be fitted before making predictions")

        tensor = torch.tensor(self._context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        freq_token = torch.tensor([self._freq_token], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = self.model(past_values=tensor, freq_token=freq_token)

        preds = output.prediction_outputs[0, :horizon, 0].cpu().numpy()
        return pd.Series(preds).clip(lower=0).reset_index(drop=True)

    def get_params(self) -> dict:
        return {
            "model": f"{self._model_id}/{self._revision}",
            "freq_token": self._freq_token,
            "device": str(self.device),
        }

    def get_name(self) -> str:
        return "ttm"
