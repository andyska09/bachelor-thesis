import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastai.callback.tracker import EarlyStoppingCallback
from tsai.all import (
    Adam,
    GRU,
    Learner,
    TSDataLoaders,
    TSDatasets,
    TSRegression,
    combine_split_data,
)

from nettseval.models.base import BaseModel


class GRUModel(BaseModel):
    def __init__(
        self,
        lookback: int = 168,
        horizon: int = 24,
        hidden_size: int = 100,
        n_layers: int = 1,
        bidirectional: bool = True,
        log_transform: bool = False,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 16,
        patience: int = 5,
        device: str = "auto",
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.log_transform = log_transform
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._learner: Learner | None = None
        self._context: np.ndarray | None = None

    def _make_windows(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        for i in range(self.lookback, len(values) - self.horizon + 1, self.horizon):
            x_raw = values[i - self.lookback : i]
            y_raw = values[i : i + self.horizon]
            x_min, x_max = x_raw.min(), x_raw.max()
            scale = x_max - x_min if x_max > x_min else 1.0
            X.append((x_raw - x_min) / scale)
            Y.append((y_raw - x_min) / scale)
        X = np.array(X, dtype=np.float32).reshape(-1, 1, self.lookback)
        Y = np.array(Y, dtype=np.float32)
        return X, Y



    def tune(
        self, train_data: pd.DataFrame, val_data: pd.DataFrame, target_column: str = "n_bytes", ts_id: int | None = None
    ) -> dict | None:
        train_vals = train_data[target_column].values.astype(np.float32)
        val_vals = val_data[target_column].values.astype(np.float32)

        combined_vals = np.concatenate([train_vals, val_vals])
        if self.log_transform:
            combined_vals = np.log1p(combined_vals)

        boundary = len(train_vals)
        X_all, Y_all = self._make_windows(combined_vals)

        train_idx, val_idx = [], []
        for j, i in enumerate(range(self.lookback, len(combined_vals) - self.horizon + 1, self.horizon)):
            if i + self.horizon <= boundary:
                train_idx.append(j)
            elif i >= boundary:
                val_idx.append(j)

        if not train_idx or not val_idx:
            return None

        X_train, Y_train = X_all[train_idx], Y_all[train_idx]
        X_val, Y_val = X_all[val_idx], Y_all[val_idx]

        X, Y, splits = combine_split_data([X_train, X_val], [Y_train, Y_val])
        dsets = TSDatasets(X, Y, tfms=[None, [TSRegression()]], splits=splits, inplace=True)
        dls = TSDataLoaders.from_dsets(
            dsets.train,
            dsets.valid,
            bs=[self.batch_size, self.batch_size],
            num_workers=0,
            device=self._device,
        )

        model = GRU(
            c_in=dls.vars,
            c_out=self.horizon,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            bidirectional=self.bidirectional,
        ).to(self._device)

        self._learner = Learner(dls, model, loss_func=nn.MSELoss(), opt_func=Adam)

        cbs = [EarlyStoppingCallback(monitor="valid_loss", min_delta=0.0, patience=self.patience)]
        self._learner.fit_one_cycle(self.epochs, lr_max=self.lr, cbs=cbs)

        return None

    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        self._context = train_data[target_column].values[-self.lookback :].astype(np.float32)

    def predict(self, horizon: int) -> pd.Series:
        if self._learner is None:
            raise RuntimeError("Model must be tuned before making predictions")

        ctx = np.log1p(self._context) if self.log_transform else self._context
        ctx_min, ctx_max = ctx.min(), ctx.max()
        scale = ctx_max - ctx_min if ctx_max > ctx_min else 1.0
        scaled = ((ctx - ctx_min) / scale).astype(np.float32)
        x = torch.tensor(scaled[-self.lookback :].reshape(1, 1, self.lookback)).to(self._learner.dls.device)

        self._learner.model.eval()
        with torch.no_grad():
            preds_norm = self._learner.model(x).cpu().numpy().flatten()

        preds_norm = np.clip(preds_norm, None, 1)
        preds = preds_norm * scale + ctx_min
        if self.log_transform:
            preds = np.expm1(preds)
        return pd.Series(np.clip(preds[:horizon], 0, None))

    def get_required_context_size(self) -> int:
        return self.lookback

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "horizon": self.horizon,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "bidirectional": self.bidirectional,
            "log_transform": self.log_transform,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "device": str(self._device),
        }

    def get_name(self) -> str:
        return "gru"
