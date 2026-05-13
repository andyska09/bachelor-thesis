from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    supports_batch: bool = False

    def tune(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_column: str = "n_bytes",
        ts_id: int | None = None,
    ) -> dict | None:
        return None

    def get_required_context_size(self) -> int | None:
        return None

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        pass

    def predict_batch(
        self,
        contexts: list[pd.DataFrame],
        target_column: str,
        horizon: int,
    ) -> list[pd.Series]:
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> dict:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
