from dataclasses import dataclass, field

from nettseval.constants import DEFAULT_EVAL_METRICS, SEASONAL_PERIOD


@dataclass
class EvaluationConfig:
    forecast_horizon: int
    context_size: int
    target_column: str = "n_bytes"
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_EVAL_METRICS))
    freq: str = "h"
    mase_m: int = field(init=False)

    def __post_init__(self):
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")

        if self.context_size <= 0:
            raise ValueError("context_size must be positive")

        self.mase_m = SEASONAL_PERIOD.get(self.freq, 24)
