# Adding a New Model to NetTS-eval

This guide walks through every step required to integrate a new forecasting model.

---

## 1. Create the model file

Create `nettseval/models/mymodel.py`. Inherit from `BaseModel` and implement the required interface.

```python
import numpy as np
import pandas as pd

from nettseval.models.base import BaseModel


class MyModel(BaseModel):
    # Set to True only if you implement predict_batch() (see section 4)
    supports_batch: bool = False

    def __init__(self, lookback: int = 168, horizon: int = 24, **kwargs):
        self.lookback = lookback
        self.horizon = horizon
        # store any other hyperparams

    # ------------------------------------------------------------------ #
    # Optional: called once per time series before the sliding-window loop.
    # Use it to fit scalers or train neural networks.
    # Return a dict of tuned params (saved to results) or None.
    # ------------------------------------------------------------------ #
    def tune(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        target_column: str = "n_bytes",
        ts_id: int | None = None,
    ) -> dict | None:
        return None

    # ------------------------------------------------------------------ #
    # Required: called once per sliding window with the context DataFrame.
    # Store whatever state predict() will need.
    # ------------------------------------------------------------------ #
    def fit(self, train_data: pd.DataFrame, target_column: str = "n_bytes") -> None:
        self._context = train_data[target_column].values

    # ------------------------------------------------------------------ #
    # Required: produce `horizon` predictions as a plain pd.Series.
    # Values must be in the original scale (n_bytes, not normalised).
    # Clip negative values to 0 — traffic cannot be negative.
    # ------------------------------------------------------------------ #
    def predict(self, horizon: int) -> pd.Series:
        preds = ...  # your model's forecast
        return pd.Series(np.clip(preds[:horizon], 0, None))

    # ------------------------------------------------------------------ #
    # Required: return a dict of all hyperparameters (saved to JSON).
    # ------------------------------------------------------------------ #
    def get_params(self) -> dict:
        return {"lookback": self.lookback, "horizon": self.horizon}

    # ------------------------------------------------------------------ #
    # Required: short lowercase identifier used in filenames and CLI.
    # ------------------------------------------------------------------ #
    def get_name(self) -> str:
        return "mymodel"
```

---

## 2. Optional extensions

### Fixed context size

Override `get_required_context_size()` if the model requires exactly N input steps (e.g. a pretrained transformer with a fixed context length). The evaluator will use this value instead of the CLI `--lookback`.

```python
def get_required_context_size(self) -> int:
    return self.lookback  # or read from model config
```

### Batched inference (`supports_batch = True`)

If your model can forecast all sliding windows in a single forward pass (e.g. a foundation model), set `supports_batch = True` and implement `predict_batch()`. The evaluator skips the per-window `fit()` + `predict()` loop and calls this instead.

```python
supports_batch: bool = True

def predict_batch(
    self,
    contexts: list[pd.DataFrame],
    target_column: str,
    horizon: int,
) -> list[pd.Series]:
    # contexts[i] is the context DataFrame for window i
    # return one pd.Series per context, each of length horizon
    return [pd.Series(...) for _ in contexts]
```

`fit()` is still called per window when `supports_batch = False`, so keep `fit()` lightweight in that path (store context only; heavy work belongs in `tune()`).

---

## 3. Register the model

**`nettseval/models/__init__.py`** — add the import and `__all__` entry:

```python
from nettseval.models.mymodel import MyModel

__all__ = [..., "MyModel"]
```

**`nettseval/constants.py`** — add to `MODEL_ORDER`:

```python
MODEL_ORDER = [..., "mymodel"]
```

**`run_evaluation.py`** — add to the import block and `create_model()`:

```python
# imports at top
from nettseval.models import ..., MyModel

# in create_model()
if model_name == "mymodel":
    return MyModel(lookback=lookback, horizon=horizon, **config)
```

---

## 4. Run the evaluation

```bash
python run_evaluation.py --model mymodel --benchmark drift seasonal
python run_evaluation.py --model mymodel --benchmark all --n-runs 3
```

Results land in `results/{timestamp}_mymodel_{freq}_{lookback}_{horizon}/`.

---

## Interface contract summary

| Method | Required | Called by evaluator |
|---|---|---|
| `fit(ctx, target_col)` | Yes | Per sliding window (non-batch path) |
| `predict(horizon)` | Yes | Per sliding window (non-batch path) |
| `predict_batch(contexts, target_col, horizon)` | Only if `supports_batch=True` | Once per series (batch path) |
| `tune(train, val, target_col, ts_id)` | No | Once per series, before any window |
| `get_required_context_size()` | No | Once at eval start |
| `get_params()` | Yes | Once (saved to `model_config.json`) |
| `get_name()` | Yes | Throughout (filenames, logging) |

### Data format

- All DataFrames have a `datetime` column and numeric metric columns.
- Target column is `n_bytes` by default.
- Context windows are pre-sliced by the evaluator; `fit()` receives exactly `context_size` rows.
- `predict()` must return a `pd.Series` of length `horizon` (or less for the final window).
- Missing values in the data are filled with `0` upstream — no NaN handling needed.
