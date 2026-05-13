from scoring.seasonality import score_seasonality
from scoring.drift import score_drift, score_drift_sweep
from scoring.periodic_spaces import score_periodic_spaces
from scoring.workstations import score_workstations
from scoring.random import score_random

SCORE_FUNCTIONS = {
    "SEASON": score_seasonality,
    "DRIFT": score_drift,
    "DRIFT_SWEEP": score_drift_sweep,
    "PERIODIC_SPACES": score_periodic_spaces,
    "WORKERS": score_workstations,
    "RANDOM": score_random,
}
