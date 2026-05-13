# MetaCentrum Guide

## 1. Connect to a frontend node

```bash
ssh <your-username>@perian.metacentrum.cz
```

Other frontends: `skirit.ics.muni.cz`, `zuphux.cerit-sc.cz`, `elmo.metacentrum.cz`.
Full list: https://docs.metacentrum.cz/computing/frontends/

## 2. First-time setup

```bash
# Clone the repo
git clone <repo-url> ~/NetTS-eval && cd ~/NetTS-eval

# Create conda environment
conda create -n nettseval python=3.12 -y
conda activate nettseval
pip install -r requirements.txt

# Create logs directory
mkdir -p logs
```

## 3. Sync code updates

```bash
# On the frontend
cd ~/NetTS-eval && git pull
```

## 4. Submit jobs

One PBS job is submitted per model. GPU/CPU resources and walltime are selected automatically.

```bash
cd ~/NetTS-eval

# Single model
python -m metacentrum.submit --model sarima --benchmark all

# Multiple models (each becomes a separate job)
python -m metacentrum.submit --model timesfm ttm chronos moirai --benchmark all

# Custom lookback/horizon
python -m metacentrum.submit --model sarima --benchmark drift seasonal --lookback 744 --horizon 168

# Multiple runs (for nondeterministic models)
python -m metacentrum.submit --model gru lstm gru_fcn --benchmark all --n-runs 10 --save-predictions

# Preview qsub commands without submitting
python -m metacentrum.submit --model sarima --benchmark drift --dry-run
```

Resource allocation (set automatically by `submit.py`):

| Type | Models | Resources | Walltime |
|------|--------|-----------|----------|
| GPU  | timesfm, ttm, chronos, moirai, gru, lstm, gru_fcn | 4 CPUs, 1 GPU (16 GB), 32 GB RAM | 4 h |
| CPU  | sarima, prophet | 4 CPUs, 16 GB RAM | 12 h |

## 5. Monitor jobs

```bash
# List running/queued jobs
qstat -u <your-username>

# Watch job list (refresh every 30s)
watch -n 30 qstat -u <your-username>

# Check exit status of a finished job
qstat -x <job_id>

# Tail a specific model's log
tail -f ~/NetTS-eval/logs/sarima_*.log

# Tail whatever is currently running
tail -f ~/NetTS-eval/logs/*.log
```

## 6. Download results to local machine

```bash
# All results (never overwrites existing local files)
rsync -av --progress --ignore-existing \
  <your-username>@perian.metacentrum.cz:~/NetTS-eval/results/ ./results/

# Only a specific date prefix
rsync -av --progress --ignore-existing \
  --filter="+ */" \
  --filter="+ 20260324*/**" \
  --filter="- *" \
  <your-username>@perian.metacentrum.cz:~/NetTS-eval/results/ ./results/
```

## Reference submission commands

```bash
# === Hourly short (168/24) ===
python -m metacentrum.submit --model seasonal_naive --benchmark all --lookback 168 --horizon 24 --save-predictions
python -m metacentrum.submit --model sarima prophet --benchmark all --lookback 168 --horizon 24 --save-predictions
python -m metacentrum.submit --model timesfm ttm chronos moirai --benchmark all --lookback 168 --horizon 24 --save-predictions
python -m metacentrum.submit --model gru lstm gru_fcn --benchmark all --lookback 168 --horizon 24 --n-runs 10 --save-predictions

# === Hourly long (744/168) ===
python -m metacentrum.submit --model seasonal_naive --benchmark all --lookback 744 --horizon 168 --save-predictions
python -m metacentrum.submit --model sarima prophet --benchmark all --lookback 744 --horizon 168 --save-predictions
python -m metacentrum.submit --model timesfm ttm chronos moirai --benchmark all --lookback 744 --horizon 168 --save-predictions
python -m metacentrum.submit --model gru lstm gru_fcn --benchmark all --lookback 744 --horizon 168 --n-runs 10 --save-predictions

# === 10-min short (288/144) ===
python -m metacentrum.submit --model seasonal_naive --benchmark all --freq m --lookback 288 --horizon 144 --save-predictions
python -m metacentrum.submit --model sarima prophet --benchmark all --freq m --lookback 288 --horizon 144 --save-predictions
python -m metacentrum.submit --model timesfm ttm chronos moirai --benchmark all --freq m --lookback 288 --horizon 144 --save-predictions
python -m metacentrum.submit --model gru lstm gru_fcn --benchmark all --freq m --lookback 288 --horizon 144 --n-runs 10 --save-predictions

# === 10-min long (1008/144) ===
python -m metacentrum.submit --model seasonal_naive --benchmark all --freq m --lookback 1008 --horizon 144 --save-predictions
python -m metacentrum.submit --model sarima prophet --benchmark all --freq m --lookback 1008 --horizon 144 --save-predictions
python -m metacentrum.submit --model timesfm ttm chronos moirai --benchmark all --freq m --lookback 1008 --horizon 144 --save-predictions
python -m metacentrum.submit --model gru lstm gru_fcn --benchmark all --freq m --lookback 1008 --horizon 144 --n-runs 10 --save-predictions
```
