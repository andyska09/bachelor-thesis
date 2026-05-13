import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from nettseval.benchmarks import Benchmark, BenchmarkLoader
from nettseval.evaluation import (
    Evaluator,
    BenchmarkSourceResult,
    save_benchmark_results,
    save_run_summary,
    save_summary,
)
from nettseval.models import (
    SeasonalNaiveModel,
    SARIMAModel,
    ProphetModel,
    TimesFMModel,
    TTMModel,
    ChronosModel,
    MoraiModel,
    GRUModel,
    LSTMModel,
    GRUFCNModel,
    ZeroModel,
    MeanModel,
)
from nettseval.constants import MODEL_ORDER, FREQ_DIR
from nettseval.utils import EvaluationConfig

FREQ_DEFAULTS = {
    "h": {"lookback": 168, "horizon": 24},
    "d": {"lookback": 30, "horizon": 7},
    "m": {"lookback": 288, "horizon": 144},
}


def create_model(model_name: str, freq: str, config_path: str = None, lookback: int = None, horizon: int = None):
    config = {}
    if config_path:
        with open(config_path) as f:
            config = json.load(f)

    if model_name == "zero":
        return ZeroModel(**config)
    if model_name == "mean":
        return MeanModel(**config)
    if model_name == "seasonal_naive":
        return SeasonalNaiveModel(freq=freq, **config)
    if model_name == "sarima":
        return SARIMAModel(freq=freq, **config)
    if model_name == "prophet":
        return ProphetModel(freq=freq, country_holidays=config.pop("country_holidays", "CZ"), **config)
    if model_name == "timesfm":
        return TimesFMModel(**config)
    if model_name == "ttm":
        return TTMModel(freq=freq, lookback=lookback, horizon=horizon, **config)
    if model_name == "chronos":
        return ChronosModel(freq=freq, **config)
    if model_name == "moirai":
        return MoraiModel(freq=freq, **config)
    if model_name == "gru":
        return GRUModel(lookback=lookback, horizon=horizon, log_transform=True, **config)
    if model_name == "lstm":
        return LSTMModel(lookback=lookback, horizon=horizon, log_transform=True, **config)
    if model_name == "gru_fcn":
        return GRUFCNModel(lookback=lookback, horizon=horizon, log_transform=True, **config)

    raise ValueError(f"Unknown model: {model_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run time series forecasting evaluation")
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        choices=MODEL_ORDER,
        help="Model(s) to evaluate, e.g. --model sarima prophet ttm",
    )
    parser.add_argument(
        "--freq",
        choices=["h", "d", "m"],
        default="h",
        help="Data resolution: h=hourly, d=daily, m=10min (default: h)",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        default=["all"],
        help="Benchmark type(s), e.g. --benchmark drift seasonal. Use 'all' to run all available (default).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Context window size. Defaults: h=168, d=30, m=288",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast horizon. Defaults: h=24, d=7, m=144",
    )
    parser.add_argument("--n-runs", type=int, default=1, dest="n_runs")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        dest="save_predictions",
        help="Save prediction CSVs alongside metrics",
    )
    parser.add_argument("--config", type=str, default=None, help="JSON file for model param overrides")
    return parser.parse_args()


def load_benchmarks(bench_types: list[str], freq_dir: str) -> list[Benchmark]:
    benchmarks = []
    for bench_type in bench_types:
        try:
            benchmark = BenchmarkLoader.load(bench_type, freq_dir)
            benchmarks.append(benchmark)
        except FileNotFoundError as e:
            print(f"Skipping {freq_dir}/{bench_type}: {e}")
    return benchmarks


def compute_n_windows(benchmark: Benchmark, horizon: int) -> int:
    first_source = next(iter(benchmark.sources.values()))
    test_series = first_source.get_series(first_source.ts_ids[0], "test")
    return math.ceil(len(test_series) / horizon)


def setup_model_dir(
    model_name: str, args, bench_types: list[str], lookback: int, horizon: int, benchmark: Benchmark
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("results") / f"{timestamp}_{model_name}_{args.freq}_{lookback}_{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "config.json", "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "model": model_name,
                "freq": args.freq,
                "benchmarks": bench_types,
                "lookback": lookback,
                "horizon": horizon,
                "n_windows": compute_n_windows(benchmark, horizon),
                "n_runs": args.n_runs,
                "model_config_file": args.config,
            },
            f,
            indent=2,
        )
    return model_dir


def run_model(
    model_name: str,
    args,
    benchmarks: list[Benchmark],
    eval_config: EvaluationConfig,
    model_dir: Path,
):
    print(f"\n{'#' * 60}")
    print(f"MODEL: {model_name}")
    model_dir.mkdir(parents=True, exist_ok=True)

    all_run_results: list[list[BenchmarkSourceResult]] = []

    for run_idx in range(args.n_runs):
        print(f"\n{'=' * 60}")
        if args.n_runs > 1:
            print(f"Run {run_idx + 1}/{args.n_runs}")

        model = create_model(
            model_name,
            args.freq,
            args.config,
            lookback=eval_config.context_size,
            horizon=eval_config.forecast_horizon,
        )

        if run_idx == 0:
            with open(model_dir / "model_config.json", "w") as f:
                json.dump({"model": model_name, "params": model.get_params()}, f, indent=2)

        run_dir = model_dir / f"run_{run_idx + 1}"
        run_results: list[BenchmarkSourceResult] = []
        for benchmark in benchmarks:
            bench_results: list[BenchmarkSourceResult] = []
            for source_name, source in benchmark.sources.items():
                print(f"\n  {benchmark.bench_type}/{source_name} ({source.n_series} series)")
                result = Evaluator(source, model, eval_config).evaluate()
                bench_results.append(result)
                run_results.append(result)
                print(f"  -> {result.aggregate_metrics}")
            save_benchmark_results(bench_results, run_dir, save_predictions=args.save_predictions)

        save_run_summary(run_results, run_dir)
        all_run_results.append(run_results)
        if args.n_runs > 1:
            print(f"\nRun {run_idx + 1} saved to {run_dir}/")

    save_summary(all_run_results, model_dir)
    print(f"Summary saved to {model_dir}/summary.csv")


def main():
    args = parse_args()

    freq_dir = FREQ_DIR[args.freq]
    lookback = args.lookback or FREQ_DEFAULTS[args.freq]["lookback"]
    horizon = args.horizon or FREQ_DEFAULTS[args.freq]["horizon"]
    benchmarks_root = BenchmarkLoader.DEFAULT_BENCHMARKS_DIR / freq_dir

    if args.benchmark == ["all"]:
        bench_types = (
            sorted(d.name for d in benchmarks_root.iterdir() if d.is_dir()) if benchmarks_root.exists() else []
        )
    else:
        bench_types = args.benchmark
    if not bench_types:
        print(f"No benchmarks found in {benchmarks_root}")
        return

    benchmarks = load_benchmarks(bench_types, freq_dir)
    if not benchmarks:
        print("No benchmarks could be loaded.")
        return

    eval_config = EvaluationConfig(forecast_horizon=horizon, context_size=lookback, freq=args.freq)

    print(f"Freq: {args.freq} ({freq_dir}), lookback={lookback}, horizon={horizon}, n_runs={args.n_runs}")
    print(f"Benchmarks: {[(b.bench_type, list(b.sources.keys())) for b in benchmarks]}")

    for model_name in args.model:
        model_dir = setup_model_dir(model_name, args, bench_types, lookback, horizon, benchmarks[0])
        print(f"\nModel directory: {model_dir}")
        run_model(model_name, args, benchmarks, eval_config, model_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
