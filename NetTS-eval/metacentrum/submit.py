#!/usr/bin/env python3
"""Submit NetTS-eval evaluation jobs to MetaCentrum PBS. Run with: python -m metacentrum.submit. See metacentrum.md for usage."""

import argparse
import subprocess
import sys
from pathlib import Path

from nettseval.constants import MODEL_ORDER

GPU_MODELS = {"timesfm", "ttm", "chronos", "moirai", "gru", "lstm", "gru_fcn"}

RESOURCES = {
    "cpu": "select=1:ncpus=4:mem=16gb",
    "gpu": "select=1:ncpus=4:ngpus=1:mem=32gb:gpu_mem=16gb",
}
WALLTIME = {
    "cpu": "12:00:00",
    "gpu": "4:00:00",
}

JOB_SCRIPT = Path(__file__).parent / "job.sh"


def submit_job(model: str, args) -> str:
    kind = "gpu" if model in GPU_MODELS else "cpu"

    qsub_vars = [
        f"MODEL={model}",
        f"BENCHMARKS={' '.join(args.benchmark)}",
        f"FREQ={args.freq}",
    ]
    if args.lookback:
        qsub_vars.append(f"LOOKBACK={args.lookback}")
    if args.horizon:
        qsub_vars.append(f"HORIZON={args.horizon}")
    if args.n_runs != 1:
        qsub_vars.append(f"N_RUNS={args.n_runs}")
    if args.save_predictions:
        qsub_vars.append("SAVE_PREDICTIONS=1")

    bench_tag = "-".join(args.benchmark[:2])
    cmd = [
        "qsub",
        "-N",
        f"nettseval_{model}_{bench_tag}",
        "-l",
        RESOURCES[kind],
        "-l",
        f"walltime={WALLTIME[kind]}",
        "-j",
        "oe",
        "-o",
        "logs/",
        "-v",
        ",".join(qsub_vars),
        str(JOB_SCRIPT),
    ]

    if args.dry_run:
        print("  DRY RUN:", " ".join(cmd))
        return "dry-run"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        return "error"

    job_id = result.stdout.strip()
    print(f"  Submitted: {job_id}")
    return job_id


def parse_args():
    p = argparse.ArgumentParser(description="Submit NetTS-eval jobs to MetaCentrum PBS")
    p.add_argument(
        "--model",
        nargs="+",
        required=True,
        choices=MODEL_ORDER,
    )
    p.add_argument("--benchmark", nargs="+", default=["all"])
    p.add_argument("--freq", choices=["h", "d", "m"], default="h")
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--n-runs", type=int, default=1, dest="n_runs")
    p.add_argument("--save-predictions", action="store_true", dest="save_predictions")
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print qsub commands without submitting",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Submitting {len(args.model)} job(s)...")
    for model in args.model:
        kind = "gpu" if model in GPU_MODELS else "cpu"
        print(f"\n[{model}] ({kind.upper()}, walltime={WALLTIME[kind]})")
        submit_job(model, args)


if __name__ == "__main__":
    main()
