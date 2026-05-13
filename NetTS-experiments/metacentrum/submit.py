#!/usr/bin/env python3
"""Submit scoring jobs to MetaCentrum PBS. Run with: python -m metacentrum.submit"""

import argparse
import subprocess
import sys
from pathlib import Path

JOB_SCRIPT = Path(__file__).parent / "job.sh"

RESOURCES = "select=1:ncpus=4:mem=32gb"
WALLTIME = "24:00:00"


def _submit_one(args, shard=None, num_shards=None):
    qsub_vars = [
        f"AGGREGATION={args.aggregation}",
        f"BENCHMARK={args.benchmark}",
        f"SOURCE={args.source}",
    ]
    job_name = f"scoring_{args.benchmark}_{args.source}"

    if shard is not None:
        qsub_vars.append(f"SHARD={shard}")
        qsub_vars.append(f"NUM_SHARDS={num_shards}")
        job_name += f"_s{shard}"

    if args.workers:
        qsub_vars.append(f"WORKERS={args.workers}")

    cmd = [
        "qsub",
        "-N",
        job_name,
        "-l",
        RESOURCES,
        "-l",
        f"walltime={WALLTIME}",
        "-j",
        "oe",
        "-o",
        "logs/",
        "-v",
        ",".join(qsub_vars),
        str(JOB_SCRIPT),
    ]

    if args.dry_run:
        print("DRY RUN:", " ".join(cmd))
        return

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    print(f"Submitted: {result.stdout.strip()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregation", "-a", default="hourly")
    p.add_argument("--benchmark", "-b", default="SEASON")
    p.add_argument("--source", "-s", default="ips_full")
    p.add_argument("--num-shards", "-n", type=int, default=None)
    p.add_argument("--workers", "-w", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = p.parse_args()

    if args.num_shards and args.num_shards > 1:
        for shard in range(args.num_shards):
            _submit_one(args, shard=shard, num_shards=args.num_shards)
    else:
        _submit_one(args)


if __name__ == "__main__":
    main()
