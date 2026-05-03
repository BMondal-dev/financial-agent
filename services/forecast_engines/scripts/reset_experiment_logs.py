#!/usr/bin/env python3
"""
Reset experiment history to start fresh (keeps raw CSVs, metadata, graph).

Usage (from repo root or this directory):
    uv run python scripts/reset_experiment_logs.py
    uv run python scripts/reset_experiment_logs.py --no-analyze

Then re-run experiments and optionally:
    uv run python scripts/analyze_experiments.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SERVICES = Path(__file__).resolve().parent.parent.parent
BASE = SERVICES / "data"
EXPERIMENTS = BASE / "experiments.json"
LOCK = BASE / "experiments.json.lock"
RUNS_DIR = BASE / "experiments_runs"
REPO_ROOT = SERVICES.parent
ANALYZE = SERVICES / "forecast_engines" / "scripts" / "analyze_experiments.py"
DASHBOARD_ANALYSIS = REPO_ROOT / "apps" / "dashboard" / "analysis.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Clear experiments.json and run artifacts.")
    p.add_argument(
        "--no-analyze",
        action="store_true",
        help="Do not regenerate analysis.json after reset.",
    )
    args = p.parse_args()

    EXPERIMENTS.write_text("[]\n", encoding="utf-8")
    print(f"Wrote empty {EXPERIMENTS}")

    if LOCK.exists():
        LOCK.unlink()
        print(f"Removed {LOCK}")

    if RUNS_DIR.exists():
        for f in RUNS_DIR.iterdir():
            if f.is_file():
                f.unlink()
                print(f"Removed {f.name}")
    else:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_analyze:
        print("Regenerating analysis.json …")
        fe_dir = SERVICES / "forecast_engines"
        rc = subprocess.run(
            [sys.executable, str(ANALYZE)],
            cwd=str(fe_dir),
        ).returncode
        if rc != 0:
            print("analyze_experiments.py failed; run it manually.", file=sys.stderr)
            sys.exit(rc)
        src = BASE / "analysis.json"
        if src.exists() and DASHBOARD_ANALYSIS.parent.is_dir():
            shutil.copy(src, DASHBOARD_ANALYSIS)
            print(f"Copied analysis.json → {DASHBOARD_ANALYSIS.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
