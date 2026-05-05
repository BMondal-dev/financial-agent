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
RUNS_DIR = BASE / "experiments_runs"
REPO_ROOT = SERVICES.parent
ANALYZE = SERVICES / "forecast_engines" / "scripts" / "analyze_experiments.py"
DASHBOARD_DIR = REPO_ROOT / "apps" / "dashboard"


def main() -> None:
    p = argparse.ArgumentParser(description="Clear experiments.json and run artifacts.")
    p.add_argument(
        "--no-analyze",
        action="store_true",
        help="Do not regenerate analysis.json after reset.",
    )
    args = p.parse_args()

    # Reset default experiments.json
    experiments_file = BASE / "experiments.json"
    experiments_file.write_text("[]\n", encoding="utf-8")
    print(f"Wrote empty {experiments_file}")

    lock = BASE / "experiments.json.lock"
    if lock.exists():
        lock.unlink()
        print(f"Removed {lock}")

    # Reset all run-specific experiment files
    for exp in BASE.glob("experiments_*.json"):
        if exp.name.endswith(".lock"):
            continue
        exp.write_text("[]\n", encoding="utf-8")
        print(f"Wrote empty {exp}")
        run_id = exp.name.replace("experiments_", "").replace(".json", "")
        run_lock = BASE / f"experiments_{run_id}.json.lock"
        if run_lock.exists():
            run_lock.unlink()
            print(f"Removed {run_lock}")

    # Reset all analysis files
    for analysis in BASE.glob("analysis*.json"):
        analysis.unlink()
        print(f"Removed {analysis}")

    # Reset dashboard analysis files
    for analysis in DASHBOARD_DIR.glob("analysis*.json"):
        analysis.unlink()
        print(f"Removed {analysis}")
    runs_manifest = DASHBOARD_DIR / "runs.json"
    if runs_manifest.exists():
        runs_manifest.unlink()
        print(f"Removed {runs_manifest}")

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
        if src.exists() and DASHBOARD_DIR.is_dir():
            shutil.copy(src, DASHBOARD_DIR / "analysis.json")
            print(f"Copied analysis.json → apps/dashboard/analysis.json")


if __name__ == "__main__":
    main()
