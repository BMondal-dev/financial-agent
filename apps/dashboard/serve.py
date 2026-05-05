#!/usr/bin/env python3
"""
Serve the Financial Agent dashboard on localhost:8080.

From repo root:
    python3 apps/dashboard/serve.py

Behavior:
1. Discovers all experiment files (experiments.json, experiments_h5.json, etc.)
2. For each, if the experiments file is newer than the analysis file,
   runs analyze_experiments.py to refresh aggregates.
3. Copies all analysis_*.json → apps/dashboard/
4. Writes a runs.json manifest for the dashboard dropdown.
"""
import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent.resolve()
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = REPO_ROOT / "services" / "data"
ANALYZE_SCRIPT = (
    REPO_ROOT / "services" / "forecast_engines" / "scripts" / "analyze_experiments.py"
)


def discover_experiment_files() -> list[tuple[str, Path, Path]]:
    """Return list of (run_id_or_None, experiments_path, analysis_path)."""
    results = []
    # Default file
    default_exp = DATA_DIR / "experiments.json"
    default_analysis = DATA_DIR / "analysis.json"
    if default_exp.exists():
        results.append((None, default_exp, default_analysis))

    # Named files: experiments_{run_id}.json
    for exp in sorted(DATA_DIR.glob("experiments_*.json")):
        # Skip lock files
        if exp.name.endswith(".lock"):
            continue
        run_id = exp.name.replace("experiments_", "").replace(".json", "")
        analysis = DATA_DIR / f"analysis_{run_id}.json"
        results.append((run_id, exp, analysis))

    return results


def maybe_regenerate_analysis() -> None:
    """For each experiment file, regenerate analysis if needed."""
    files = discover_experiment_files()
    for run_id, exp_path, analysis_path in files:
        label = f"analysis_{run_id}.json" if run_id else "analysis.json"
        exp_label = exp_path.name

        needs = False
        if not analysis_path.exists():
            needs = True
        elif exp_path.stat().st_mtime > analysis_path.stat().st_mtime:
            needs = True

        if not needs:
            continue

        print(f"Regenerating {label} from {exp_label} …")
        cmd = [sys.executable, str(ANALYZE_SCRIPT)]
        if run_id:
            cmd.extend(["--run-id", run_id])
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            print(
                f"Warning: analyze_experiments.py exited with non-zero status for "
                f"{label}; using existing if any."
            )


def sync_analysis_into_dashboard() -> tuple[bool, list[dict]]:
    """Copy all analysis files to dashboard, return (success, runs_manifest)."""
    runs = []
    any_file = False

    # Copy default analysis.json
    default_analysis = DATA_DIR / "analysis.json"
    if default_analysis.exists():
        dst = DASHBOARD_DIR / "analysis.json"
        shutil.copy(default_analysis, dst)
        print(f"Synced analysis.json → apps/dashboard/analysis.json")
        runs.append({"id": "default", "label": "All horizons", "file": "analysis.json"})
        any_file = True

    # Copy named analysis files
    for analysis in sorted(DATA_DIR.glob("analysis_*.json")):
        if analysis.name.endswith(".lock"):
            continue
        run_id = analysis.name.replace("analysis_", "").replace(".json", "")
        dst = DASHBOARD_DIR / analysis.name
        shutil.copy(analysis, dst)
        print(f"Synced {analysis.name} → apps/dashboard/{analysis.name}")
        # Extract horizon label
        if run_id.startswith("h"):
            try:
                days = run_id[1:]
                label = f"{days}-day horizon"
            except ValueError:
                label = run_id
        else:
            label = run_id
        runs.append({"id": run_id, "label": label, "file": analysis.name})
        any_file = True

    # Write runs manifest
    runs_manifest = DASHBOARD_DIR / "runs.json"
    json.dump(runs, indent=2, fp=runs_manifest.open("w"))
    print(f"Wrote runs.json with {len(runs)} run(s)")

    if not any_file:
        if (DASHBOARD_DIR / "analysis.json").exists():
            print(f"Using bundled analysis.json (no {DATA_DIR}/).")
            runs.append({"id": "default", "label": "All horizons", "file": "analysis.json"})
        else:
            print("ERROR: analysis.json not found.")
            print(
                "Generate it:  cd services/forecast_engines && "
                "uv run python scripts/analyze_experiments.py"
            )
            return False, []

    return True, runs


def main() -> None:
    maybe_regenerate_analysis()
    ok, runs = sync_analysis_into_dashboard()
    if not ok:
        sys.exit(1)

    port = 8080
    print(f"\n  Dashboard → http://localhost:{port}\n")
    webbrowser.open(f"http://localhost:{port}")
    os.chdir(DASHBOARD_DIR)
    subprocess.run([sys.executable, "-m", "http.server", str(port)])


if __name__ == "__main__":
    main()
