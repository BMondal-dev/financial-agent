#!/usr/bin/env python3
"""
Serve the Financial Agent dashboard on localhost:8080.

From repo root:
    python3 apps/dashboard/serve.py

Behavior:
1. If services/data/experiments.json is newer than services/data/analysis.json,
   runs analyze_experiments.py to refresh aggregates.
2. Always copies services/data/analysis.json → apps/dashboard/analysis.json
   so the static UI loads the canonical dataset.
"""
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent.resolve()
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
ANALYSIS_SRC = REPO_ROOT / "services" / "data" / "analysis.json"
ANALYSIS_DST = DASHBOARD_DIR / "analysis.json"
EXPERIMENTS_JSON = REPO_ROOT / "services" / "data" / "experiments.json"
ANALYZE_SCRIPT = (
    REPO_ROOT / "services" / "forecast_engines" / "scripts" / "analyze_experiments.py"
)


def maybe_regenerate_analysis() -> None:
    if not EXPERIMENTS_JSON.exists():
        return
    if not ANALYSIS_SRC.exists():
        needs = True
    else:
        needs = EXPERIMENTS_JSON.stat().st_mtime > ANALYSIS_SRC.stat().st_mtime
    if not needs:
        return
    print("Regenerating analysis.json from experiments.json …")
    rc = subprocess.run(
        [sys.executable, str(ANALYZE_SCRIPT)],
        cwd=str(REPO_ROOT),
    ).returncode
    if rc != 0:
        print(
            "Warning: analyze_experiments.py exited with non-zero status; "
            "using existing analysis if any."
        )


def sync_analysis_into_dashboard() -> bool:
    if not ANALYSIS_SRC.exists():
        if ANALYSIS_DST.exists():
            print(f"Using bundled {ANALYSIS_DST} (no {ANALYSIS_SRC}).")
            return True
        print("ERROR: analysis.json not found.")
        print(
            "Generate it:  cd services/forecast_engines && "
            "uv run python scripts/analyze_experiments.py"
        )
        return False
    shutil.copy(ANALYSIS_SRC, ANALYSIS_DST)
    print(f"Synced {ANALYSIS_SRC.name} → {ANALYSIS_DST.relative_to(REPO_ROOT)}")
    return True


def main() -> None:
    maybe_regenerate_analysis()
    if not sync_analysis_into_dashboard():
        sys.exit(1)

    port = 8080
    print(f"\n  Dashboard → http://localhost:{port}\n")
    webbrowser.open(f"http://localhost:{port}")
    os.chdir(DASHBOARD_DIR)
    subprocess.run([sys.executable, "-m", "http.server", str(port)])


if __name__ == "__main__":
    main()
