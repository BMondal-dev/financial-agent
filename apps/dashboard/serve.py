#!/usr/bin/env python3
"""
Serve the Financial Agent dashboard on localhost:8080.
Run this from the repo root:
    python apps/dashboard/serve.py
"""
import os, subprocess, sys, webbrowser
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent
ANALYSIS_SRC  = Path(__file__).parent.parent.parent / "services" / "data" / "analysis.json"
ANALYSIS_DST  = DASHBOARD_DIR / "analysis.json"

def main():
    # Re-run analysis if source is newer than cached copy
    if ANALYSIS_SRC.exists():
        src_mtime = ANALYSIS_SRC.stat().st_mtime
        dst_mtime = ANALYSIS_DST.stat().st_mtime if ANALYSIS_DST.exists() else 0
        if src_mtime > dst_mtime:
            print("Refreshing analysis.json …")
            import shutil
            shutil.copy(ANALYSIS_SRC, ANALYSIS_DST)
    elif not ANALYSIS_DST.exists():
        print("ERROR: analysis.json not found.")
        print("Run:  python services/forecast_engines/scripts/analyze_experiments.py")
        sys.exit(1)

    port = 8080
    print(f"\n  Dashboard → http://localhost:{port}\n")
    webbrowser.open(f"http://localhost:{port}")
    os.chdir(DASHBOARD_DIR)
    subprocess.run([sys.executable, "-m", "http.server", str(port)])

if __name__ == "__main__":
    main()
