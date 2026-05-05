#!/usr/bin/env python3
"""
Compare three forecast approaches on the same data/split:
  A: Target-only features (no neighbors)
  B: Fixed top-3 correlation neighbors from metadata
  Ours: Best agent-proposed neighbors from experiments log

Usage (from repo root):
    python3 -m services.forecast_engines.scripts.compare_baselines --horizon 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE.parent / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_FILE = DATA_DIR / "metadata" / "metadata.json"
EXPERIMENTS_FILE = DATA_DIR / "experiments_h5.json"
OUTPUT_FILE = DATA_DIR / "baseline_comparison.json"

# Default experiments file for backward compat
DEFAULT_EXPERIMENTS_FILE = DATA_DIR / "experiments.json"


def load_returns(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}.csv"
    df = pd.read_csv(path, skiprows=3, header=None, names=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df.set_index("Date", inplace=True)
    df["return"] = df["Close"].pct_change()
    return df[["Close", "return"]].dropna()


def build_features(target: str, neighbors: list[str], horizon: int):
    """Build feature matrix and cumulative-return target."""
    target_df = load_returns(target)

    for lag in range(1, 6):
        target_df[f"target_lag_{lag}"] = target_df["return"].shift(lag)

    target_df["rolling_mean_10"] = target_df["return"].rolling(10).mean()
    target_df["rolling_std_10"] = target_df["return"].rolling(10).std()

    for neighbor in neighbors:
        if neighbor == target:
            continue
        npath = RAW_DIR / f"{neighbor}.csv"
        if not npath.exists():
            continue
        ndf = load_returns(neighbor)
        nr = ndf["return"].reindex(target_df.index)
        target_df[f"{neighbor}_lag_1"] = nr.shift(1)
        target_df[f"{neighbor}_rolling_mean_5"] = nr.shift(1).rolling(5).mean()

    target_df["future_return"] = target_df["Close"].shift(-horizon) / target_df["Close"] - 1
    target_df = target_df.dropna()
    dates = target_df.index.to_series()

    X = target_df.drop(columns=["Close", "return", "future_return"])
    y = target_df["future_return"]

    return X, y, dates


def evaluate(X, y, horizon: int):
    """80/20 chronological split, train XGB, return metrics."""
    n = len(X)
    split = int(n * 0.8)
    train_end = split - horizon
    if train_end < 1 or split >= n:
        return None

    X_train, X_test = X.iloc[:train_end], X.iloc[split:]
    y_train, y_test = y.iloc[:train_end], y.iloc[split:]

    mae_zero = float(mean_absolute_error(y_test, np.zeros(len(y_test))))
    mae_mean = float(mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())))

    model = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))

    return {
        "mae": mae,
        "mae_baseline_zero": mae_zero,
        "mae_baseline_mean": mae_mean,
        "beats_baseline_zero": mae < mae_zero,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def get_best_agent_neighbors(experiments: list[dict], target: str, horizon: int) -> list[str]:
    """Pick best neighbor set from experiment log for a target."""
    cand = [
        e for e in experiments
        if e.get("target") == target
        and e.get("horizon", horizon) == horizon
    ]
    agentish = [e for e in cand if e.get("source") in ("experiment-multiple", "experiment-agent") or e.get("experiment_id") is not None]
    pool = agentish if agentish else [e for e in cand if "mae_for_ranking" in e]
    if not pool:
        return []
    best = min(pool, key=lambda e: float(e.get("mae_for_ranking", e.get("mae", float("inf")))))
    return best["neighbors"]


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline A/B/Ours comparison")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--experiment-file", type=str, default=None,
                   help="Path to experiments file (default: auto-detect by horizon)")
    args = p.parse_args()

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    # Load experiments
    if args.experiment_file:
        exp_path = Path(args.experiment_file)
    else:
        exp_path = DATA_DIR / f"experiments_h{args.horizon}.json"
        if not exp_path.exists():
            exp_path = DEFAULT_EXPERIMENTS_FILE

    if exp_path.exists():
        with open(exp_path) as f:
            experiments = json.load(f)
    else:
        experiments = []
        print("Warning: no experiments file found, 'Ours' will be empty.")

    symbols = sorted(metadata.keys())
    results = []
    win_counts = {"A": 0, "B": 0, "Ours": 0}
    total = 0

    for sym in symbols:
        print(f"  Evaluating {sym} ...", end=" ")

        npath = RAW_DIR / f"{sym}.csv"
        if not npath.exists():
            print("SKIP (no CSV)")
            continue

        # Get neighbors
        meta = metadata.get(sym, {})
        corr_neighbors = [c["symbol"] for c in meta.get("top_correlated", [])[:3]]
        agent_neighbors = get_best_agent_neighbors(experiments, sym, args.horizon)

        arms = {}
        # A: target-only
        result_a = evaluate(*build_features(sym, [], args.horizon), args.horizon)
        if result_a:
            arms["A"] = result_a

        # B: fixed correlation neighbors
        if corr_neighbors:
            result_b = evaluate(*build_features(sym, corr_neighbors, args.horizon), args.horizon)
            if result_b:
                arms["B"] = result_b

        # Ours: best agent neighbors
        if agent_neighbors:
            result_o = evaluate(*build_features(sym, agent_neighbors, args.horizon), args.horizon)
            if result_o:
                arms["Ours"] = result_o

        if len(arms) < 2:
            print(f"PARTIAL (only {list(arms.keys())})")
            continue

        # Find winner (lowest MAE)
        total += 1
        best_arm = min(arms, key=lambda k: arms[k]["mae"])
        win_counts[best_arm] += 1

        row = {
            "target": sym,
            "sector": meta.get("sector", "Unknown"),
            "A": None,
            "B": None,
            "Ours": None,
            "A_beats_0": None,
            "B_beats_0": None,
            "Ours_beats_0": None,
            "winner": best_arm,
        }
        for arm in ("A", "B", "Ours"):
            if arms.get(arm):
                row[arm] = round(arms[arm]["mae"], 6)
                row[f"{arm}_beats_0"] = arms[arm]["beats_baseline_zero"]
        results.append(row)
        print(f"A={row['A']}, B={row['B']}, Ours={row['Ours']} | Winner: {best_arm}")

    # Summary
    summary = {
        "horizon": args.horizon,
        "total_targets": total,
        "win_counts": win_counts,
    }

    output = {
        "summary": summary,
        "per_target": results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Horizon: {args.horizon}")
    print(f"Targets evaluated: {total}")
    print(f"Win counts: A={win_counts['A']}, B={win_counts['B']}, Ours={win_counts['Ours']}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
