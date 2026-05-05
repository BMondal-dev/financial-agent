#!/usr/bin/env python3
"""Benchmark XGB vs LSTM models on the same targets and neighbor sets.

This script evaluates both model types under identical conditions (same horizon,
neighbors, and data splits) to ensure fair comparison. Results are saved to
benchmark_results.json for use by the analyzer and dashboard.

Usage:
    python scripts/benchmark_models.py [--horizon 10] [--run-id h10]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecast_eval import evaluate_forecast, build_dataset
from experiment_logger import _experiment_path

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_BASE_DIR, "../data")
BENCHMARK_PATH = os.path.join(DATA_DIR, "benchmark_results.json")


def load_best_per_target(run_id: str | None = None) -> dict[str, dict]:
    """Load experiment log and extract best neighbor set per target.

    Returns a dict mapping target symbol to its best experiment record.
    """
    path = os.path.abspath(_experiment_path(run_id))
    if not os.path.exists(path):
        print(f"No experiment file found at {path}")
        return {}

    with open(path) as f:
        experiments = json.load(f)

    best_per_target: dict[str, dict] = {}
    for exp in experiments:
        target = exp.get("target")
        if not target:
            continue

        ranking_mae = exp.get("mae_for_ranking", exp.get("mae", float("inf")))

        if target not in best_per_target:
            best_per_target[target] = exp
        else:
            existing_mae = best_per_target[target].get(
                "mae_for_ranking", best_per_target[target].get("mae", float("inf"))
            )
            if ranking_mae < existing_mae:
                best_per_target[target] = exp

    return best_per_target


def run_benchmark(horizon: int, run_id: str | None = None) -> dict:
    """Run XGB vs LSTM benchmark for all targets.

    For each target, uses the best neighbor set from previous experiments
    and evaluates both XGB and LSTM under identical conditions.
    """
    best_per_target = load_best_per_target(run_id)

    if not best_per_target:
        print("No experiments found to benchmark.")
        return {"error": "No experiments found", "horizon": horizon}

    print(f"\nBenchmarking {len(best_per_target)} targets at horizon {horizon}")
    print("=" * 60)

    results_per_target = []
    xgb_wins = 0
    lstm_wins = 0
    ties = 0
    xgb_maes = []
    lstm_maes = []

    for target, best_exp in sorted(best_per_target.items()):
        neighbors = best_exp.get("neighbors", [])
        print(f"\n[{target}] neighbors: {neighbors}")

        try:
            xgb_result = evaluate_forecast(
                target, neighbors, horizon, include_cv=False, model_type="xgb"
            )
            lstm_result = evaluate_forecast(
                target, neighbors, horizon, include_cv=False, model_type="lstm"
            )

            if "error" in xgb_result or "error" in lstm_result:
                print(f"  Skipped: insufficient data")
                continue

            xgb_mae = xgb_result["mae"]
            lstm_mae = lstm_result["mae"]

            xgb_maes.append(xgb_mae)
            lstm_maes.append(lstm_mae)

            if xgb_mae < lstm_mae:
                winner = "xgb"
                xgb_wins += 1
            elif lstm_mae < xgb_mae:
                winner = "lstm"
                lstm_wins += 1
            else:
                winner = "tie"
                ties += 1

            mae_gap = abs(xgb_mae - lstm_mae)

            result_row = {
                "target": target,
                "neighbors": neighbors,
                "xgb_mae": round(xgb_mae, 6),
                "lstm_mae": round(lstm_mae, 6),
                "winner": winner,
                "mae_gap": round(mae_gap, 6),
                "xgb_beats_zero": xgb_result.get("beats_baseline_zero", False),
                "lstm_beats_zero": lstm_result.get("beats_baseline_zero", False),
                "test_date_start": xgb_result.get("eval", {}).get("test_date_start"),
                "test_date_end": xgb_result.get("eval", {}).get("test_date_end"),
            }
            results_per_target.append(result_row)

            print(f"  XGB: {xgb_mae:.4f}  LSTM: {lstm_mae:.4f}  Winner: {winner}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    aggregate = {
        "xgb_wins": xgb_wins,
        "lstm_wins": lstm_wins,
        "ties": ties,
        "total_targets": len(results_per_target),
        "xgb_avg_mae": round(sum(xgb_maes) / len(xgb_maes), 6) if xgb_maes else None,
        "lstm_avg_mae": round(sum(lstm_maes) / len(lstm_maes), 6) if lstm_maes else None,
        "xgb_min_mae": round(min(xgb_maes), 6) if xgb_maes else None,
        "lstm_min_mae": round(min(lstm_maes), 6) if lstm_maes else None,
        "xgb_max_mae": round(max(xgb_maes), 6) if xgb_maes else None,
        "lstm_max_mae": round(max(lstm_maes), 6) if lstm_maes else None,
    }

    benchmark_results = {
        "horizon": horizon,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_target": results_per_target,
        "aggregate": aggregate,
        "fairness_note": "Same data split, features, and test window for both models",
    }

    print("\n" + "=" * 60)
    print(f"Benchmark Summary (Horizon {horizon}):")
    print(f"  XGB wins:  {xgb_wins}")
    print(f"  LSTM wins: {lstm_wins}")
    print(f"  Ties:      {ties}")
    print(f"  XGB avg MAE:  {aggregate['xgb_avg_mae']}")
    print(f"  LSTM avg MAE: {aggregate['lstm_avg_mae']}")

    return benchmark_results


def save_benchmark(results: dict, horizon: int) -> str:
    """Save benchmark results to JSON file."""
    filename = f"benchmark_results_h{horizon}.json"
    output_path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark XGB vs LSTM models")
    parser.add_argument(
        "--horizon", type=int, default=10, help="Forecast horizon in days (default: 10)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for experiment file (e.g., 'h10'). Defaults to h{horizon}.",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"h{args.horizon}"

    results = run_benchmark(args.horizon, run_id)
    if "error" not in results:
        save_benchmark(results, args.horizon)


if __name__ == "__main__":
    main()
