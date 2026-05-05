"""
Experiment Analysis Script
Analyzes experiments.json and generates a rich analysis JSON for the dashboard.

Usage:
    python scripts/analyze_experiments.py              # reads experiments.json → analysis.json
    python scripts/analyze_experiments.py --run-id h5   # reads experiments_h5.json → analysis_h5.json
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

_FORECAST_ENGINES = Path(__file__).resolve().parent.parent
if str(_FORECAST_ENGINES) not in sys.path:
    sys.path.insert(0, str(_FORECAST_ENGINES))
from models import get_model  # noqa: E402

BASE = Path(__file__).parent.parent.parent / "data"
METADATA_FILE = BASE / "metadata" / "metadata.json"
RAW_DIR = BASE / "raw"
BENCHMARK_DIR = BASE


def _paths(run_id: str | None = None):
    """Return (experiments_file, output_file) for a given run_id."""
    if run_id:
        return BASE / f"experiments_{run_id}.json", BASE / f"analysis_{run_id}.json"
    return BASE / "experiments.json", BASE / "analysis.json"


def load_data(experiments_file: Path):
    with open(experiments_file) as f:
        experiments = json.load(f)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    return experiments, metadata


def load_returns_with_date(symbol: str) -> pd.DataFrame:
    """Load date + Close + return series aligned with services/forecast_engines/main.py."""
    path = RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Close", "return"])

    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close"],
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df.set_index("Date", inplace=True)
    df["return"] = df["Close"].pct_change()
    df = df.dropna(subset=["return"])
    return df[["Close", "return"]].reset_index()


def build_backtest_series(target: str, neighbors: list[str], horizon: int) -> dict:
    """Recreate 80/20 test predictions (predicted vs actual) for one target."""
    target_df = load_returns_with_date(target)
    if target_df.empty:
        return {"rows": [], "mae": None, "count": 0}

    target_df.set_index("Date", inplace=True)

    for lag in range(1, 6):
        target_df[f"target_lag_{lag}"] = target_df["return"].shift(lag)

    target_df["rolling_mean_10"] = target_df["return"].rolling(10).mean()
    target_df["rolling_std_10"] = target_df["return"].rolling(10).std()

    # Compute future return on target-only data to fix the date range.
    # This ensures the test split is identical regardless of neighbor set.
    target_df["future_return"] = target_df["Close"].shift(-horizon) / target_df["Close"] - 1
    target_valid = target_df.dropna()

    valid_neighbors: list[str] = []
    for neighbor in neighbors:
        if neighbor == target:
            continue
        ndf = load_returns_with_date(neighbor)
        if ndf.empty:
            continue
        ndf.set_index("Date", inplace=True)
        # Reindex to target's valid date index
        neighbor_ret = ndf["return"].reindex(target_valid.index)
        target_df[f"{neighbor}_lag_1"] = neighbor_ret.shift(1)
        target_df[f"{neighbor}_rolling_mean_5"] = (
            neighbor_ret.shift(1).rolling(5).mean()
        )
        valid_neighbors.append(neighbor)

    # Only drop rows where neighbor features are NaN — date range already fixed
    target_df = target_df.loc[target_valid.index].dropna()

    if len(target_df) < 20:
        return {"rows": [], "mae": None, "count": 0}

    dates = target_df.index.to_series()
    X = target_df.drop(columns=["Close", "return", "future_return"])
    y = target_df["future_return"]

    split = int(len(X) * 0.8)
    train_end = split - horizon
    if split >= len(X) or train_end < 1:
        return {"rows": [], "mae": None, "count": 0}

    X_train, X_test = X.iloc[:train_end], X.iloc[split:]
    y_train, y_test = y.iloc[:train_end], y.iloc[split:]
    d_test = dates.iloc[split:]

    model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))

    rows = []
    for dt, a, p in zip(d_test, y_test, preds):
        rows.append(
            {
                "date": str(dt),
                "actual_return": round(float(a) * 100, 4),
                "predicted_return": round(float(p) * 100, 4),
            }
        )

    return {
        "rows": rows,
        "mae": round(mae, 6),
        "count": len(rows),
        "neighbors": valid_neighbors,
    }


def load_price_history(symbol: str, days: int = 365) -> list[dict]:
    """Load last `days` rows of close price + daily return for a symbol.

    CSV format produced by yfinance download scripts:
        Row 0: Price,Close      <- column headers ("Price" col holds dates)
        Row 1: Ticker,SYMBOL    <- skip
        Row 2: Date,            <- skip
        Row 3+: 2021-03-05,163.94...
    """
    path = RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        lines = f.readlines()
    # Skip the 3 non-data header rows, then parse positionally
    for line in lines[3:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 2 and parts[1].strip():
            try:
                rows.append(
                    {"date": parts[0].strip(), "close": float(parts[1].strip())}
                )
            except ValueError:
                continue
    rows = rows[-days:]
    # compute daily returns
    for i, r in enumerate(rows):
        if i == 0:
            r["return"] = None
        else:
            prev = rows[i - 1]["close"]
            r["return"] = round((r["close"] - prev) / prev * 100, 4) if prev else None
    return rows


def get_sector(symbol, metadata):
    return metadata.get(symbol, {}).get("sector", "Unknown")


def build_baseline_comparison(experiments: list, metadata: dict, horizon: int) -> dict:
    """Compare A (target-only), B (fixed correlation), Ours (best agent neighbors) per target.

    Computes the same 80/20 split and feature plumbing for **XGBoost** and **LSTM** (if PyTorch
    is installed). ``Ours`` uses the best logged experiment *for that model type* per target so
    LSTM arms align with LSTM-logged neighbor sets.

    Returned JSON includes ``by_model: { "xgb": {...}, "lstm": {...} }``. Top-level
    ``per_target`` / ``win_counts`` / ``total_targets`` mirror **xgb** for backward compatibility.
    """
    import numpy as np

    def _load_returns(sym):
        path = RAW_DIR / f"{sym}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path, skiprows=3, header=None, names=["Date", "Close"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"])
        df.set_index("Date", inplace=True)
        df["return"] = df["Close"].pct_change()
        return df[["Close", "return"]].dropna()

    def _build(target, neighbors):
        target_df = _load_returns(target)
        if target_df is None:
            return None, None, None
        for lag in range(1, 6):
            target_df[f"target_lag_{lag}"] = target_df["return"].shift(lag)
        target_df["rolling_mean_10"] = target_df["return"].rolling(10).mean()
        target_df["rolling_std_10"] = target_df["return"].rolling(10).std()
        # Compute future return on target-only data to fix the date range.
        # This ensures the test split (and mae_baseline_zero) is identical
        # regardless of which neighbor set is used.
        target_df["future_return"] = target_df["Close"].shift(-horizon) / target_df["Close"] - 1
        target_valid = target_df.dropna()
        for n in neighbors:
            if n == target:
                continue
            ndf = _load_returns(n)
            if ndf is None:
                continue
            nr = ndf["return"].reindex(target_valid.index)
            target_df[f"{n}_lag_1"] = nr.shift(1)
            target_df[f"{n}_rolling_mean_5"] = nr.shift(1).rolling(5).mean()
        target_df = target_df.loc[target_valid.index].dropna()
        if len(target_df) < 30:
            return None, None, None
        X = target_df.drop(columns=["Close", "return", "future_return"])
        y = target_df["future_return"]
        dates = target_df.index.to_series()
        return X, y, dates

    def _eval_arm(X, y, model_type: str):
        n = len(X)
        split = int(n * 0.8)
        train_end = split - horizon
        if train_end < 1 or split >= n:
            return None
        X_train, X_test = X.iloc[:train_end], X.iloc[split:]
        y_train, y_test = y.iloc[:train_end], y.iloc[split:]
        mae_zero = float(mean_absolute_error(y_test, np.zeros(len(y_test))))
        try:
            model = get_model(model_type)  # type: ignore[arg-type]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred = np.asarray(pred, dtype=float).reshape(-1)
            if pred.shape[0] != len(y_test):
                return None
            mae = float(mean_absolute_error(y_test, pred))
        except Exception:
            return None
        return {"mae": round(mae, 6), "beats_0": mae < mae_zero}

    def _agent_best_for_model(model_type: str) -> dict:
        """Pick best logged neighbor set per target, scoped to experiments for that model."""
        agent_best = {}
        for e in experiments:
            if e.get("horizon", horizon) != horizon:
                continue
            exp_mt = e.get("model_type", "xgb")
            if model_type == "xgb":
                if exp_mt == "lstm":
                    continue
            elif model_type == "lstm":
                if exp_mt != "lstm":
                    continue
            else:
                continue
            t = e["target"]
            key = e.get("mae_for_ranking", float("inf"))
            if t not in agent_best or key < agent_best[t]["score"]:
                agent_best[t] = {"score": key, "neighbors": e["neighbors"]}
        return agent_best

    def _run_model(model_type: str) -> dict | None:
        agent_best = _agent_best_for_model(model_type)
        per_target = []
        win_counts = Counter()
        total = 0

        for sym in sorted(metadata.keys()):
            meta = metadata[sym]
            corr_nb = [c["symbol"] for c in meta.get("top_correlated", [])[:3]]
            agent_nb = agent_best.get(sym, {}).get("neighbors", [])

            arms = {}
            X, y, _dates = _build(sym, [])
            if X is not None:
                r = _eval_arm(X, y, model_type)
                if r:
                    arms["A"] = r

            if corr_nb:
                X, y, _dates = _build(sym, corr_nb)
                if X is not None:
                    r = _eval_arm(X, y, model_type)
                    if r:
                        arms["B"] = r

            if agent_nb:
                X, y, _dates = _build(sym, agent_nb)
                if X is not None:
                    r = _eval_arm(X, y, model_type)
                    if r:
                        arms["Ours"] = r

            if len(arms) < 2:
                continue

            total += 1
            winner = min(arms, key=lambda k: arms[k]["mae"])
            win_counts[winner] += 1

            per_target.append({
                "target": sym,
                "sector": meta.get("sector", "Unknown"),
                "A_mae": arms.get("A", {}).get("mae"),
                "B_mae": arms.get("B", {}).get("mae"),
                "Ours_mae": arms.get("Ours", {}).get("mae"),
                "A_beats_0": arms.get("A", {}).get("beats_0"),
                "B_beats_0": arms.get("B", {}).get("beats_0"),
                "Ours_beats_0": arms.get("Ours", {}).get("beats_0"),
                "winner": winner,
            })

        if not per_target:
            return None
        return {
            "model_type": model_type,
            "total_targets": total,
            "win_counts": dict(win_counts),
            "per_target": per_target,
        }

    by_model: dict[str, dict] = {}
    xgb_res = _run_model("xgb")
    if xgb_res:
        by_model["xgb"] = xgb_res

    lstm_res = None
    try:
        get_model("lstm")
        lstm_res = _run_model("lstm")
    except Exception:
        lstm_res = None
    if lstm_res:
        by_model["lstm"] = lstm_res

    out: dict = {
        "horizon": horizon,
        "by_model": by_model,
    }
    primary = xgb_res or lstm_res
    if primary:
        out["total_targets"] = primary["total_targets"]
        out["win_counts"] = primary["win_counts"]
        out["per_target"] = primary["per_target"]
    else:
        out["total_targets"] = 0
        out["win_counts"] = {}
        out["per_target"] = []
    return out


def is_calibrated_experiment(e: dict) -> bool:
    return "mae_baseline_zero" in e


def load_benchmark_results(horizon: int) -> dict | None:
    """Load benchmark results for XGB vs LSTM comparison if available."""
    benchmark_file = BENCHMARK_DIR / f"benchmark_results_h{horizon}.json"
    if not benchmark_file.exists():
        return None
    with open(benchmark_file) as f:
        return json.load(f)


def build_model_benchmark(experiments: list, metadata: dict, horizon: int) -> dict:
    """Build model benchmark statistics from experiment data.

    Groups experiments by model_type and computes per-model aggregates.
    Also checks for benchmark_results file for direct XGB vs LSTM comparison.
    """
    # Group experiments by model_type
    by_model = defaultdict(list)
    for e in experiments:
        if e.get("horizon", horizon) != horizon:
            continue
        model = e.get("model_type", "xgb")
        by_model[model].append(e)

    # Compute per-model stats
    model_stats = {}
    for model, exps in by_model.items():
        maes = [e["mae"] for e in exps]
        beats_zero = sum(1 for e in exps if e.get("beats_baseline_zero"))
        
        # Find best per target for this model
        best_per_target = {}
        for e in exps:
            t = e["target"]
            key = e.get("mae_for_ranking", e.get("mae", float("inf")))
            if t not in best_per_target or key < best_per_target[t]["score"]:
                best_per_target[t] = {"score": key, "exp": e}

        targets_list = [
            {
                "target": t,
                "sector": get_sector(t, metadata),
                "mae": round(v["exp"]["mae"], 6),
                "beats_zero": v["exp"].get("beats_baseline_zero", False),
            }
            for t, v in sorted(best_per_target.items(), key=lambda x: x[1]["score"])
        ]

        model_stats[model] = {
            "experiment_count": len(exps),
            "target_count": len(best_per_target),
            "avg_mae": round(statistics.mean(maes), 6) if maes else None,
            "min_mae": round(min(maes), 6) if maes else None,
            "max_mae": round(max(maes), 6) if maes else None,
            "beats_zero_count": beats_zero,
            "beats_zero_pct": round(100 * beats_zero / len(exps), 2) if exps else None,
            "targets": targets_list,
        }

    # Load benchmark results if available (from benchmark_models.py)
    benchmark_data = load_benchmark_results(horizon)

    # Build per-target comparison if we have both models
    per_target_comparison = []
    if "xgb" in model_stats and "lstm" in model_stats:
        xgb_targets = {t["target"]: t["mae"] for t in model_stats["xgb"]["targets"]}
        lstm_targets = {t["target"]: t["mae"] for t in model_stats["lstm"]["targets"]}
        
        common_targets = set(xgb_targets.keys()) & set(lstm_targets.keys())
        xgb_wins = 0
        lstm_wins = 0
        
        for t in sorted(common_targets):
            xgb_mae = xgb_targets[t]
            lstm_mae = lstm_targets[t]
            if xgb_mae < lstm_mae:
                winner = "xgb"
                xgb_wins += 1
            elif lstm_mae < xgb_mae:
                winner = "lstm"
                lstm_wins += 1
            else:
                winner = "tie"
            
            per_target_comparison.append({
                "target": t,
                "sector": get_sector(t, metadata),
                "xgb_mae": xgb_mae,
                "lstm_mae": lstm_mae,
                "winner": winner,
                "mae_gap": round(abs(xgb_mae - lstm_mae), 6),
            })
    elif benchmark_data:
        per_target_comparison = benchmark_data.get("per_target", [])
        xgb_wins = benchmark_data.get("aggregate", {}).get("xgb_wins", 0)
        lstm_wins = benchmark_data.get("aggregate", {}).get("lstm_wins", 0)
    else:
        xgb_wins = 0
        lstm_wins = 0

    return {
        "horizon": horizon,
        "by_model": model_stats,
        "per_target": per_target_comparison,
        "aggregate": {
            "xgb_wins": xgb_wins,
            "lstm_wins": lstm_wins,
            "total_compared": len(per_target_comparison),
        },
        "benchmark_file_loaded": benchmark_data is not None,
        "fairness_note": "Same data split, features, and test window for both models",
    }


def analyze(experiments, metadata, horizon_days: int):
    experiments = [e for e in experiments if is_calibrated_experiment(e)]
    total = len(experiments)
    targets = set(e["target"] for e in experiments)

    # ── 1. Predictor Frequency ────────────────────────────────────────────────
    all_neighbors = []
    for e in experiments:
        all_neighbors.extend(e["neighbors"])
    neighbor_counts = Counter(all_neighbors)

    predictor_frequency = [
        {
            "symbol": sym,
            "count": cnt,
            "sector": get_sector(sym, metadata),
        }
        for sym, cnt in neighbor_counts.most_common(20)
    ]

    # ── 2. Best Experiment Per Target ─────────────────────────────────────────
    best_per_target = {}
    worst_per_target = {}

    calibrated_h = [
        e for e in experiments if e.get("horizon", horizon_days) == horizon_days
    ]

    for e in calibrated_h:
        t = e["target"]
        if t not in best_per_target or e["mae"] < best_per_target[t]["mae"]:
            best_per_target[t] = e
        if t not in worst_per_target or e["mae"] > worst_per_target[t]["mae"]:
            worst_per_target[t] = e

    best_list = sorted(
        [
            {
                "target": t,
                "sector": get_sector(t, metadata),
                "model_type": v.get("model_type", "xgb"),
                "best_mae": round(v["mae"], 6),
                "worst_mae": round(worst_per_target[t]["mae"], 6),
                "best_neighbors": v["neighbors"],
                "predicted_return": round(v["predicted_return"] * 100, 4),
                "beats_baseline_zero": v.get("beats_baseline_zero"),
                "mae_baseline_zero": round(float(v["mae_baseline_zero"]), 6)
                if v.get("mae_baseline_zero") is not None
                else None,
            }
            for t, v in best_per_target.items()
        ],
        key=lambda x: x["best_mae"],
    )

    # ── 3. Sector Influence Matrix ────────────────────────────────────────────
    # For each experiment, map each neighbor's sector → target's sector
    # Track: for combinations where neighbor sector→target sector reduced MAE
    # We measure: avg mae when using neighbor from sector X to predict sector Y
    sector_mae = defaultdict(list)  # (neighbor_sector, target_sector) → [mae]

    for e in experiments:
        t_sector = get_sector(e["target"], metadata)
        for n in e["neighbors"]:
            n_sector = get_sector(n, metadata)
            sector_mae[(n_sector, t_sector)].append(e["mae"])

    # Get unique sectors
    all_sectors = sorted(
        set(
            get_sector(s, metadata)
            for s in targets
            if get_sector(s, metadata) != "Unknown"
        )
    )
    neighbor_sectors = sorted(
        set(
            get_sector(n, metadata)
            for n in neighbor_counts
            if get_sector(n, metadata) != "Unknown"
        )
    )
    all_sectors_union = sorted(set(all_sectors + neighbor_sectors))

    # Build matrix: lower MAE = stronger predictive power
    # Normalize: score = 1 - (mae / global_max_mae) → higher = better predictor
    global_avg_mae = statistics.mean(e["mae"] for e in experiments)

    sector_matrix = []
    for n_sec in all_sectors_union:
        row = []
        for t_sec in all_sectors_union:
            key = (n_sec, t_sec)
            if key in sector_mae and len(sector_mae[key]) >= 2:
                avg = statistics.mean(sector_mae[key])
                count = len(sector_mae[key])
                # Score: relative improvement vs global avg (inverted, scaled)
                score = round((global_avg_mae - avg) / global_avg_mae * 100, 2)
                row.append({"avg_mae": round(avg, 6), "count": count, "score": score})
            else:
                row.append(None)
        sector_matrix.append(row)

    # ── 4. Network Graph (top predictor→target edges) ─────────────────────────
    # Weighted by: times appeared as neighbor × (1/mae improvement)
    edge_data = defaultdict(lambda: {"count": 0, "mae_sum": 0.0})

    for e in experiments:
        for n in e["neighbors"]:
            key = (n, e["target"])
            edge_data[key]["count"] += 1
            edge_data[key]["mae_sum"] += e["mae"]

    # Keep top edges by count
    edges_sorted = sorted(
        [
            {
                "source": k[0],
                "target": k[1],
                "count": v["count"],
                "avg_mae": round(v["mae_sum"] / v["count"], 6),
                "source_sector": get_sector(k[0], metadata),
                "target_sector": get_sector(k[1], metadata),
            }
            for k, v in edge_data.items()
        ],
        key=lambda x: -x["count"],
    )[:150]  # top 150 edges for network

    # Collect nodes from top edges
    node_symbols = set()
    for e in edges_sorted:
        node_symbols.add(e["source"])
        node_symbols.add(e["target"])

    nodes = [
        {
            "id": sym,
            "sector": get_sector(sym, metadata),
            "is_target": sym in targets,
            "is_predictor": sym in neighbor_counts,
            "predictor_count": neighbor_counts.get(sym, 0),
        }
        for sym in node_symbols
    ]

    # ── 5. MAE Distribution per Target ───────────────────────────────────────
    mae_by_target = {}
    for e in experiments:
        mae_by_target.setdefault(e["target"], []).append(e["mae"])

    mae_distribution = [
        {
            "target": t,
            "sector": get_sector(t, metadata),
            "mean_mae": round(statistics.mean(maes), 6),
            "min_mae": round(min(maes), 6),
            "max_mae": round(max(maes), 6),
            "std_mae": round(statistics.stdev(maes) if len(maes) > 1 else 0, 6),
            "count": len(maes),
        }
        for t, maes in sorted(
            mae_by_target.items(), key=lambda x: statistics.mean(x[1])
        )
    ]

    # ── 6. Top Cross-Sector Predictive Pairs ─────────────────────────────────
    # Best predictor→target where they are from DIFFERENT sectors
    cross_sector = [
        e for e in edges_sorted if e["source_sector"] != e["target_sector"]
    ][:30]

    # ── 7. Summary Stats ─────────────────────────────────────────────────────
    all_maes = [e["mae"] for e in calibrated_h]
    beats_all = sum(1 for e in calibrated_h if e.get("beats_baseline_zero"))

    # Per-target beats baseline: does the BEST experiment for each target beat zero?
    best_beats_n = sum(1 for b in best_list if b.get("beats_baseline_zero"))
    best_beats_pct = round(100.0 * best_beats_n / len(best_list), 2) if best_list else None

    summary = {
        "total_experiments": total,
        "total_targets": len(targets),
        "total_unique_predictors": len(neighbor_counts),
        "global_avg_mae": round(statistics.mean(all_maes), 6),
        "global_min_mae": round(min(all_maes), 6),
        "global_max_mae": round(max(all_maes), 6),
        "sectors_covered": len(all_sectors_union),
        "horizon_days": horizon_days,
        "calibrated_experiment_count": len(experiments),
        "calibrated_horizon_matched_count": len(calibrated_h),
        "calibrated_beats_baseline_all_count": beats_all,
        "calibrated_beats_baseline_all_pct": round(100.0 * beats_all / len(calibrated_h), 2)
        if calibrated_h
        else None,
        "calibrated_beats_baseline_best_per_target_count": best_beats_n,
        "calibrated_beats_baseline_best_per_target_pct": best_beats_pct,
        "global_avg_mae_calibrated": round(statistics.mean(all_maes), 6)
        if all_maes
        else None,
    }

    # ── 8. Price History per Target ───────────────────────────────────────────
    print("  Loading price history for each target…")
    # Build lookup: target → best predicted_return
    best_pred_return = {b["target"]: b["predicted_return"] for b in best_list}
    best_mae_map = {b["target"]: b["best_mae"] for b in best_list}
    best_neighbors_map = {b["target"]: b["best_neighbors"] for b in best_list}

    stocks_history = {}
    for sym in sorted(targets):
        history = load_price_history(sym, days=365)
        stocks_history[sym] = {
            "sector": get_sector(sym, metadata),
            "history": history,
            "predicted_return": best_pred_return.get(sym),
            "best_mae": best_mae_map.get(sym),
            "best_neighbors": best_neighbors_map.get(sym, []),
        }

    # ── 9. Backtest Prediction vs Actual (80/20 test split) ─────────────────
    print("  Building backtest series (predicted vs actual on test split)…")
    backtest = {}
    for item in best_list:
        target = item["target"]
        backtest[target] = build_backtest_series(
            target=target,
            neighbors=item["best_neighbors"],
            horizon=horizon_days,
        )

    # ── 10. Model Benchmark (XGB vs LSTM) ───────────────────────────────────
    print("  Building model benchmark comparison…")
    model_benchmark = build_model_benchmark(experiments, metadata, horizon_days)

    return {
        "summary": summary,
        "predictor_frequency": predictor_frequency,
        "best_per_target": best_list,
        "sector_matrix": {
            "labels": all_sectors_union,
            "data": sector_matrix,
        },
        "network": {
            "nodes": nodes,
            "edges": edges_sorted,
        },
        "mae_distribution": mae_distribution,
        "cross_sector_pairs": cross_sector,
        "stocks_history": stocks_history,
        "backtest": {
            "horizon_days": horizon_days,
            "by_target": backtest,
        },
        "baseline_comparison": build_baseline_comparison(experiments, metadata, horizon_days),
        "model_benchmark": model_benchmark,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID (e.g. h5, h10). Reads experiments_{id}.json, writes analysis_{id}.json. Omit for default experiments.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exp_file, out_file = _paths(args.run_id)
    run_label = f" [run={args.run_id}]" if args.run_id else ""

    print(f"Loading data{run_label}...")
    experiments, metadata = load_data(exp_file)
    print(f"  {len(experiments)} experiments, {len(metadata)} stocks in metadata")

    # Derive horizon from the data (most common horizon)
    horizons = [e.get("horizon", 5) for e in experiments]
    if horizons:
        from collections import Counter as _Counter
        horizon_days = _Counter(horizons).most_common(1)[0][0]
    else:
        horizon_days = 5

    print(f"  Detected horizon: {horizon_days} days")

    print("Analyzing…")
    result = analyze(experiments, metadata, horizon_days)

    s = result["summary"]
    print(f"\n── Summary ──────────────────────────────")
    print(f"  Total experiments  : {s['total_experiments']}")
    print(f"  Unique targets     : {s['total_targets']}")
    print(f"  Unique predictors  : {s['total_unique_predictors']}")
    print(f"  Global avg MAE     : {s['global_avg_mae']}")
    print(f"  Sectors covered    : {s['sectors_covered']}")

    print(f"\n── Top 10 Predictor Stocks ──────────────")
    for p in result["predictor_frequency"][:10]:
        print(f"  {p['symbol']:15s} {p['count']:4d}x  [{p['sector']}]")

    print(f"\n── Top 5 Best-Predicted Targets ─────────")
    for b in result["best_per_target"][:5]:
        print(
            f"  {b['target']:15s} MAE={b['best_mae']:.6f}  neighbors={b['best_neighbors']}"
        )

    print(f"\nWriting to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print("Done ✓")
