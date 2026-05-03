"""
Experiment Analysis Script
Analyzes experiments.json and generates a rich analysis JSON for the dashboard.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

BASE = Path(__file__).parent.parent.parent / "data"
EXPERIMENTS_FILE = BASE / "experiments.json"
METADATA_FILE = BASE / "metadata" / "metadata.json"
RAW_DIR = BASE / "raw"
OUTPUT_FILE = BASE / "analysis.json"
HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "5"))


def load_data():
    with open(EXPERIMENTS_FILE) as f:
        experiments = json.load(f)
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    return experiments, metadata


def load_returns_with_date(symbol: str) -> pd.DataFrame:
    """Load date + return series aligned with services/forecast_engines/main.py."""
    path = RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Date", "return"])

    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close"],
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df["return"] = df["Close"].pct_change()
    df = df.dropna(subset=["return"])
    return df[["Date", "return"]].reset_index(drop=True)


def build_backtest_series(target: str, neighbors: list[str], horizon: int) -> dict:
    """Recreate 80/20 test predictions (predicted vs actual) for one target."""
    target_df = load_returns_with_date(target).rename(columns={"return": "target_return"})
    if target_df.empty:
        return {"rows": [], "mae": None, "count": 0}

    merged = target_df
    valid_neighbors: list[str] = []
    for neighbor in neighbors:
        if neighbor == target:
            continue
        ndf = load_returns_with_date(neighbor).rename(
            columns={"return": f"{neighbor}_return"}
        )
        if ndf.empty:
            continue
        merged = merged.merge(ndf, on="Date", how="inner")
        valid_neighbors.append(neighbor)

    merged = merged.sort_values("Date").reset_index(drop=True)
    tr = merged["target_return"]

    for lag in range(1, 6):
        merged[f"target_lag_{lag}"] = tr.shift(lag)

    merged["rolling_mean_10"] = tr.rolling(10).mean()
    merged["rolling_std_10"] = tr.rolling(10).std()

    for n in valid_neighbors:
        merged[f"{n}_lag_1"] = merged[f"{n}_return"].shift(1)

    merged["future_return"] = tr.shift(-horizon)
    merged = merged.dropna().reset_index(drop=True)

    if len(merged) < 20:
        return {"rows": [], "mae": None, "count": 0}

    drop_cols = ["Date", "target_return", "future_return"] + [
        f"{n}_return" for n in valid_neighbors
    ]
    X = merged.drop(columns=drop_cols)
    y = merged["future_return"]
    dates = merged["Date"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
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


def is_calibrated_experiment(e: dict) -> bool:
    return "mae_baseline_zero" in e


def analyze(experiments, metadata):
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

    for e in experiments:
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
            }
            for t, v in best_per_target.items()
        ],
        key=lambda x: x["best_mae"],
    )

    # ── 2b. Calibrated-only leaderboard (date-aligned pipeline + baselines) ──
    calibrated = [e for e in experiments if is_calibrated_experiment(e)]
    calibrated_h = [
        e for e in calibrated if e.get("horizon", HORIZON_DAYS) == HORIZON_DAYS
    ]
    best_per_target_cal: dict = {}
    worst_per_target_cal: dict = {}
    for e in calibrated_h:
        t = e["target"]
        if t not in best_per_target_cal or e["mae"] < best_per_target_cal[t]["mae"]:
            best_per_target_cal[t] = e
        if t not in worst_per_target_cal or e["mae"] > worst_per_target_cal[t]["mae"]:
            worst_per_target_cal[t] = e

    best_list_calibrated = sorted(
        [
            {
                "target": t,
                "sector": get_sector(t, metadata),
                "best_mae": round(v["mae"], 6),
                "worst_mae": round(worst_per_target_cal[t]["mae"], 6),
                "best_neighbors": v["neighbors"],
                "predicted_return": round(v["predicted_return"] * 100, 4),
                "beats_baseline_zero": v.get("beats_baseline_zero"),
                "mae_baseline_zero": round(float(v["mae_baseline_zero"]), 6)
                if v.get("mae_baseline_zero") is not None
                else None,
            }
            for t, v in best_per_target_cal.items()
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
    all_maes = [e["mae"] for e in experiments]
    legacy_count = sum(1 for e in experiments if not is_calibrated_experiment(e))
    beats_n = sum(1 for e in calibrated_h if e.get("beats_baseline_zero"))
    summary = {
        "total_experiments": total,
        "total_targets": len(targets),
        "total_unique_predictors": len(neighbor_counts),
        "global_avg_mae": round(statistics.mean(all_maes), 6),
        "global_min_mae": round(min(all_maes), 6),
        "global_max_mae": round(max(all_maes), 6),
        "sectors_covered": len(all_sectors_union),
        "horizon_days": HORIZON_DAYS,
        "legacy_experiment_count": legacy_count,
        "calibrated_experiment_count": len(calibrated),
        "calibrated_horizon_matched_count": len(calibrated_h),
        "calibrated_beats_baseline_count": beats_n,
        "calibrated_beats_baseline_pct": round(100.0 * beats_n / len(calibrated_h), 2)
        if calibrated_h
        else None,
        "global_avg_mae_calibrated": round(
            statistics.mean(e["mae"] for e in calibrated_h), 6
        )
        if calibrated_h
        else None,
        "methodology_note": (
            "Mixed pool: legacy experiments used row-aligned CSV joins; "
            "calibrated rows use calendar-aligned features and log MAE vs a zero-return "
            "baseline on the same holdout. Prefer the Calibrated leaderboard for apples-to-apples MAE."
        ),
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
            horizon=HORIZON_DAYS,
        )

    print("  Building calibrated backtests (best config per target, calibrated pool only)…")
    backtest_calibrated: dict = {}
    for item in best_list_calibrated:
        tgt = item["target"]
        backtest_calibrated[tgt] = build_backtest_series(
            target=tgt,
            neighbors=item["best_neighbors"],
            horizon=HORIZON_DAYS,
        )

    return {
        "summary": summary,
        "predictor_frequency": predictor_frequency,
        "best_per_target": best_list,
        "best_per_target_calibrated": best_list_calibrated,
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
            "horizon_days": HORIZON_DAYS,
            "by_target": backtest,
        },
        "backtest_calibrated": {
            "horizon_days": HORIZON_DAYS,
            "by_target": backtest_calibrated,
        },
    }


if __name__ == "__main__":
    print("Loading data...")
    experiments, metadata = load_data()
    print(f"  {len(experiments)} experiments, {len(metadata)} stocks in metadata")

    print("Analyzing…")
    result = analyze(experiments, metadata)

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

    print(f"\nWriting to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print("Done ✓")
