from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import pandas as pd
import json
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from experiment_logger import log_experiment, _experiment_path
from models import get_model, ModelType
from typing import Literal


app = FastAPI()

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "../data/raw")
METADATA_PATH = os.path.join(_BASE_DIR, "../data/metadata/metadata.json")
GRAPH_PATH = os.path.join(_BASE_DIR, "../data/stock_graph.json")

BASELINE_RANKING_PENALTY = 1.0


def _skip_time_series_cv() -> bool:
    """Bulk runs: set FORECAST_ENGINES_SKIP_CV=1 to skip extra XGB fits (~2× per /run-forecast)."""
    return os.environ.get("FORECAST_ENGINES_SKIP_CV", "").lower() in (
        "1",
        "true",
        "yes",
    )


class ForecastRequest(BaseModel):
    target: str
    neighbors: list[str]
    horizon: int
    model_type: ModelType = "xgb"
    experiment_id: str | None = None
    rationale: str | None = None
    source: str | None = None
    orchestrator_round: int | None = None


# -----------------------------
# Utilities
# -----------------------------


def symbol_exists(symbol: str):
    path = f"{DATA_DIR}/{symbol}.csv"
    return os.path.exists(path)


def _ts_iso(x) -> str | None:
    """Convert a datetime-like value to an ISO 8601 string for JSON serialization.

    Used when logging experiment date ranges so that pandas Timestamps are stored
    as clean ISO strings (e.g. "2024-01-15T00:00:00") rather than raw Python
    representations.

    Args:
        x: A datetime, pd.Timestamp, None, or any other value.

    Returns:
        An ISO 8601 string, or None if the input is missing/NaN.
    """
    # Return None for missing values
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    # Use .isoformat() for datetime-like objects (pd.Timestamp, datetime.datetime)
    if hasattr(x, "isoformat"):
        return x.isoformat()
    # Fallback for unexpected types
    return str(x)


# -----------------------------
# Data Loader
# -----------------------------


def load_returns(symbol: str):
    """Load a stock's price data from CSV and compute daily percentage returns.

    Reads the raw Yahoo Finance CSV (skipping the 3-row header metadata), parses
    dates and closing prices, and calculates the day-over-day percentage change.

    Args:
        symbol: The stock ticker (e.g. "BEL").

    Returns:
        A DataFrame indexed by date with "return" and "Close" columns.
        The first row is dropped since there is no prior close to compute a return.
    """
    path = f"{DATA_DIR}/{symbol}.csv"
    # Skip the 3-row Yahoo Finance header; treat remaining rows as [Date, Close]
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close"],
    )
    # Parse types, coercing malformed values to NaN
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    # Drop rows with missing dates or prices
    df = df.dropna(subset=["Date", "Close"])
    # Use date as the index
    df.set_index("Date", inplace=True)
    # Compute daily percentage change: Close(t) / Close(t-1) - 1
    df["return"] = df["Close"].pct_change()

    return df[["Close", "return"]].dropna()


# -----------------------------
# Feature Engineering
# -----------------------------


def build_dataset(
    target: str, neighbors: list[str], horizon: int
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    target_df = load_returns(target)

    # Target lag features
    for lag in range(1, 6):
        target_df[f"target_lag_{lag}"] = target_df["return"].shift(lag)

    # Rolling stats
    target_df["rolling_mean_10"] = target_df["return"].rolling(10).mean()
    target_df["rolling_std_10"] = target_df["return"].rolling(10).std()

    # Compute future return on target-only data to fix the date range.
    # This ensures the test split (and mae_baseline_zero) is identical
    # regardless of which neighbor set is used.
    target_df["future_return"] = target_df["Close"].shift(-horizon) / target_df["Close"] - 1

    # Determine valid date range from target-only features before adding neighbors
    target_valid = target_df.dropna()

    valid_neighbors = []

    for neighbor in neighbors:
        # remove self-neighbor
        if neighbor == target:
            continue
        if not symbol_exists(neighbor):
            print(f"Skipping {neighbor} (not in dataset)")
            continue

        neighbor_df = load_returns(neighbor)

        # Reindex neighbor to target's date index, then only keep rows valid for target
        nr = neighbor_df["return"].reindex(target_valid.index)

        # Immediate 1-day lagged spillover effect
        target_df[f"{neighbor}_lag_1"] = nr.shift(1)

        # 5-day rolling momentum (lagged to prevent look-ahead bias)
        target_df[f"{neighbor}_rolling_mean_5"] = (
            nr.shift(1).rolling(5).mean()
        )

        valid_neighbors.append(neighbor)

    # Only drop rows where neighbor features are NaN — date range is already fixed
    # by target_valid
    target_df = target_df.loc[target_valid.index].dropna()
    dates = target_df.index.to_series()

    X = target_df.drop(columns=["Close", "return", "future_return"])
    y = target_df["future_return"]

    return X, y, valid_neighbors, dates


def time_series_cv_mae(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    horizon: int,
    n_splits: int = 2,
    model_type: ModelType = "xgb",
) -> dict:
    """Run time-series cross-validation and return MAE statistics plus worst-split details.

    A purge gap equal to ``horizon`` is inserted between train and test folds to
    prevent look-ahead leakage: because ``y[t] = cumulative_return(t, t+horizon)``,
    the training labels depend on prices up to ``t+horizon``. The last ``horizon``
    rows of any training fold would otherwise leak labels that fall inside the
    immediately-following test fold.

    Returns a dict with:
      - cv_mae_mean: mean MAE across splits
      - cv_mae_std: std dev of MAE across splits
      - cv_worst_split_mae: MAE of the worst-performing split
      - cv_worst_split_test_date_start / _end: date range of the worst split
    Returns all-None dict if there is insufficient data.
    """
    empty_result = {
        "cv_mae_mean": None,
        "cv_mae_std": None,
        "cv_worst_split_mae": None,
        "cv_worst_split_test_date_start": None,
        "cv_worst_split_test_date_end": None,
    }

    if len(X) < 60:
        return empty_result

    # gap=horizon purges overlapping labels between train and test folds.
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    split_results: list[dict] = []

    for train_idx, test_idx in tscv.split(X):
        if len(test_idx) < 5:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model = get_model(model_type)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        split_mae = float(mean_absolute_error(y_te, pred))
        split_results.append(
            {
                "mae": split_mae,
                "test_date_start": _ts_iso(dates.iloc[test_idx[0]]),
                "test_date_end": _ts_iso(dates.iloc[test_idx[-1]]),
            }
        )

    if not split_results:
        return empty_result

    maes = [s["mae"] for s in split_results]
    worst = max(split_results, key=lambda s: s["mae"])

    return {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_worst_split_mae": worst["mae"],
        "cv_worst_split_test_date_start": worst["test_date_start"],
        "cv_worst_split_test_date_end": worst["test_date_end"],
    }


# -----------------------------
# Tool 1 — Metadata
# -----------------------------


@app.get("/metadata/{symbol}")
def get_metadata(symbol: str):
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    return metadata.get(symbol)


# -----------------------------
# Tool 2 — Candidate Neighbors
# -----------------------------


@app.get("/candidate-neighbors/{symbol}")
def candidate_neighbors(symbol: str):
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    data = metadata.get(symbol)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found in metadata",
        )

    neighbors = set()

    neighbors.update(data["same_sector"])
    neighbors.update(data["same_market_cap_bucket"])
    neighbors.update([x["symbol"] for x in data["top_correlated"]])

    return list(neighbors)


# -----------------------------
# Tool 3 — Run Forecast
# -----------------------------


@app.post("/run-forecast")
def run_forecast(req: ForecastRequest):
    X, y, valid_neighbors, dates = build_dataset(
        req.target,
        req.neighbors,
        req.horizon,
    )

    if len(X) < 20:
        return {
            "target": req.target,
            "neighbors": valid_neighbors,
            "horizon": req.horizon,
            "error": "Not enough data to train model",
        }

    split = int(len(X) * 0.8)
    train_end = split - req.horizon
    if split >= len(X) or train_end < 1:
        return {
            "target": req.target,
            "neighbors": valid_neighbors,
            "horizon": req.horizon,
            "error": "Not enough data to train model",
        }

    X_train, X_test = X.iloc[:train_end], X.iloc[split:]
    y_train, y_test = y.iloc[:train_end], y.iloc[split:]
    dates_train, dates_test = dates.iloc[:train_end], dates.iloc[split:]

    mae_baseline_zero = float(mean_absolute_error(y_test, np.zeros(len(y_test))))
    train_mean = float(y_train.mean())
    mae_baseline_mean = float(
        mean_absolute_error(y_test, np.full(len(y_test), train_mean))
    )

    model = get_model(req.model_type)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    beats_baseline_zero = mae < mae_baseline_zero
    mae_for_ranking = mae if beats_baseline_zero else mae + BASELINE_RANKING_PENALTY

    if _skip_time_series_cv():
        cv_results = {
            "cv_mae_mean": None,
            "cv_mae_std": None,
            "cv_worst_split_mae": None,
            "cv_worst_split_test_date_start": None,
            "cv_worst_split_test_date_end": None,
        }
    else:
        cv_results = time_series_cv_mae(X, y, dates, horizon=req.horizon, n_splits=2, model_type=req.model_type)

    model_final = get_model(req.model_type)
    model_final.fit(X, y)

    latest = X.iloc[-1:]

    prediction = float(model_final.predict(latest)[0])

    log_experiment(
        req.target,
        valid_neighbors,
        req.horizon,
        mae,
        prediction,
        run_id=f"h{req.horizon}",
        model_type=req.model_type,
        mae_baseline_zero=mae_baseline_zero,
        mae_baseline_mean=mae_baseline_mean,
        beats_baseline_zero=beats_baseline_zero,
        mae_for_ranking=mae_for_ranking,
        cv_mae_mean=cv_results["cv_mae_mean"],
        cv_mae_std=cv_results["cv_mae_std"],
        cv_worst_split_mae=cv_results["cv_worst_split_mae"],
        cv_worst_split_test_date_start=cv_results["cv_worst_split_test_date_start"],
        cv_worst_split_test_date_end=cv_results["cv_worst_split_test_date_end"],
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
        train_date_start=_ts_iso(dates_train.iloc[0]),
        train_date_end=_ts_iso(dates_train.iloc[-1]),
        test_date_start=_ts_iso(dates_test.iloc[0]),
        test_date_end=_ts_iso(dates_test.iloc[-1]),
        experiment_id=req.experiment_id,
        rationale=req.rationale,
        source=req.source,
        orchestrator_round=req.orchestrator_round,
    )

    return {
        "target": req.target,
        "neighbors": valid_neighbors,
        "horizon": req.horizon,
        "model_type": req.model_type,
        "mae": mae,
        "mae_baseline_zero": mae_baseline_zero,
        "mae_baseline_mean": mae_baseline_mean,
        "beats_baseline_zero": beats_baseline_zero,
        "mae_for_ranking": mae_for_ranking,
        "cv_mae_mean": cv_results["cv_mae_mean"],
        "cv_mae_std": cv_results["cv_mae_std"],
        "cv_worst_split_mae": cv_results["cv_worst_split_mae"],
        "cv_worst_split_test_date_start": cv_results["cv_worst_split_test_date_start"],
        "cv_worst_split_test_date_end": cv_results["cv_worst_split_test_date_end"],
        "predicted_return": prediction,
        "eval": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_date_start": _ts_iso(dates_train.iloc[0]),
            "train_date_end": _ts_iso(dates_train.iloc[-1]),
            "test_date_start": _ts_iso(dates_test.iloc[0]),
            "test_date_end": _ts_iso(dates_test.iloc[-1]),
        },
    }


# -----------------------------
# Tool 4 — Best Neighbors
# -----------------------------


@app.get("/best-neighbors/{symbol}")
def best_neighbors(symbol: str, horizon: int | None = Query(default=None)):
    """Return past experiment results for a target, scoped to the horizon if provided."""
    # Use horizon-specific file when available, fall back to default
    run_id = f"h{horizon}" if horizon else None
    path = os.path.abspath(_experiment_path(run_id))
    if not os.path.exists(path):
        return []

    with open(path) as f:
        data = json.load(f)

    filtered = [x for x in data if x["target"] == symbol]

    if not filtered:
        return []

    unique = {}

    for exp in filtered:
        key = (tuple(sorted(exp["neighbors"])), exp.get("horizon"))

        if key not in unique or exp["mae"] < unique[key]["mae"]:
            unique[key] = exp

    best = sorted(unique.values(), key=lambda x: x["mae"])

    return best[:5]


@app.get("/graph-neighbors/{symbol}")
def graph_neighbors(symbol: str):
    import networkx as nx

    with open(GRAPH_PATH) as f:
        graph_data = json.load(f)

    G = nx.node_link_graph(graph_data)

    neighbors = []

    for n in G[symbol]:
        neighbors.append({"symbol": n, "weight": G[symbol][n]["weight"]})

    neighbors.sort(key=lambda x: x["weight"], reverse=True)

    return neighbors[:5]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
