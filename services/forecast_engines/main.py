from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
import pandas as pd
import json
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from experiment_logger import log_experiment
from typing import Literal


app = FastAPI()

DATA_DIR = "../data/raw"
METADATA_PATH = "../data/metadata/metadata.json"
EXPERIMENT_PATH = "../data/experiments.json"
GRAPH_PATH = "../data/stock_graph.json"

BASELINE_RANKING_PENALTY = 1.0
LEGACY_EXPERIMENT_HORIZON = 5


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
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return str(x)


# -----------------------------
# Data Loader
# -----------------------------

def load_returns(symbol: str):
    path = f"{DATA_DIR}/{symbol}.csv"
    df = pd.read_csv(path, skiprows=3, header=None, names=["Date", "Close"],)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df.set_index("Date", inplace=True)
    df["return"] = df["Close"].pct_change()

    return df[["return"]].dropna()


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

    valid_neighbors = []

    for neighbor in neighbors:

        # remove self-neighbor
        if neighbor == target:
            continue
        if not symbol_exists(neighbor):
            print(f"Skipping {neighbor} (not in dataset)")
            continue

        neighbor_df = load_returns(neighbor)

        target_df[f"{neighbor}_lag_1"] = neighbor_df["return"].shift(1)

        valid_neighbors.append(neighbor)

    target_df["future_return"] = target_df["return"].shift(-horizon)

    target_df = target_df.dropna()
    dates = target_df.index.to_series()

    X = target_df.drop(columns=["return", "future_return"])
    y = target_df["future_return"]

    return X, y, valid_neighbors, dates


def time_series_cv_mae(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 2
) -> tuple[float | None, float | None]:
    if len(X) < 60:
        return None, None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        if len(test_idx) < 5:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        maes.append(mean_absolute_error(y_te, pred))
    if not maes:
        return None, None
    return float(np.mean(maes)), float(np.std(maes))





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

    data = metadata[symbol]

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

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    dates_train, dates_test = dates.iloc[:split], dates.iloc[split:]

    mae_baseline_zero = float(mean_absolute_error(y_test, np.zeros(len(y_test))))
    train_mean = float(y_train.mean())
    mae_baseline_mean = float(
        mean_absolute_error(y_test, np.full(len(y_test), train_mean))
    )

    model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    beats_baseline_zero = mae < mae_baseline_zero
    mae_for_ranking = mae if beats_baseline_zero else mae + BASELINE_RANKING_PENALTY

    if _skip_time_series_cv():
        cv_mean, cv_std = None, None
    else:
        cv_mean, cv_std = time_series_cv_mae(X, y, n_splits=2)

    model.fit(X, y)

    latest = X.iloc[-1:]

    prediction = float(model.predict(latest)[0])

    log_experiment(
        req.target,
        valid_neighbors,
        req.horizon,
        mae,
        prediction,
        mae_baseline_zero=mae_baseline_zero,
        mae_baseline_mean=mae_baseline_mean,
        beats_baseline_zero=beats_baseline_zero,
        mae_for_ranking=mae_for_ranking,
        cv_mae_mean=cv_mean,
        cv_mae_std=cv_std,
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
        "mae": mae,
        "mae_baseline_zero": mae_baseline_zero,
        "mae_baseline_mean": mae_baseline_mean,
        "beats_baseline_zero": beats_baseline_zero,
        "mae_for_ranking": mae_for_ranking,
        "cv_mae_mean": cv_mean,
        "cv_mae_std": cv_std,
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
    if not os.path.exists(EXPERIMENT_PATH):
        return []

    with open(EXPERIMENT_PATH) as f:
        data = json.load(f)

    filtered = [x for x in data if x["target"] == symbol]
    if horizon is not None:
        filtered = [
            x
            for x in filtered
            if x.get("horizon", LEGACY_EXPERIMENT_HORIZON) == horizon
        ]

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

