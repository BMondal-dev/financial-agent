"""Shared forecasting evaluation (no FastAPI). Used by main.py and analyze_experiments."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

DATA_DIR = "../data/raw"
BASELINE_RANKING_PENALTY = 1.0
LEGACY_EXPERIMENT_HORIZON = 5


def _skip_time_series_cv() -> bool:
    return os.environ.get("FORECAST_ENGINES_SKIP_CV", "").lower() in (
        "1",
        "true",
        "yes",
    )


def symbol_exists(symbol: str) -> bool:
    return os.path.exists(f"{DATA_DIR}/{symbol}.csv")


def _ts_iso(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return str(x)


def load_returns(symbol: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/{symbol}.csv"
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
    return df[["Close", "return"]].dropna()


def build_dataset(
    target: str, neighbors: list[str], horizon: int
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    target_df = load_returns(target)

    for lag in range(1, 6):
        target_df[f"target_lag_{lag}"] = target_df["return"].shift(lag)

    target_df["rolling_mean_10"] = target_df["return"].rolling(10).mean()
    target_df["rolling_std_10"] = target_df["return"].rolling(10).std()

    valid_neighbors: list[str] = []

    for neighbor in neighbors:
        if neighbor == target:
            continue
        if not symbol_exists(neighbor):
            continue
        neighbor_df = load_returns(neighbor)
        neighbor_ret = neighbor_df["return"].reindex(target_df.index)
        target_df[f"{neighbor}_lag_1"] = neighbor_ret.shift(1)
        target_df[f"{neighbor}_rolling_mean_5"] = (
            neighbor_ret.shift(1).rolling(5).mean()
        )
        valid_neighbors.append(neighbor)

    target_df["future_return"] = target_df["Close"].shift(-horizon) / target_df["Close"] - 1
    target_df = target_df.dropna()
    dates = target_df.index.to_series()

    X = target_df.drop(columns=["Close", "return", "future_return"])
    y = target_df["future_return"]

    return X, y, valid_neighbors, dates


def time_series_cv_mae(
    X: pd.DataFrame, y: pd.Series, horizon: int, n_splits: int = 2
) -> tuple[float | None, float | None]:
    if len(X) < 60:
        return None, None
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
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


def evaluate_forecast(
    target: str,
    neighbors: list[str],
    horizon: int,
    *,
    include_cv: bool = True,
) -> dict[str, Any]:
    """Train XGB on 80/20 chronological split; return metrics without logging."""
    X, y, valid_neighbors, dates = build_dataset(target, neighbors, horizon)

    if len(X) < 20:
        return {
            "error": "Not enough data to train model",
            "target": target,
            "neighbors": valid_neighbors,
            "horizon": horizon,
        }

    split = int(len(X) * 0.8)
    train_end = split - horizon
    if split >= len(X) or train_end < 1:
        return {
            "error": "Not enough data to train model",
            "target": target,
            "neighbors": valid_neighbors,
            "horizon": horizon,
        }
    X_train, X_test = X.iloc[:train_end], X.iloc[split:]
    y_train, y_test = y.iloc[:train_end], y.iloc[split:]
    dates_train, dates_test = dates.iloc[:train_end], dates.iloc[split:]

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

    if include_cv and not _skip_time_series_cv():
        cv_mean, cv_std = time_series_cv_mae(X, y, horizon=horizon, n_splits=2)
    else:
        cv_mean, cv_std = None, None

    model.fit(X, y)
    latest = X.iloc[-1:]
    prediction = float(model.predict(latest)[0])

    return {
        "target": target,
        "neighbors": valid_neighbors,
        "horizon": horizon,
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


def pick_best_agent_neighbors_from_log(
    experiments: list[dict],
    target: str,
    horizon: int,
) -> tuple[list[str], dict | None]:
    """Best neighbor set from logged agent runs (by mae_for_ranking, then mae)."""
    cand = [
        e
        for e in experiments
        if e.get("target") == target
        and e.get("horizon", LEGACY_EXPERIMENT_HORIZON) == horizon
    ]
    agentish = [
        e
        for e in cand
        if e.get("source") in ("experiment-multiple", "experiment-agent")
        or e.get("experiment_id") is not None
    ]
    pool = agentish if agentish else [e for e in cand if "mae_for_ranking" in e]
    if not pool:
        return [], None

    def sort_key(e: dict) -> float:
        return float(e.get("mae_for_ranking", e.get("mae", float("inf"))))

    best = min(pool, key=sort_key)
    return list(best["neighbors"]), best
