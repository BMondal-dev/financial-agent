from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
import json
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from experiment_logger import log_experiment

app = FastAPI()

DATA_DIR = "../data/raw"
METADATA_PATH = "../data/metadata/metadata.json"
EXPERIMENT_PATH = "../data/experiments.json"
GRAPH_PATH = "../data/stock_graph.json"


class ForecastRequest(BaseModel):
    target: str
    neighbors: list[str]
    horizon: int


# -----------------------------
# Utilities
# -----------------------------

def symbol_exists(symbol: str):
    path = f"{DATA_DIR}/{symbol}.csv"
    return os.path.exists(path)


# -----------------------------
# Data Loader
# -----------------------------

def load_returns(symbol: str):
    path = f"{DATA_DIR}/{symbol}.csv"

    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        names=["Date", "Close"]
    )

    df["return"] = df["Close"].pct_change()

    return df[["return"]].dropna()


# -----------------------------
# Feature Engineering
# -----------------------------

def build_dataset(target: str, neighbors: list[str], horizon: int):

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

    X = target_df.drop(columns=["return", "future_return"])
    y = target_df["future_return"]

    return X, y, valid_neighbors


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

    X, y, valid_neighbors = build_dataset(
        req.target,
        req.neighbors,
        req.horizon
    )

    if len(X) < 20:
        return {
            "target": req.target,
            "neighbors": valid_neighbors,
            "error": "Not enough data to train model"
        }

    split = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    # retrain on full dataset
    model.fit(X, y)

    latest = X.iloc[-1:]

    prediction = model.predict(latest)[0]

    # log experiment
    log_experiment(
        req.target,
        valid_neighbors,
        float(mae),
        float(prediction)
    )

    return {
        "target": req.target,
        "neighbors": valid_neighbors,
        "mae": float(mae),
        "predicted_return": float(prediction)
    }


# -----------------------------
# Tool 4 — Best Neighbors
# -----------------------------

@app.get("/best-neighbors/{symbol}")
def best_neighbors(symbol: str):

    if not os.path.exists(EXPERIMENT_PATH):
        return []

    with open(EXPERIMENT_PATH) as f:
        data = json.load(f)

    # filter target
    filtered = [x for x in data if x["target"] == symbol]

    if not filtered:
        return []

    # remove duplicate neighbor combinations
    unique = {}

    for exp in filtered:

        key = tuple(sorted(exp["neighbors"]))

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
        neighbors.append({
            "symbol": n,
            "weight": G[symbol][n]["weight"]
        })

    neighbors.sort(key=lambda x: x["weight"], reverse=True)

    return neighbors[:5]