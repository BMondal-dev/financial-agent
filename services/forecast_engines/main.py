from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

app = FastAPI()

DATA_DIR = "../../data/stocks"

class ForecastRequest(BaseModel):
    target: str
    neighbors: list[str]
    horizon: int


def load_returns(symbol: str):
    path = f"{DATA_DIR}/{symbol}.csv"
    # Skip the first 3 rows of metadata and use the first row as header
    df = pd.read_csv(path, skiprows=3, header=None, names=['Date', 'Close'])
    df["return"] = df["Close"].pct_change()
    return df[["return"]].dropna()


def build_dataset(target: str, neighbors: list[str], horizon: int):
    target_df = load_returns(target)

    # ---- Target Lag Features ----
    for lag in range(1, 6):  # lag 1 to 5
        target_df[f"target_lag_{lag}"] = target_df["return"].shift(lag)

    # ---- Rolling Features ----
    target_df["rolling_mean_10"] = target_df["return"].rolling(10).mean()
    target_df["rolling_std_10"] = target_df["return"].rolling(10).std()

    # ---- Neighbor Features (lag1 only) ----
    for neighbor in neighbors:
        neighbor_df = load_returns(neighbor)
        target_df[f"{neighbor}_lag_1"] = neighbor_df["return"].shift(1)

    # ---- Future Target ----
    target_df["future_return"] = target_df["return"].shift(-horizon)

    # ---- Drop NA ----
    target_df = target_df.dropna()

    X = target_df.drop(columns=["return", "future_return"])
    y = target_df["future_return"]

    return X, y


@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        X, y = build_dataset(req.target, req.neighbors, req.horizon)

        split_idx = int(len(X) * 0.8)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        # -------- REAL FUTURE FORECAST --------
        # Train on full data
        model.fit(X, y)

        # Take latest row as input
        latest_features = X.iloc[-1:].copy()
        future_prediction = model.predict(latest_features)[0]

        return {
            "target": req.target,
            "neighbors": req.neighbors,
            "horizon": req.horizon,
            "mae": float(mae),
            "predicted_return": float(future_prediction)
        }

    except Exception as e:
        return {"error": str(e)}