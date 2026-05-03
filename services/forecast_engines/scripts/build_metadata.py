# build_metadata.py

import yfinance as yf
import pandas as pd
import os
import json
import numpy as np

RAW_DIR = "../data/raw"
METADATA_PATH = "../data/metadata/metadata.json"
NIFTY_SYMBOL = "^NSEI"


# -------------------------------
# Utility Functions
# -------------------------------

def safe_float(val):
    if pd.isna(val):
        return None
    return float(val)

def percentile_bucket(value, all_values, labels):
    percentile = (all_values < value).mean()

    if percentile >= 0.8:
        return labels[0]
    elif percentile >= 0.5:
        return labels[1]
    elif percentile >= 0.2:
        return labels[2]
    else:
        return labels[3]


def compute_drawdown(close):
    rolling_max = close.rolling(252).max()
    drawdown = (close - rolling_max) / rolling_max
    return drawdown.min(), drawdown.iloc[-1]


def compute_trend_regime(close):
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    current_price = close.iloc[-1]

    if current_price > ma50 > ma200:
        return "Bullish"
    elif current_price < ma50 < ma200:
        return "Bearish"
    else:
        return "Sideways"


# -------------------------------
# Main Builder
# -------------------------------

def build_metadata():

    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    price_df = pd.DataFrame()
    fundamentals = {}

    print("Loading price data...")

    for file in os.listdir(RAW_DIR):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "")
        df = pd.read_csv(f"{RAW_DIR}/{file}", header=[0, 1], index_col=0, parse_dates=True)

        price_df[symbol] = pd.to_numeric(df.iloc[:, 0], errors="coerce")

        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.info

        fundamentals[symbol] = {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "beta": info.get("beta")
        }

    returns = price_df.pct_change()

    print("Downloading NIFTY benchmark...")
    nifty = yf.download(NIFTY_SYMBOL, period="5y", progress=False)["Close"]
    nifty_returns = nifty.pct_change()

    volatility = returns.rolling(30).std().iloc[-1]
    correlation = returns.corr()

    all_vols = volatility.dropna()
    all_mcaps = pd.Series(
        [fundamentals[s]["market_cap"] for s in fundamentals if fundamentals[s]["market_cap"]]
    )

    metadata = {}

    print("Computing metadata...")

    for stock in price_df.columns:

        close = price_df[stock].dropna()
        if len(close) < 250:
            continue

        vol = safe_float(volatility.get(stock, 0))
        mcap = fundamentals[stock]["market_cap"]

        # Static Correlation
        top_corr = (
            correlation[stock]
            .drop(stock)
            .sort_values(ascending=False)
            .head(5)
        )

        # Rolling Correlation
        rolling_corr_60 = (
            returns[stock]
            .rolling(60)
            .corr(returns)
            .iloc[-1]
            .drop(stock)
            .sort_values(ascending=False)
            .head(3)
        )

        rolling_corr_120 = (
            returns[stock]
            .rolling(120)
            .corr(returns)
            .iloc[-1]
            .drop(stock)
            .sort_values(ascending=False)
            .head(3)
        )

        # Momentum
        momentum_20d = safe_float(close.pct_change(20).iloc[-1])
        momentum_60d = safe_float(close.pct_change(60).iloc[-1])

        # Relative Strength
        stock_returns = close.pct_change()
        aligned = pd.concat([stock_returns, nifty_returns], axis=1, sort=False).dropna()

        if not aligned.empty:
            rs = (
                (1 + aligned.iloc[:, 0]).cumprod()
                / (1 + aligned.iloc[:, 1]).cumprod()
            )
            relative_strength = safe_float(rs.iloc[-1])
        else:
            relative_strength = None

        # Drawdown
        max_dd, current_dd = compute_drawdown(close)
        # Trend Regime
        regime = compute_trend_regime(close)

        metadata[stock] = {
            "sector": fundamentals[stock]["sector"],
            "industry": fundamentals[stock]["industry"],
            "market_cap": mcap,
            "market_cap_bucket": percentile_bucket(
                mcap, all_mcaps,
                ["UltraMega", "Mega", "Large", "Mid"]
            ) if mcap else "Unknown",
            "beta": fundamentals[stock]["beta"],

            "volatility_30d": vol,
            "volatility_bucket": percentile_bucket(
                vol, all_vols,
                ["High", "Medium", "Low", "VeryLow"]
            ),

            "momentum_20d": momentum_20d,
            "momentum_60d": momentum_60d,
            "relative_strength_vs_nifty": relative_strength,

            "max_drawdown_1y": safe_float(max_dd),
            "current_drawdown": safe_float(current_dd),

            "trend_regime": regime,

            "top_correlated": [
                {"symbol": s, "corr": safe_float(c)}
                for s, c in top_corr.items()
            ],

            "rolling_corr_60d": [
                {"symbol": s, "corr": safe_float(c)}
                for s, c in rolling_corr_60.items()
            ],

            "rolling_corr_120d": [
                {"symbol": s, "corr": safe_float(c)}
                for s, c in rolling_corr_120.items()
            ]
        }

    # Build Similarity Sets
    for stock in metadata:

        metadata[stock]["same_sector"] = [
            s for s in metadata
            if metadata[s]["sector"] == metadata[stock]["sector"]
            and s != stock
        ]

        metadata[stock]["same_industry"] = [
            s for s in metadata
            if metadata[s]["industry"] == metadata[stock]["industry"]
            and s != stock
        ]

        metadata[stock]["same_market_cap_bucket"] = [
            s for s in metadata
            if metadata[s]["market_cap_bucket"] == metadata[stock]["market_cap_bucket"]
            and s != stock
        ]

        metadata[stock]["same_volatility_bucket"] = [
            s for s in metadata
            if metadata[s]["volatility_bucket"] == metadata[stock]["volatility_bucket"]
            and s != stock
        ]

        metadata[stock]["same_trend_regime"] = [
            s for s in metadata
            if metadata[s]["trend_regime"] == metadata[stock]["trend_regime"]
            and s != stock
        ]

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Dynamic metadata built successfully.")


if __name__ == "__main__":
    build_metadata()