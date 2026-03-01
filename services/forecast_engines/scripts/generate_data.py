import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timedelta

NIFTY_50 = [
    "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "RELIANCE.NS", "ITC.NS", "LT.NS", "SBIN.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "BHARTIARTL.NS",
    "ASIANPAINT.NS", "DMART.NS", "TITAN.NS", "SUNPHARMA.NS",
    "MARUTI.NS", "ULTRACEMCO.NS", "WIPRO.NS", "TECHM.NS",
    "HCLTECH.NS", "INDUSINDBK.NS", "BAJFINANCE.NS", "ADANIENT.NS",
    "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "ONGC.NS", "COALINDIA.NS", "IOC.NS", "BPCL.NS",
    "HINDALCO.NS", "VEDL.NS", "BRITANNIA.NS", "DIVISLAB.NS",
    "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "BAJAJFINSV.NS",
    "GRASIM.NS", "HEROMOTOCO.NS", "M&M.NS", "NESTLEIND.NS",
    "SHREECEM.NS", "UPL.NS"
]

SECTOR_MAP = {
    "INFY": "IT",
    "TCS": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",

    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "AXISBANK": "Banking",
    "KOTAKBANK": "Banking",
    "SBIN": "Banking",
    "INDUSINDBK": "Banking",

    "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services",

    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "IOC": "Energy",
    "BPCL": "Energy",
    "COALINDIA": "Energy",

    "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",
    "HINDALCO": "Metals",
    "VEDL": "Metals",

    "SUNPHARMA": "Pharma",
    "CIPLA": "Pharma",
    "DRREDDY": "Pharma",
    "DIVISLAB": "Pharma",

    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",

    "LT": "Infrastructure",
    "ULTRACEMCO": "Cement",
    "SHREECEM": "Cement",
    "GRASIM": "Cement",

    "MARUTI": "Auto",
    "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto",
    "M&M": "Auto",

    "BHARTIARTL": "Telecom",
    "ASIANPAINT": "Paints",
    "TITAN": "Consumer",
    "UPL": "Chemicals",
    "POWERGRID": "Utilities",
    "NTPC": "Utilities",
    "ADANIENT": "Conglomerate",
    "DMART": "Retail"
}

DATA_DIR = "../../data/stocks"
METADATA_PATH = "../../data/metadata.json"


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    end = datetime.today()
    start = end - timedelta(days=5 * 365)

    for symbol in NIFTY_50:
        print(f"Downloading {symbol}...")
        df = yf.download(symbol, start=start, end=end, progress=False)

        if df.empty:
            print(f"⚠️ Skipping {symbol} (no data)")
            continue

        df = df[['Close']].dropna()
        df.to_csv(f"{DATA_DIR}/{symbol.replace('.NS','')}.csv")


def generate_metadata():
    price_df = pd.DataFrame()

    for symbol in NIFTY_50:
        name = symbol.replace(".NS", "")
        path = f"{DATA_DIR}/{name}.csv"

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        # Convert Close column to numeric, coercing errors to NaN
        price_df[name] = pd.to_numeric(df["Close"], errors='coerce')

    returns = price_df.pct_change()

    volatility = returns.rolling(30).std().iloc[-1]
    correlation = returns.corr()

    metadata = {}

    for stock in price_df.columns:
        # Correlation neighbors
        top_corr = (
            correlation[stock]
            .drop(stock)
            .sort_values(ascending=False)
            .head(5)
        )

        # Sector neighbors
        sector = SECTOR_MAP.get(stock, "Unknown")
        same_sector = [
            s for s, sec in SECTOR_MAP.items()
            if sec == sector and s != stock
        ]

        metadata[stock] = {
            "sector": sector,
            "volatility_30d": float(volatility.get(stock, 0)),
            "top_correlated": [
                {"symbol": s, "corr": float(c)}
                for s, c in top_corr.items()
            ],
            "same_sector": same_sector
        }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Metadata generated successfully.")


if __name__ == "__main__":
    download_data()
    generate_metadata()