import yfinance as yf
import os
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
DATA_DIR = "../data/raw"

def download_prices():
    os.makedirs(DATA_DIR, exist_ok=True)

    end = datetime.today()
    start = end - timedelta(days=5 * 365)

    for symbol in NIFTY_50:
        print(f"Downloading {symbol}")
        df = yf.download(symbol, start=start, end=end, progress=False)

        if df.empty:
            continue

        df[['Close']].dropna().to_csv(
            f"{DATA_DIR}/{symbol.replace('.NS','')}.csv"
        )

if __name__ == "__main__":
    download_prices()