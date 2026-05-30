"""
Download US equity data via yfinance and write the pickled dataset files
expected by dataset.py / train_predictor.py.

Usage:
    cd finetune
    python prepare_us_data.py            # uses default SP500 tickers
    python prepare_us_data.py --tickers AAPL MSFT NVDA TSLA SOFI

The script writes three files into ./data/processed_datasets/:
    train_dataset.pkl
    val_dataset.pkl
    test_dataset.pkl

Each file is a list of (x, x_stamp, y, y_stamp) numpy arrays matching the
format expected by QlibDataset, so train_predictor.py works unchanged.
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ── Default tickers (large-cap US, diverse sectors) ──────────────────────────
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "BAC", "ABBV", "CVX",
    "MRK", "COST", "PEP", "NFLX", "AMD", "SOFI", "PLTR", "COIN", "HOOD",
    "BA", "GS", "MS", "C", "WFC", "T", "VZ", "DIS", "INTC", "MU", "QCOM",
]

# ── Config (mirrors finetune/config.py defaults) ──────────────────────────────
LOOKBACK   = 90
PRED_WIN   = 10
CLIP       = 5.0
TRAIN_END  = "2023-12-31"
VAL_START  = "2023-09-01"
VAL_END    = "2024-09-30"
TEST_START = "2024-07-01"
OUT_DIR    = "./data/processed_datasets"


def fetch_daily(ticker: str) -> pd.DataFrame | None:
    try:
        raw = yf.download(ticker, start="2018-01-01", interval="1d",
                          auto_adjust=True, progress=False)
        raw = raw.dropna()
        if len(raw) < LOOKBACK + PRED_WIN + 10:
            return None
        df = pd.DataFrame({
            "timestamps": raw.index.tz_localize(None) if raw.index.tz else raw.index,
            "open":   raw["Open"].values.flatten(),
            "high":   raw["High"].values.flatten(),
            "low":    raw["Low"].values.flatten(),
            "close":  raw["Close"].values.flatten(),
            "volume": raw["Volume"].values.flatten(),
            "amount": raw["Open"].values.flatten() * raw["Volume"].values.flatten(),
        })
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        return df
    except Exception as e:
        print(f"  [{ticker}] fetch failed: {e}")
        return None


def calc_time_stamps(timestamps: pd.Series) -> np.ndarray:
    ts = pd.Series(pd.to_datetime(timestamps))
    return np.stack([
        ts.dt.minute.values,
        ts.dt.hour.values,
        ts.dt.weekday.values,
        ts.dt.day.values,
        ts.dt.month.values,
    ], axis=1).astype(np.float32)


def make_samples(df: pd.DataFrame, start: str | None, end: str | None) -> list:
    """Slide a window across the DataFrame and return (x, x_stamp, y, y_stamp) tuples."""
    feat_cols = ["open", "high", "low", "close", "volume", "amount"]
    mask = pd.Series([True] * len(df))
    if start:
        mask &= df["timestamps"] >= pd.Timestamp(start)
    if end:
        mask &= df["timestamps"] <= pd.Timestamp(end)
    sub = df[mask].reset_index(drop=True)
    if len(sub) < LOOKBACK + PRED_WIN:
        return []

    samples = []
    for i in range(len(sub) - LOOKBACK - PRED_WIN + 1):
        x_raw = sub[feat_cols].iloc[i : i + LOOKBACK].values.astype(np.float32)
        y_raw = sub[feat_cols].iloc[i + LOOKBACK : i + LOOKBACK + PRED_WIN].values.astype(np.float32)

        # Normalise using recent 60-bar window (change #4)
        norm_win = min(60, len(x_raw))
        mu  = x_raw[-norm_win:].mean(axis=0)
        sig = x_raw[-norm_win:].std(axis=0) + 1e-5
        x_norm = np.clip((x_raw - mu) / sig, -CLIP, CLIP)
        y_norm = np.clip((y_raw - mu) / sig, -CLIP, CLIP)

        x_stamp = calc_time_stamps(sub["timestamps"].iloc[i : i + LOOKBACK])
        y_stamp = calc_time_stamps(sub["timestamps"].iloc[i + LOOKBACK : i + LOOKBACK + PRED_WIN])

        samples.append((x_norm, x_stamp, y_norm, y_stamp))
    return samples


def main(tickers: list[str]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    train_samples, val_samples, test_samples = [], [], []

    print(f"Fetching {len(tickers)} tickers …")
    for ticker in tqdm(tickers):
        df = fetch_daily(ticker)
        if df is None:
            continue
        train_samples += make_samples(df, "2018-01-01", TRAIN_END)
        val_samples   += make_samples(df, VAL_START,    VAL_END)
        test_samples  += make_samples(df, TEST_START,   None)

    print(f"\nSamples — train: {len(train_samples)}  val: {len(val_samples)}  test: {len(test_samples)}")

    for name, data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        path = os.path.join(OUT_DIR, f"{name}_dataset.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {path}")

    print("\nDone. Now run:")
    print("  python train_predictor.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    args = parser.parse_args()
    main(args.tickers)
