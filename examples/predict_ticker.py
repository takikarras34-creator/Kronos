import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

warnings.filterwarnings('ignore')
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_ohlcv(ticker: str, lookback: int = 400, interval: str = "1h") -> pd.DataFrame:
    data = yf.download(ticker, period="60d", interval=interval, auto_adjust=True, progress=False)
    data = data.dropna()
    df = pd.DataFrame({
        "timestamps": data.index.tz_localize(None) if data.index.tz else data.index,
        "open":   data["Open"].values.flatten(),
        "high":   data["High"].values.flatten(),
        "low":    data["Low"].values.flatten(),
        "close":  data["Close"].values.flatten(),
        "volume": data["Volume"].values.flatten(),
        "amount": (data["Open"].values.flatten() * data["Volume"].values.flatten()),
    })
    df = df.tail(lookback).reset_index(drop=True)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df


def plot_prediction(ticker, kline_df, pred_df):
    pred_df = pred_df.copy()
    pred_df.index = pd.RangeIndex(len(kline_df) - len(pred_df), len(kline_df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Kronos Forecast — {ticker.upper()}", fontsize=15, fontweight="bold")

    hist_end = len(kline_df) - len(pred_df)
    ax1.plot(range(hist_end), kline_df["close"].iloc[:hist_end], color="steelblue", linewidth=1.5, label="History")
    ax1.plot(pred_df.index, pred_df["close"], color="tomato", linewidth=1.5, linestyle="--", label="Forecast")
    ax1.axvline(hist_end - 1, color="gray", linestyle=":", linewidth=1)
    ax1.set_ylabel("Close Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(hist_end), kline_df["volume"].iloc[:hist_end], color="steelblue", alpha=0.6, label="History")
    ax2.bar(pred_df.index, pred_df["volume"], color="tomato", alpha=0.6, label="Forecast")
    ax2.axvline(hist_end - 1, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("Volume")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"forecast_{ticker.upper()}.png", dpi=150)
    print(f"\nPlot saved to examples/forecast_{ticker.upper()}.png")
    plt.show()


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SOFI"
    lookback = 400
    pred_len = 24  # 24 hours ahead

    print(f"Fetching 1h data for {ticker}...")
    df = fetch_ohlcv(ticker, lookback=lookback)
    print(f"Got {len(df)} bars. Last close: {df['close'].iloc[-1]:.4f}")

    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df["timestamps"]

    last_ts = df["timestamps"].iloc[-1]
    y_ts = pd.date_range(start=last_ts, periods=pred_len + 1, freq="1h")[1:]
    y_ts = pd.Series(y_ts)

    print(f"\nRunning forecast for next {pred_len} hours...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_ts,
        y_timestamp=y_ts,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    print("\nForecast (first 5 bars):")
    print(pred_df[["open", "high", "low", "close", "volume"]].head().to_string())
    print(f"\nLast known close : {df['close'].iloc[-1]:.4f}")
    print(f"Forecast close   : {pred_df['close'].iloc[-1]:.4f}  ({pred_len}h out)")

    plot_prediction(ticker, df, pred_df)
