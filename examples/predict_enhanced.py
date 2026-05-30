"""
Enhanced Kronos prediction pipeline.

Changes vs baseline:
  1. T=0.7  (lower temperature → more conservative)
  2. sample_count=20 + median aggregation  (already in kronos.py)
  3. Kronos-mini (2048 context)  (passed via max_context)
  4. Recent 60-bar normalisation window  (already in kronos.py)
  5. Confidence intervals (10th / 90th percentile)
  6. Earnings calendar masking
  7. SPY market-regime beta correction
  8. Rolling 5-day predictions

Usage:
    python predict_enhanced.py SOFI
    python predict_enhanced.py AAPL --pred_len 25 --no_rolling
"""

import argparse
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from model.enhancements import (
    check_earnings_in_window,
    compute_beta,
    get_spy_momentum,
    apply_regime_correction,
    predict_rolling,
    get_vix_level,
    apply_vix_band_adjustment,
    get_analyst_consensus,
    apply_analyst_correction,
    get_earnings_surprise,
    apply_earnings_surprise_correction,
    get_news_sentiment,
    apply_sentiment_correction,
    check_fed_meetings_in_window,
)

LOOKBACK = 400


def load_daily(ticker: str, lookback: int = LOOKBACK) -> pd.DataFrame:
    raw = yf.download(ticker, period="5y", interval="1d",
                      auto_adjust=True, progress=False)
    raw = raw.dropna()
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
    return df.tail(lookback + 60).reset_index(drop=True)


def run(ticker: str, pred_len: int = 25, rolling: bool = True,
        sample_count: int = 20, T: float = 0.7,
        finnhub_key: str = None) -> None:

    print(f"\n{'='*60}")
    print(f"  Enhanced Kronos — {ticker.upper()}  ({pred_len}-day forecast)")
    print(f"{'='*60}")

    # ── Load model ───────────────────────────────────────────────
    print("Loading Kronos-mini …")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, max_context=2048)

    # ── Fetch data ───────────────────────────────────────────────
    df_full = load_daily(ticker, lookback=LOOKBACK)
    df_input = df_full.tail(LOOKBACK).reset_index(drop=True)
    cutoff_close = df_input["close"].iloc[-1]
    cutoff_date  = df_input["timestamps"].iloc[-1]
    print(f"Cutoff: {cutoff_date.date()}  close=${cutoff_close:.2f}")

    pred_start = cutoff_date
    pred_end   = pred_start + pd.Timedelta(days=pred_len + 10)

    # ── Fetch all signals in parallel (informational) ─────────────
    print("Fetching market signals …")

    # Earnings window check
    has_earnings, earn_dates = check_earnings_in_window(ticker, pred_start, pred_end)

    # Fed meeting check
    has_fed, fed_dates = check_fed_meetings_in_window(pred_start, pred_end)

    # SPY regime + beta
    spy_momentum = get_spy_momentum(lookback_days=20)
    beta         = compute_beta(ticker)
    regime_label = "BULLISH" if spy_momentum > 0.02 else \
                   "BEARISH" if spy_momentum < -0.02 else "NEUTRAL"

    # VIX
    vix = get_vix_level()
    vix_label = "HIGH FEAR" if vix > 30 else "ELEVATED" if vix > 20 else "NORMAL"

    # Analyst consensus
    analyst  = get_analyst_consensus(ticker)
    target   = analyst.get("target_mean")
    rec_mean = analyst.get("recommendation_mean")
    rec_label = ("Strong Buy" if rec_mean and rec_mean < 1.5 else
                 "Buy"        if rec_mean and rec_mean < 2.5 else
                 "Hold"       if rec_mean and rec_mean < 3.5 else
                 "Sell"       if rec_mean else "N/A")

    # Earnings surprise
    surprise = get_earnings_surprise(ticker)

    # News sentiment
    sentiment = get_news_sentiment(ticker, finnhub_key=finnhub_key)
    sent_label = ("Positive" if sentiment >  0.1 else
                  "Negative" if sentiment < -0.1 else "Neutral")

    # ── Print signal dashboard ────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  SIGNAL DASHBOARD — {ticker.upper()}")
    print(f"{'─'*60}")
    print(f"  SPY 20d momentum : {spy_momentum:+.1%}  ({regime_label})")
    print(f"  {ticker} beta     : {beta:.2f}")
    print(f"  VIX              : {vix:.1f}  ({vix_label})")
    if target:
        upside = (target / cutoff_close - 1) * 100
        print(f"  Analyst target   : ${target:.2f}  ({upside:+.1f}% from current)  "
              f"[{analyst['num_analysts']} analysts, {rec_label}]")
    else:
        print(f"  Analyst target   : N/A")
    print(f"  Last EPS surprise: {surprise:+.1%}" if surprise != 0 else
          f"  Last EPS surprise: N/A")
    print(f"  News sentiment   : {sentiment:+.2f}  ({sent_label})")

    if has_earnings:
        print(f"\n  ⚠  EARNINGS in window: {', '.join(str(d.date()) for d in earn_dates)}")
    if has_fed:
        print(f"  ⚠  FED MEETING in window: {', '.join(str(d.date()) for d in fed_dates)}")
    print(f"{'─'*60}\n")

    # ── Predict ──────────────────────────────────────────────────
    if rolling:
        print(f"\nRunning rolling {pred_len // 5}×5-day forecast (improvement 8) …")
        pred_df = predict_rolling(
            predictor, df_input, step=5, total_days=pred_len,
            lookback=LOOKBACK, T=T, sample_count=sample_count,
            use_confidence=True, verbose=False,
        )
    else:
        y_ts = pd.Series(pd.bdate_range(start=cutoff_date, periods=pred_len + 1)[1:])
        print(f"\nRunning single {pred_len}-day forecast …")
        pred_df = predictor.predict_with_confidence(
            df=df_input[["open","high","low","close","volume","amount"]],
            x_timestamp=df_input["timestamps"],
            y_timestamp=y_ts,
            pred_len=pred_len, T=T, top_p=0.9,
            sample_count=sample_count, verbose=True,
        )

    # ── Apply all signal corrections in order ────────────────────
    pred_df = apply_regime_correction(pred_df, cutoff_close, spy_momentum, beta, scale=0.5)
    pred_df = apply_analyst_correction(pred_df, cutoff_close, analyst, scale=0.3)
    pred_df = apply_earnings_surprise_correction(pred_df, cutoff_close, surprise, scale=0.2)
    pred_df = apply_sentiment_correction(pred_df, cutoff_close, sentiment, scale=0.15)
    pred_df = apply_vix_band_adjustment(pred_df, vix)   # widens bands, doesn't shift center

    # ── Results ──────────────────────────────────────────────────
    final_close    = pred_df["close"].iloc[-1]
    final_lo       = pred_df["close_lo"].iloc[-1]
    final_hi       = pred_df["close_hi"].iloc[-1]
    pred_move      = (final_close / cutoff_close - 1) * 100
    width_pct      = (final_hi - final_lo) / final_close * 100

    print(f"\n{'─'*60}")
    print(f"  Last known close  : ${cutoff_close:.2f}")
    print(f"  Forecast ({pred_len}d)    : ${final_close:.2f}  ({pred_move:+.1f}%)")
    print(f"  Confidence band   : ${final_lo:.2f} – ${final_hi:.2f}  (±{width_pct/2:.1f}% width)")
    print(f"  Corrections applied: SPY regime | Analyst | EPS surprise | Sentiment | VIX bands")
    if has_earnings or has_fed:
        print(f"  ⚠  High-impact events in window — treat forecast with caution")
    print(f"{'─'*60}")

    # ── Chart ────────────────────────────────────────────────────
    hist_show = df_full.tail(80)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    warnings_str = "  |  ".join(filter(None, [
        f"⚠ Earnings {earn_dates[0].strftime('%b %d')}" if has_earnings else None,
        f"⚠ Fed {fed_dates[0].strftime('%b %d')}"       if has_fed      else None,
    ])) or "✓ No major events"
    title = (f"Kronos Enhanced — {ticker.upper()}  |  {pred_len}-day forecast\n"
             f"Regime: {regime_label} (SPY {spy_momentum:+.1%})  β={beta:.2f}  "
             f"VIX={vix:.0f}  Sent={sentiment:+.2f}  EPS surprise={surprise:+.1%}  "
             f"|  {warnings_str}")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax1.plot(hist_show["timestamps"], hist_show["close"],
             color="steelblue", linewidth=1.5, label="History")
    ax1.axvline(cutoff_date, color="gray", linestyle=":", linewidth=1.5)
    ax1.plot(pred_df.index, pred_df["close"],
             color="tomato", linewidth=2, linestyle="--", label="Forecast (median)")
    ax1.fill_between(pred_df.index, pred_df["close_lo"], pred_df["close_hi"],
                     color="tomato", alpha=0.18, label="10th–90th pct band")
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax2.bar(hist_show["timestamps"], hist_show["volume"],
            color="steelblue", alpha=0.5, width=1, label="History")
    ax2.bar(pred_df.index, pred_df["volume"],
            color="tomato", alpha=0.5, width=1, label="Forecast")
    ax2.fill_between(pred_df.index, pred_df["volume_lo"], pred_df["volume_hi"],
                     color="tomato", alpha=0.12)
    ax2.axvline(cutoff_date, color="gray", linestyle=":", linewidth=1.5)
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    plt.tight_layout()
    out = f"forecast_enhanced_{ticker.upper()}.png"
    plt.savefig(out, dpi=150)
    print(f"\nChart saved → examples/{out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default="SOFI")
    parser.add_argument("--pred_len",     type=int,  default=25)
    parser.add_argument("--no_rolling",   action="store_true")
    parser.add_argument("--sample_count", type=int,  default=20)
    parser.add_argument("--T",            type=float, default=0.7)
    parser.add_argument("--finnhub_key",  type=str,   default=None,
                        help="Optional Finnhub API key for better news sentiment")
    args = parser.parse_args()
    run(args.ticker, args.pred_len, rolling=not args.no_rolling,
        sample_count=args.sample_count, T=args.T, finnhub_key=args.finnhub_key)
