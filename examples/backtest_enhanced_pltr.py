"""
Enhanced Kronos backtest for PLTR using all 9 signal corrections:
  SPY regime, analyst consensus, EPS surprise, news sentiment,
  VIX bands, earnings flag, Fed meeting flag, rolling predictions,
  confidence intervals.

Sets cutoff CUTOFF_AGO trading days back, predicts PRED_DAYS forward,
then compares to actual prices that have since occurred.
"""

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
    apply_regime_correction,
    predict_rolling,
    apply_vix_band_adjustment,
    get_analyst_consensus,
    apply_analyst_correction,
    get_earnings_surprise,
    apply_earnings_surprise_correction,
    apply_sentiment_correction,
    check_fed_meetings_in_window,
)

TICKER       = "PLTR"
CUTOFF_AGO   = 24    # trading days back from today
PRED_DAYS    = 20    # days to predict
ROLL_STEP    = 5     # rolling window size (must divide PRED_DAYS)
LOOKBACK     = 400   # bars fed to model
SAMPLE_COUNT = 15
T            = 0.7


def load_daily(ticker: str) -> pd.DataFrame:
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
    return df.reset_index(drop=True)


def get_spy_momentum_at(cutoff: pd.Timestamp, lookback_days: int = 20) -> float:
    try:
        data = yf.download("SPY", start="2024-01-01",
                           end=(cutoff + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                           interval="1d", auto_adjust=True, progress=False)
        close = data["Close"].squeeze().dropna()
        if len(close) < lookback_days + 1:
            return 0.0
        return float(close.iloc[-1] / close.iloc[-lookback_days - 1] - 1)
    except Exception:
        return 0.0


def get_vix_at(cutoff: pd.Timestamp) -> float:
    try:
        data = yf.download("^VIX", start="2024-01-01",
                           end=(cutoff + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                           interval="1d", auto_adjust=True, progress=False)
        close = data["Close"].squeeze().dropna()
        return float(close.iloc[-1]) if len(close) > 0 else 20.0
    except Exception:
        return 20.0


def main():
    print(f"\n{'='*65}")
    print(f"  Enhanced Kronos Backtest — {TICKER}")
    print(f"  Cutoff: {CUTOFF_AGO} trading days ago  |  Predict: {PRED_DAYS} days")
    print(f"{'='*65}")

    # ── Data ──────────────────────────────────────────────────────
    print(f"\nFetching {TICKER} data …")
    df_full = load_daily(TICKER)

    cutoff_idx   = len(df_full) - CUTOFF_AGO - 1
    df_in        = df_full.iloc[:cutoff_idx + 1].reset_index(drop=True)
    df_act       = df_full.iloc[cutoff_idx + 1 : cutoff_idx + 1 + PRED_DAYS].reset_index(drop=True)
    cutoff_date  = df_in["timestamps"].iloc[-1]
    cutoff_close = df_in["close"].iloc[-1]
    actual_days  = len(df_act)

    print(f"Cutoff date  : {cutoff_date.date()}   close=${cutoff_close:.2f}")
    print(f"Actual bars  : {actual_days} trading days available")

    # ── Historical signals ─────────────────────────────────────────
    print("\nFetching signals as of cutoff date …")
    spy_momentum = get_spy_momentum_at(cutoff_date, lookback_days=20)
    vix          = get_vix_at(cutoff_date)
    beta         = compute_beta(TICKER)
    analyst      = get_analyst_consensus(TICKER)
    surprise     = get_earnings_surprise(TICKER)
    sentiment    = 0.0   # can't reconstruct historical news; use neutral

    regime_label = "BULLISH" if spy_momentum > 0.02 else \
                   "BEARISH" if spy_momentum < -0.02 else "NEUTRAL"
    vix_label    = "HIGH FEAR" if vix > 30 else "ELEVATED" if vix > 20 else "NORMAL"

    pred_end = cutoff_date + pd.Timedelta(days=PRED_DAYS + 10)
    has_earnings, earn_dates = check_earnings_in_window(TICKER, cutoff_date, pred_end)
    has_fed,      fed_dates  = check_fed_meetings_in_window(cutoff_date, pred_end)

    print(f"\n{'─'*65}")
    print(f"  SIGNAL SNAPSHOT — {TICKER} as of {cutoff_date.date()}")
    print(f"{'─'*65}")
    print(f"  SPY 20d momentum : {spy_momentum:+.1%}  ({regime_label})")
    print(f"  VIX              : {vix:.1f}  ({vix_label})")
    print(f"  {TICKER} beta    : {beta:.2f}")
    if analyst.get("target_mean"):
        upside = (analyst["target_mean"] / cutoff_close - 1) * 100
        print(f"  Analyst target   : ${analyst['target_mean']:.2f}  ({upside:+.1f}% upside)  "
              f"[{analyst['num_analysts']} analysts]")
    else:
        print(f"  Analyst target   : N/A")
    print(f"  EPS surprise     : {surprise:+.1%}" if surprise else
          f"  EPS surprise     : N/A")
    if has_earnings:
        print(f"  ⚠  Earnings in window: {', '.join(str(d.date()) for d in earn_dates)}")
    if has_fed:
        print(f"  ⚠  Fed meeting in window: {', '.join(str(d.date()) for d in fed_dates)}")
    print(f"{'─'*65}")

    # ── Load model ─────────────────────────────────────────────────
    print("\nLoading Kronos-mini …")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, max_context=2048)

    # ── Rolling prediction ─────────────────────────────────────────
    n_steps = PRED_DAYS // ROLL_STEP
    print(f"\nRunning {n_steps}×{ROLL_STEP}-day rolling forecast (sample_count={SAMPLE_COUNT}) …")
    df_input = df_in.tail(LOOKBACK).reset_index(drop=True)
    pred_df  = predict_rolling(
        predictor, df_input,
        step=ROLL_STEP, total_days=PRED_DAYS, lookback=LOOKBACK,
        T=T, sample_count=SAMPLE_COUNT, use_confidence=True, verbose=False,
    )

    # ── Apply all corrections ──────────────────────────────────────
    print("Applying signal corrections …")
    pred_df = apply_regime_correction(pred_df, cutoff_close, spy_momentum, beta, scale=0.5)
    pred_df = apply_analyst_correction(pred_df, cutoff_close, analyst, scale=0.3)
    pred_df = apply_earnings_surprise_correction(pred_df, cutoff_close, surprise, scale=0.2)
    pred_df = apply_sentiment_correction(pred_df, cutoff_close, sentiment, scale=0.15)
    pred_df = apply_vix_band_adjustment(pred_df, vix)

    # ── Accuracy metrics ───────────────────────────────────────────
    actual_closes = df_act["close"].values[:actual_days]
    pred_closes   = pred_df["close"].values[:actual_days]

    mape = float(np.mean(np.abs((actual_closes - pred_closes) / actual_closes)) * 100)
    mae  = float(np.mean(np.abs(actual_closes - pred_closes)))

    actual_dirs = np.sign(np.diff(np.concatenate([[cutoff_close], actual_closes])))
    pred_dirs   = np.sign(np.diff(np.concatenate([[cutoff_close], pred_closes])))
    dir_acc     = float(np.mean(actual_dirs == pred_dirs) * 100)

    final_actual = actual_closes[-1]
    final_pred   = pred_closes[-1]
    final_err    = (final_pred / final_actual - 1) * 100

    has_bands = "close_lo" in pred_df.columns
    if has_bands:
        lo       = pred_df["close_lo"].values[:actual_days]
        hi       = pred_df["close_hi"].values[:actual_days]
        coverage = float(np.mean((actual_closes >= lo) & (actual_closes <= hi)) * 100)
    else:
        coverage = None

    print(f"\n{'─'*65}")
    print(f"  ACCURACY RESULTS — {TICKER}")
    print(f"{'─'*65}")
    print(f"  MAPE              : {mape:.2f}%")
    print(f"  MAE (avg)         : ${mae:.2f} per day")
    print(f"  Direction accuracy: {dir_acc:.1f}%  ({int(dir_acc/100*actual_days)}/{actual_days} days correct)")
    print(f"  Final day error   : {final_err:+.1f}%  "
          f"(predicted ${final_pred:.2f}  actual ${final_actual:.2f})")
    if coverage is not None:
        print(f"  Band coverage     : {coverage:.1f}%  (actuals inside 10-90% band)")
    print(f"{'─'*65}")

    print(f"\n  Day-by-day breakdown:")
    print(f"  {'Day':<4} {'Date':<12} {'Actual':>8} {'Predicted':>10} {'Error':>8}"
          + ("  In Band" if has_bands else ""))
    print(f"  {'-'*55}")
    for i in range(actual_days):
        date_str = df_act["timestamps"].iloc[i].strftime("%b %d") if i < len(df_act) else "---"
        a, p = actual_closes[i], pred_closes[i]
        err  = (p / a - 1) * 100
        band_str = ""
        if has_bands:
            band_str = "  yes" if lo[i] <= a <= hi[i] else "  no"
        print(f"  {i+1:<4} {date_str:<12} ${a:>7.2f}  ${p:>8.2f}  {err:>+6.1f}%{band_str}")

    # ── Chart ──────────────────────────────────────────────────────
    hist_show = df_in.tail(60)
    fig, ax   = plt.subplots(figsize=(14, 6))

    ax.plot(hist_show["timestamps"], hist_show["close"],
            color="steelblue", linewidth=1.5, label="History")
    ax.plot(df_act["timestamps"].iloc[:actual_days], actual_closes,
            color="limegreen", linewidth=2.5, label="Actual", marker="o", markersize=5)
    ax.plot(pred_df.index[:actual_days], pred_closes[:actual_days],
            color="tomato", linewidth=2, linestyle="--",
            label=f"Predicted (enhanced)  MAPE={mape:.1f}%  Dir={dir_acc:.0f}%")
    if has_bands:
        ax.fill_between(pred_df.index[:actual_days], lo, hi,
                        color="tomato", alpha=0.18,
                        label=f"10–90% band  (coverage={coverage:.0f}%)")
    ax.axvline(cutoff_date, color="gray", linestyle=":", linewidth=1.5, label="Cutoff")

    events = []
    if has_earnings:
        for d in earn_dates:
            ax.axvline(d, color="orange", linestyle="--", linewidth=1, alpha=0.8)
            events.append(f"Earnings {d.strftime('%b %d')}")
    if has_fed:
        for d in fed_dates:
            ax.axvline(d, color="purple", linestyle="--", linewidth=1, alpha=0.6)
            events.append(f"Fed {d.strftime('%b %d')}")

    event_str = "  ⚠ " + " | ".join(events) if events else "  ✓ No major events"
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_title(
        f"Kronos Enhanced Backtest — {TICKER}  |  cutoff {cutoff_date.date()}\n"
        f"Regime: {regime_label} (SPY {spy_momentum:+.1%})  VIX={vix:.0f}  "
        f"β={beta:.2f}  EPS surprise={surprise:+.1%}{event_str}",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.tight_layout()

    out = f"backtest_enhanced_{TICKER}.png"
    plt.savefig(out, dpi=150)
    print(f"\nChart saved → examples/{out}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
