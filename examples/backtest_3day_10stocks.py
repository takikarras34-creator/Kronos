"""
3-day enhanced Kronos backtest — 10 US stocks.

Changes v2:
  1. Ensemble across 3 starting points (cutoff, cutoff-1, cutoff-2)
  2. 15 samples per ensemble member (45 effective estimates)
  3. Realized volatility bands (replaces too-narrow model CI)
  4. Temperature auto-tuned to per-stock 20-day ATR
  5. Lookback reduced 400 → 120 bars (focus on recent context)
"""

import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from model.enhancements import (
    compute_beta,
    apply_regime_correction,
    apply_vix_band_adjustment,
    check_earnings_in_window,
    check_fed_meetings_in_window,
)

TICKERS      = ["PLTR", "SNAP", "SOFI", "SHOP", "NVDA"]
CUTOFF_AGO   = 5      # trading days back from today
PRED_DAYS    = 3
LOOKBACK     = 400    # reverted: shorter lookback hurt trending stocks
SAMPLE_COUNT = 15     # up from 10: better median estimate
N_ENSEMBLE   = 1      # reverted: ensemble didn't help direction accuracy
FEAT_COLS    = ["open", "high", "low", "close", "volume", "amount"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_daily(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="3y", interval="1d",
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
        data = yf.download("SPY", start="2025-01-01",
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
        data = yf.download("^VIX", start="2025-01-01",
                           end=(cutoff + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                           interval="1d", auto_adjust=True, progress=False)
        close = data["Close"].squeeze().dropna()
        return float(close.iloc[-1]) if len(close) > 0 else 20.0
    except Exception:
        return 20.0


def vol_adjusted_temperature(df_in: pd.DataFrame, window: int = 20) -> float:
    """Change 4: set T based on recent 20-day ATR as % of price."""
    closes = df_in["close"].values[-window - 1:]
    if len(closes) < 2:
        return 0.7
    daily_returns = np.abs(np.diff(closes) / closes[:-1])
    atr_pct = float(np.mean(daily_returns)) * 100   # e.g. 1.5 = 1.5%/day
    # T ramps from 0.6 (calm) to 0.9 (very volatile)
    T = float(np.clip(0.6 + atr_pct * 0.06, 0.60, 0.90))
    return T


def realized_vol_bands(pred_closes: np.ndarray, df_in: pd.DataFrame,
                        window: int = 20, z: float = 1.28) -> tuple:
    """
    Change 3: bands based on realized daily vol, not model percentiles.
    z=1.28 → ~80% CI (10th-90th percentile) for normally distributed returns.
    Scales as sqrt(t) over the forecast horizon.
    """
    closes = df_in["close"].values[-window - 1:]
    daily_vol = float(np.std(np.diff(closes) / closes[:-1]))  # daily return std

    lo, hi = [], []
    for t, p in enumerate(pred_closes, start=1):
        half_band = p * z * daily_vol * np.sqrt(t)
        lo.append(p - half_band)
        hi.append(p + half_band)
    return np.array(lo), np.array(hi)


def ensemble_predict(predictor, df_in: pd.DataFrame,
                     T: float, sample_count: int,
                     pred_days: int, n_ensemble: int) -> np.ndarray:
    """
    Change 1: run n_ensemble predictions from slightly different starting
    points and average the close forecasts.

    Starting point i uses df_in[:-i] (or full df if i==0) and predicts
    pred_days+i days so all predictions land on the same target dates.
    Returns averaged close array of length pred_days.
    """
    all_closes = []

    for offset in range(n_ensemble):
        # Slice history: drop `offset` most-recent bars
        df_slice = df_in.iloc[:len(df_in) - offset] if offset > 0 else df_in
        df_slice = df_slice.tail(LOOKBACK).reset_index(drop=True)

        last_ts  = df_slice["timestamps"].iloc[-1]
        n_pred   = pred_days + offset
        y_ts     = pd.Series(pd.bdate_range(start=last_ts, periods=n_pred + 1)[1:])

        chunk = predictor.predict_with_confidence(
            df=df_slice[FEAT_COLS],
            x_timestamp=df_slice["timestamps"],
            y_timestamp=y_ts,
            pred_len=n_pred, T=T, top_p=0.9,
            sample_count=sample_count, verbose=False,
        )

        # Take the last pred_days rows — these correspond to the target dates
        all_closes.append(chunk["close"].values[-pred_days:])

    return np.mean(all_closes, axis=0)


# ── Per-ticker runner ─────────────────────────────────────────────────────────

def run_ticker(predictor, ticker: str, cutoff_date: pd.Timestamp,
               spy_momentum: float, vix: float) -> dict:

    df_full = load_daily(ticker)
    df_in   = df_full[df_full["timestamps"] <= cutoff_date].reset_index(drop=True)
    df_act  = df_full[df_full["timestamps"] > cutoff_date].head(PRED_DAYS).reset_index(drop=True)

    if len(df_in) < LOOKBACK or len(df_act) < PRED_DAYS:
        return {"ticker": ticker, "error": f"insufficient data ({len(df_act)} actual bars)"}

    cutoff_close = df_in["close"].iloc[-1]

    # Change 4: volatility-adjusted temperature
    T = vol_adjusted_temperature(df_in)

    # Check events in window
    pred_end = cutoff_date + pd.Timedelta(days=10)
    has_earn, earn_dates = check_earnings_in_window(ticker, cutoff_date, pred_end)
    has_fed,  fed_dates  = check_fed_meetings_in_window(cutoff_date, pred_end)

    # Change 1+2: ensemble prediction (avg close across 3 starting points)
    beta          = compute_beta(ticker)
    pred_closes   = ensemble_predict(predictor, df_in, T, SAMPLE_COUNT, PRED_DAYS, N_ENSEMBLE)

    # Build a minimal pred_df for regime correction (only close needed)
    target_dates = pd.bdate_range(start=cutoff_date, periods=PRED_DAYS + 1)[1:]
    pred_df      = pd.DataFrame({"close": pred_closes}, index=target_dates)
    for col in ["open", "high", "low"]:
        pred_df[col] = pred_closes   # approximate; only close is evaluated

    # Regime correction (small scale — momentum nudge only)
    pred_df = apply_regime_correction(pred_df, cutoff_close, spy_momentum, beta, scale=0.1)
    pred_closes = pred_df["close"].values

    # Change 3: realized volatility bands (replace model CI)
    lo, hi = realized_vol_bands(pred_closes, df_in)

    actual_closes = df_act["close"].values[:PRED_DAYS]

    mape     = float(np.mean(np.abs((actual_closes - pred_closes) / actual_closes)) * 100)
    mae      = float(np.mean(np.abs(actual_closes - pred_closes)))
    act_dirs = np.sign(np.diff(np.concatenate([[cutoff_close], actual_closes])))
    pred_dirs = np.sign(np.diff(np.concatenate([[cutoff_close], pred_closes])))
    dir_acc  = float(np.mean(act_dirs == pred_dirs) * 100)
    final_err = (pred_closes[-1] / actual_closes[-1] - 1) * 100
    coverage  = float(np.mean((actual_closes >= lo) & (actual_closes <= hi)) * 100)

    return {
        "ticker":        ticker,
        "cutoff_close":  cutoff_close,
        "actual_closes": actual_closes,
        "pred_closes":   pred_closes,
        "dates":         df_act["timestamps"].values,
        "lo":            lo,
        "hi":            hi,
        "mape":          mape,
        "mae":           mae,
        "dir_acc":       dir_acc,
        "final_err":     final_err,
        "coverage":      coverage,
        "T":             T,
        "has_earnings":  has_earn,
        "has_fed":       has_fed,
        "error":         None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching shared market signals …")
    ref = load_daily("SPY")
    cutoff_idx  = len(ref) - CUTOFF_AGO - 1
    cutoff_date = ref["timestamps"].iloc[cutoff_idx]

    spy_momentum = get_spy_momentum_at(cutoff_date, lookback_days=20)
    vix          = get_vix_at(cutoff_date)
    regime_label = "BULLISH" if spy_momentum > 0.02 else \
                   "BEARISH" if spy_momentum < -0.02 else "NEUTRAL"

    print(f"Cutoff: {cutoff_date.date()}   "
          f"SPY 20d: {spy_momentum:+.1%} ({regime_label})   VIX: {vix:.1f}")
    print(f"Config : lookback={LOOKBACK}  samples={SAMPLE_COUNT}×{N_ENSEMBLE}ensemble  "
          f"vol-adjusted T  realized-vol bands\n")

    print("Loading Kronos-mini …")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, max_context=2048)

    results = []
    for ticker in TICKERS:
        print(f"  {ticker:<8}", end="", flush=True)
        r = run_ticker(predictor, ticker, cutoff_date, spy_momentum, vix)
        if r["error"]:
            print(f"  SKIP — {r['error']}")
        else:
            print(f"  T={r['T']:.2f}  MAPE={r['mape']:.1f}%  "
                  f"Dir={r['dir_acc']:.0f}%  FinalErr={r['final_err']:+.1f}%")
        results.append(r)

    good = [r for r in results if not r["error"]]

    # ── Summary ───────────────────────────────────────────────────
    mapes    = [r["mape"]     for r in good]
    dir_accs = [r["dir_acc"]  for r in good]
    covs     = [r["coverage"] for r in good]

    print(f"\n{'='*80}")
    print(f"  3-DAY BACKTEST — cutoff {cutoff_date.date()}")
    print(f"  Market: {regime_label} (SPY {spy_momentum:+.1%} over 20 days)   VIX: {vix:.1f}")
    print(f"{'='*80}")

    for r in good:
        a    = r["actual_closes"]
        p    = r["pred_closes"]
        lo   = r["lo"]
        hi   = r["hi"]
        note = ("  ⚠ EARNINGS in window" if r["has_earnings"] else "") + \
               ("  ⚠ FED MEETING in window" if r["has_fed"] else "")

        print(f"\n  {r['ticker']}  (starting price: ${r['cutoff_close']:.2f}){note}")
        print(f"  {'':4} {'Date':<10} {'Actual':>10} {'Predicted':>10} {'Band Low':>10} {'Band High':>10}  {'Correct?':>9}")
        print(f"  {'-'*68}")
        correct = 0
        prev_actual = r["cutoff_close"]
        prev_pred   = r["cutoff_close"]
        for i in range(len(a)):
            date_str  = pd.Timestamp(r["dates"][i]).strftime("%b %d")
            act_dir   = "up" if a[i] > prev_actual else "down"
            pred_dir  = "up" if p[i] > prev_pred   else "down"
            direction = "yes" if act_dir == pred_dir else "no"
            if act_dir == pred_dir:
                correct += 1
            in_band = lo[i] <= a[i] <= hi[i]
            band_str = f"${lo[i]:>7.2f}" if in_band else f" ${lo[i]:>6.2f}"
            print(f"  D{i+1}  {date_str:<10} ${a[i]:>9.2f}  ${p[i]:>9.2f}  "
                  f"${lo[i]:>9.2f}  ${hi[i]:>9.2f}  {direction:>9}"
                  + (" ✓ in band" if in_band else ""))
            prev_actual = a[i]
            prev_pred   = p[i]

        final_err  = (p[-1] / a[-1] - 1) * 100
        total_move_actual = (a[-1] / r["cutoff_close"] - 1) * 100
        total_move_pred   = (p[-1] / r["cutoff_close"] - 1) * 100
        print(f"  {'-'*68}")
        print(f"  Summary: started ${r['cutoff_close']:.2f}  →  "
              f"actual ${a[-1]:.2f} ({total_move_actual:+.1f}%)  |  "
              f"predicted ${p[-1]:.2f} ({total_move_pred:+.1f}%)  |  "
              f"off by {abs(final_err):.1f}%")
        print(f"  MAPE={r['mape']:.1f}%   Direction={r['dir_acc']:.0f}%   "
              f"Band coverage={r['coverage']:.0f}%")

    print(f"\n{'='*80}")
    print(f"  AVERAGES  —  MAPE={np.mean(mapes):.1f}%   "
          f"Direction={np.mean(dir_accs):.0f}%   "
          f"Band coverage={np.mean(covs):.0f}%")
    print(f"{'='*80}")

    # ── Plain-English metric explanations ─────────────────────────
    print(f"""
  WHAT THESE NUMBERS MEAN
  ───────────────────────
  Direction ({np.mean(dir_accs):.0f}% avg)
    Did we correctly predict which way the stock moved each day?
    Up or down — that's it. 50% = coin flip. 67% = got 2 out of 3 days right.
    This is the most practically useful number if you're trading.

  MAPE — Mean Absolute Percentage Error ({np.mean(mapes):.1f}% avg)
    On average, how far off was our price prediction as a percentage?
    Example: MAPE of 3% on a $200 stock = off by $6 on average per day.
    Lower is better. Under 3% for a 3-day window is reasonable.

  Band Coverage ({np.mean(covs):.0f}% avg)
    We predict not just a single price but a range (low to high).
    This tells you what % of actual prices landed inside that range.
    A perfect 10th-90th percentile band should capture 80% of outcomes.
    Higher = our uncertainty range is realistic and useful.
""")

    # ── Chart ─────────────────────────────────────────────────────
    n    = len(good)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()

    fig.suptitle(
        f"Kronos v2 — 3-Day Backtest  |  cutoff {cutoff_date.date()}\n"
        f"Regime: {regime_label} ({spy_momentum:+.1%})  VIX={vix:.0f}  "
        f"Ensemble×{N_ENSEMBLE}  Samples={SAMPLE_COUNT}  Lookback={LOOKBACK}\n"
        f"Avg MAPE={np.mean(mapes):.1f}%  Avg Dir={np.mean(dir_accs):.0f}%  "
        f"Avg Band Coverage={np.mean(covs):.0f}%",
        fontsize=11, fontweight="bold"
    )

    for i, r in enumerate(good):
        ax  = axes[i]
        act = np.concatenate([[r["cutoff_close"]], r["actual_closes"]])
        prd = np.concatenate([[r["cutoff_close"]], r["pred_closes"]])
        lo  = np.concatenate([[r["cutoff_close"]], r["lo"]])
        hi  = np.concatenate([[r["cutoff_close"]], r["hi"]])
        xs  = range(len(act))

        ax.plot(xs, act, color="limegreen",  linewidth=2,   marker="o", markersize=5, label="Actual")
        ax.plot(xs, prd, color="tomato",     linewidth=2,   linestyle="--", marker="s", markersize=4, label="Predicted")
        ax.fill_between(xs, lo, hi, color="tomato", alpha=0.2, label="Realized vol band")

        note = ("⚠E" if r["has_earnings"] else "") + ("⚠F" if r["has_fed"] else "")
        ax.set_title(
            f"{r['ticker']}  T={r['T']:.2f}  MAPE={r['mape']:.1f}%  "
            f"Dir={r['dir_acc']:.0f}%  Cov={r['coverage']:.0f}%  {note}",
            fontsize=8.5, fontweight="bold"
        )
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Cut", "D1", "D2", "D3"], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for j in range(len(good), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = "backtest_3day_10stocks.png"
    plt.savefig(out, dpi=150)
    print(f"\nChart saved → examples/{out}")


if __name__ == "__main__":
    main()
