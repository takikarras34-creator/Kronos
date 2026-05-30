"""
Iron condor simulation using Kronos 3-day band predictions.

Runs 5 stocks across 3 historical cutoff dates = 15 simulated trades.
Prices condors with Black-Scholes. Tracks P&L on a $37,000 account.

Fast mode: sample_count=5, no ensemble, lookback=120 (~5 min runtime).
"""

import sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor
from model.enhancements import apply_regime_correction, apply_vix_band_adjustment

TICKERS      = ["PLTR", "SNAP", "SOFI", "SHOP", "NVDA"]
CUTOFF_OFFSETS = [5, 10, 15]   # trading days back (calm mode)
STRESS_VIX   = 25              # VIX threshold for stress test mode
ACCOUNT      = 37_000
RISK_PCT     = 0.02            # 2% max risk per trade
SPREAD_WIDTH = None            # auto: 5% of stock price, min $1
SAMPLE_COUNT = 5               # fast mode
LOOKBACK     = 120
FEAT_COLS    = ["open","high","low","close","volume","amount"]
RISK_FREE    = 0.05            # ~current T-bill rate


# ── Black-Scholes pricing ─────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Standard Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return float(intrinsic)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def estimate_iv(df_in, window=20):
    """Estimate implied vol from historical realized vol + IV premium (~15%)."""
    closes = df_in["close"].values[-window - 1:]
    if len(closes) < 2:
        return 0.30
    daily_vol = np.std(np.diff(np.log(closes)))
    annual_vol = daily_vol * np.sqrt(252)
    return float(annual_vol * 1.15)   # IV tends to run ~15% above realized


def price_iron_condor(S, lo, hi, T, r, iv, spread_width):
    """
    Price a standard iron condor:
      Short call at hi, long call at hi+width
      Short put  at lo, long put  at lo-width

    Returns (net_credit, max_loss) per share.
    """
    short_call = bs_price(S, hi,              T, r, iv, "call")
    long_call  = bs_price(S, hi + spread_width, T, r, iv, "call")
    short_put  = bs_price(S, lo,              T, r, iv, "put")
    long_put   = bs_price(S, lo - spread_width, T, r, iv, "put")

    net_credit = (short_call - long_call) + (short_put - long_put)
    max_loss   = spread_width - net_credit
    return round(net_credit, 3), round(max_loss, 3)


def condor_pnl_at_expiry(S_final, lo, hi, spread_width, net_credit):
    """
    P&L per share at expiration:
      Full credit if S_final between lo and hi.
      Partial loss between short/long strikes.
      Max loss if fully outside.
    """
    if lo <= S_final <= hi:
        return net_credit                           # max profit
    elif S_final > hi:
        loss = min(S_final - hi, spread_width)
        return net_credit - loss
    else:
        loss = min(lo - S_final, spread_width)
        return net_credit - loss


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_daily(ticker):
    raw = yf.download(ticker, period="2y", interval="1d",
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


def get_spy_momentum(cutoff, lookback=20):
    try:
        data = yf.download("SPY", start="2025-01-01",
                           end=(cutoff + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                           interval="1d", auto_adjust=True, progress=False)
        close = data["Close"].squeeze().dropna()
        return float(close.iloc[-1] / close.iloc[-lookback-1] - 1) if len(close) > lookback else 0.0
    except:
        return 0.0


def get_vix(cutoff):
    try:
        data = yf.download("^VIX", start="2025-01-01",
                           end=(cutoff + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                           interval="1d", auto_adjust=True, progress=False)
        close = data["Close"].squeeze().dropna()
        return float(close.iloc[-1]) if len(close) > 0 else 20.0
    except:
        return 20.0


# ── Main ──────────────────────────────────────────────────────────────────────

def find_high_vix_cutoffs(n=3, vix_threshold=25):
    """
    Find n historical dates where VIX was above vix_threshold,
    spaced at least 5 trading days apart, each with 3 actual days after.
    Returns list of pd.Timestamps.
    """
    print(f"Scanning VIX history for periods above {vix_threshold} …")
    vix_data = yf.download("^VIX", start="2024-01-01",
                            end=pd.Timestamp.today().strftime("%Y-%m-%d"),
                            interval="1d", auto_adjust=True, progress=False)
    vix_close = vix_data["Close"].squeeze().dropna()

    # Need at least 3 trading days of actual data after cutoff
    cutoff_latest = vix_close.index[-4]
    vix_close     = vix_close[vix_close.index <= cutoff_latest]
    high_vix      = vix_close[vix_close > vix_threshold].sort_index(ascending=False)

    if len(high_vix) == 0:
        print(f"  No VIX>{vix_threshold} periods found — falling back to recent dates")
        return None

    selected, last_picked = [], pd.Timestamp("1900-01-01")
    for date, val in high_vix.items():
        ts = pd.Timestamp(date).tz_localize(None) if date.tzinfo else pd.Timestamp(date)
        if abs((ts - last_picked).days) >= 7:   # space periods apart
            selected.append((ts, val))
            last_picked = ts
        if len(selected) == n:
            break

    for ts, v in selected:
        print(f"  {ts.date()}  VIX={v:.1f}")
    return [ts for ts, _ in selected]


def main(stress=False):
    print("Loading price data …")
    stock_data = {t: load_daily(t) for t in TICKERS}
    ref_df     = stock_data[TICKERS[0]]

    # Determine cutoff dates
    if stress:
        print("\n=== STRESS TEST MODE (high-VIX periods) ===")
        cutoff_dates = find_high_vix_cutoffs(n=3, vix_threshold=STRESS_VIX)
        if cutoff_dates is None:
            stress = False

    if not stress:
        print("\n=== NORMAL MODE (recent dates) ===")
        cutoff_dates = []
        for offset in CUTOFF_OFFSETS:
            idx = len(ref_df) - offset - 1
            cutoff_dates.append(ref_df["timestamps"].iloc[idx])

    print("Loading Kronos-mini …")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, max_context=2048)

    trades     = []
    account    = ACCOUNT
    trade_num  = 0

    print(f"\nRunning {len(TICKERS)} stocks × {len(cutoff_dates)} periods "
          f"= {len(TICKERS)*len(cutoff_dates)} trades …\n")

    for cutoff_ts in cutoff_dates:
        spy_mom     = get_spy_momentum(cutoff_ts)
        vix         = get_vix(cutoff_ts)
        regime      = "BULL" if spy_mom > 0.02 else "BEAR" if spy_mom < -0.02 else "NEUT"

        print(f"Period: cutoff {cutoff_ts.date()}  "
              f"SPY {spy_mom:+.1%} ({regime})  VIX {vix:.0f}")
        print(f"  {'Ticker':<6} {'Start$':>8} {'Band':>22} {'Credit':>7} "
              f"{'Risk':>7} {'Exp$':>8} {'Result':>8} {'P&L':>8}")
        print(f"  {'-'*75}")

        for ticker in TICKERS:
            df  = stock_data[ticker]
            df_in  = df[df["timestamps"] <= cutoff_ts].tail(LOOKBACK).reset_index(drop=True)
            df_act = df[df["timestamps"]  > cutoff_ts].head(3).reset_index(drop=True)

            if len(df_in) < 60 or len(df_act) < 3:
                continue

            S     = float(df_in["close"].iloc[-1])
            S_exp = float(df_act["close"].iloc[-1])   # price at expiry (day 3)

            # ── Kronos prediction ─────────────────────────────────
            last_ts = df_in["timestamps"].iloc[-1]
            y_ts    = pd.Series(pd.bdate_range(start=last_ts, periods=4)[1:])
            pred    = predictor.predict_with_confidence(
                df=df_in[FEAT_COLS], x_timestamp=df_in["timestamps"],
                y_timestamp=y_ts, pred_len=3, T=0.7,
                top_p=0.9, sample_count=SAMPLE_COUNT, verbose=False,
            )

            # Small regime nudge only
            from model.enhancements import compute_beta
            beta = compute_beta(ticker)
            pred = apply_regime_correction(pred, S, spy_mom, beta, scale=0.1)

            # Realized vol bands (same as backtest)
            closes_hist = df_in["close"].values[-21:]
            daily_vol   = np.std(np.diff(closes_hist) / closes_hist[:-1])
            lo_d3 = pred["close"].iloc[-1] * (1 - 1.28 * daily_vol * np.sqrt(3))
            hi_d3 = pred["close"].iloc[-1] * (1 + 1.28 * daily_vol * np.sqrt(3))

            # ── Condor pricing ────────────────────────────────────
            width  = max(round(S * 0.05 / 0.5) * 0.5, 1.0)  # 5% of price, min $1
            T_exp  = 5 / 252                                  # ~5 trading days
            iv     = estimate_iv(df_in)
            credit, max_loss_per_share = price_iron_condor(S, lo_d3, hi_d3, T_exp,
                                                            RISK_FREE, iv, width)

            if credit <= 0 or max_loss_per_share <= 0:
                continue

            # ── Position sizing ───────────────────────────────────
            max_risk   = account * RISK_PCT
            n_contracts = max(1, int(max_risk / (max_loss_per_share * 100)))
            n_contracts = min(n_contracts, 5)   # cap at 5 contracts

            actual_risk   = n_contracts * max_loss_per_share * 100
            credit_total  = n_contracts * credit * 100

            # ── P&L at expiry ─────────────────────────────────────
            pnl_per_share = condor_pnl_at_expiry(S_exp, lo_d3, hi_d3, width, credit)
            pnl_total     = pnl_per_share * 100 * n_contracts

            result = "WIN" if pnl_total > 0 else "LOSS"
            account += pnl_total
            trade_num += 1

            trades.append({
                "trade":    trade_num,
                "date":     cutoff_ts.date(),
                "ticker":   ticker,
                "start":    S,
                "lo":       lo_d3,
                "hi":       hi_d3,
                "credit":   credit_total,
                "risk":     actual_risk,
                "S_exp":    S_exp,
                "pnl":      pnl_total,
                "result":   result,
                "account":  account,
            })

            print(f"  {ticker:<6} ${S:>7.2f}  "
                  f"[${lo_d3:>6.2f}–${hi_d3:>6.2f}]  "
                  f"${credit_total:>6.0f}  ${actual_risk:>6.0f}  "
                  f"${S_exp:>7.2f}  {result:>5}  ${pnl_total:>+7.0f}")

        print()

    # ── Final summary ─────────────────────────────────────────────
    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_pnl   = sum(t["pnl"] for t in trades)
    total_credit = sum(t["credit"] for t in trades)
    win_rate    = len(wins) / len(trades) * 100 if trades else 0

    print(f"{'='*65}")
    print(f"  IRON CONDOR SIMULATION — {len(trades)} trades")
    print(f"{'='*65}")
    print(f"  Starting account : ${ACCOUNT:,.0f}")
    print(f"  Ending account   : ${account:,.0f}")
    print(f"  Total P&L        : ${total_pnl:+,.0f}")
    print(f"  Return           : {total_pnl/ACCOUNT*100:+.2f}%")
    print(f"{'─'*65}")
    print(f"  Win rate         : {win_rate:.0f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg winning trade: ${np.mean([t['pnl'] for t in wins]):+.0f}" if wins else "")
    print(f"  Avg losing trade : ${np.mean([t['pnl'] for t in losses]):+.0f}" if losses else "")
    print(f"  Largest win      : ${max(t['pnl'] for t in trades):+.0f}")
    print(f"  Largest loss     : ${min(t['pnl'] for t in trades):+.0f}")
    print(f"  Total premium collected: ${total_credit:,.0f}")
    print(f"{'─'*65}")

    # Extrapolate: ~40 trades/year at this frequency
    weekly_pnl  = total_pnl / (len(CUTOFF_OFFSETS) / 4)  # rough weekly avg
    annual_est  = weekly_pnl * 52
    print(f"\n  If run consistently (40 trades/year):")
    print(f"  Estimated annual P&L : ${annual_est:+,.0f}")
    print(f"  Estimated annual ROI : {annual_est/ACCOUNT*100:+.1f}%")
    print(f"{'='*65}")
    print(f"\n  NOTE: Options priced using Black-Scholes + realized vol estimate.")
    print(f"  Real results will vary due to bid-ask spreads, early close, slippage.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stress", action="store_true",
                   help="Use high-VIX historical periods instead of recent dates")
    args = p.parse_args()
    main(stress=args.stress)
