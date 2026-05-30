"""
Weekly iron condor recommendations.
Pulls live/most-recent prices, checks VIX, computes bands and condor pricing
for every ticker, then ranks by risk-adjusted attractiveness.
"""

import warnings, sys
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import date, timedelta

import os
warnings.filterwarnings("ignore")
sys.path.append("../")
try:
    from model.enhancements import check_earnings_in_window
    HAS_EARNINGS_CHECK = True
except Exception:
    HAS_EARNINGS_CHECK = False

_APPROVED_FILE = os.path.join(os.path.dirname(__file__), "..", "approved_tickers.txt")

def _load_tickers():
    try:
        tickers = []
        with open(_APPROVED_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    tickers.append(line)
        return tickers
    except FileNotFoundError:
        return ["SPY","QQQ","META","MSFT","AAPL","AMZN","GOOG","AMD","TSLA","NVDA"]

TICKERS = _load_tickers()

ACCOUNT    = 37_000
RISK_PCT   = 0.03
MIN_CREDIT = 200   # minimum total credit per trade ($)
RISK_FREE = 0.05
T_DAYS    = 5        # options expiry in trading days
PRED_DAYS = 3        # how many days forward we're estimating price movement
VOL_WIN   = 20
Z         = 1.28     # 80% CI


def bs_price(S, K, T, r, sigma, kind="call"):
    if T <= 0 or sigma <= 0:
        return max(S-K, 0) if kind == "call" else max(K-S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if kind == "call":
        return float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))
    return float(K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))


def price_condor(S, lo, hi, T, r, iv, width):
    sc = bs_price(S, hi,        T, r, iv, "call")
    lc = bs_price(S, hi+width,  T, r, iv, "call")
    sp = bs_price(S, lo,        T, r, iv, "put")
    lp = bs_price(S, lo-width,  T, r, iv, "put")
    credit   = (sc - lc) + (sp - lp)
    max_loss = width - credit
    return round(credit, 3), round(max_loss, 3)


def next_friday():
    today = date.today()
    days_ahead = 4 - today.weekday()   # Friday = 4
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)


def main():
    today     = date.today()
    exp_date  = next_friday()
    print(f"\n{'='*68}")
    print(f"  WEEKLY IRON CONDOR RECOMMENDATIONS")
    print(f"  Analysis date : {today}  |  Target expiry : {exp_date} (Fri)")
    print(f"  Account size  : ${ACCOUNT:,.0f}  |  Risk per trade: {int(RISK_PCT*100)}%")
    print(f"{'='*68}")

    # ── VIX ───────────────────────────────────────────────────────────────
    vix_raw  = yf.download("^VIX", period="30d", interval="1d",
                            auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    vix      = float(vix_raw.iloc[-1])
    vix_1w   = float(vix_raw.iloc[-6]) if len(vix_raw) >= 6 else vix
    vix_chg  = vix - vix_1w

    vix_label = ("🔴 HIGH FEAR" if vix > 30 else
                 "🟡 ELEVATED"  if vix > 20 else
                 "🟢 CALM")

    print(f"\n  VIX : {vix:.1f}  ({vix_label})   1-week change: {vix_chg:+.1f}")

    if vix > 30:
        print("  ⚠  VIX above 30 — consider skipping or sizing down this week.")
    elif vix > 22:
        print("  ⚠  VIX elevated — bands will be wider, credit will be higher, but so will risk.")
    else:
        print("  ✓  Calm VIX — good conditions for iron condors.")

    # ── SPY momentum ─────────────────────────────────────────────────────
    spy_raw  = yf.download("SPY", period="40d", interval="1d",
                            auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    spy_mom  = float(spy_raw.iloc[-1] / spy_raw.iloc[-21] - 1)
    regime   = ("BULLISH" if spy_mom > 0.02 else
                "BEARISH" if spy_mom < -0.02 else "NEUTRAL")
    print(f"  SPY 20d momentum : {spy_mom:+.1%}  ({regime})")

    pred_start = pd.Timestamp(today)
    pred_end   = pred_start + pd.Timedelta(days=14)

    # ── Per-ticker analysis ───────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  Fetching prices and computing bands …")
    print(f"{'─'*68}")

    recs = []
    for ticker in TICKERS:
        raw = yf.download(ticker, period="60d", interval="1d",
                          auto_adjust=True, progress=False)
        raw = raw.dropna()
        if len(raw) < VOL_WIN + 5:
            continue

        closes = raw["Close"].values.flatten()
        S      = float(closes[-1])

        c_slice   = closes[-(VOL_WIN+1):]
        rets      = np.diff(c_slice) / c_slice[:-1]
        daily_vol = np.std(rets)
        iv        = daily_vol * np.sqrt(252) * 1.15

        lo = S * (1 - Z * daily_vol * np.sqrt(PRED_DAYS))
        hi = S * (1 + Z * daily_vol * np.sqrt(PRED_DAYS))

        if   S <  20:  width = 0.5
        elif S <  50:  width = 1.0
        elif S < 100:  width = 2.5
        elif S < 200:  width = 4.0
        else:          width = 5.0
        T_exp  = T_DAYS / 252

        credit, max_loss = price_condor(S, lo, hi, T_exp, RISK_FREE, iv, width)
        if credit <= 0 or max_loss <= 0:
            continue

        max_risk     = ACCOUNT * RISK_PCT
        n_contracts  = max(1, min(5, int(max_risk / (max_loss * 100))))
        actual_risk  = n_contracts * max_loss * 100
        actual_cred  = n_contracts * credit * 100

        if actual_cred < MIN_CREDIT:
            continue

        be_win_rate  = max_loss / (max_loss + credit) * 100
        credit_pct   = credit / S * 100      # credit as % of stock price
        band_width   = (hi - lo) / S * 100   # band as % of stock price

        # Earnings flag
        earnings_flag = ""
        if HAS_EARNINGS_CHECK:
            try:
                has_earn, earn_dates = check_earnings_in_window(ticker, pred_start, pred_end)
                if has_earn:
                    earnings_flag = f" ⚠ EARNINGS {earn_dates[0].strftime('%b %d')}"
            except Exception:
                pass

        # Score: higher credit%, wider band relative to IV, lower breakeven = better
        # Penalise hard for earnings in window
        score = (credit_pct * 2) + (band_width / (daily_vol * np.sqrt(252) * 100)) - (be_win_rate / 100)
        if earnings_flag:
            score -= 5   # big penalty

        recs.append({
            "ticker":       ticker,
            "price":        S,
            "lo":           lo,
            "hi":           hi,
            "width":        width,
            "band_pct":     band_width,
            "daily_vol":    daily_vol * 100,
            "iv":           iv * 100,
            "credit":       credit,
            "max_loss":     max_loss,
            "n_contracts":  n_contracts,
            "actual_risk":  actual_risk,
            "actual_cred":  actual_cred,
            "be_win_rate":  be_win_rate,
            "score":        score,
            "earnings":     earnings_flag,
        })

    if not recs:
        print("  No valid condors found."); return

    recs.sort(key=lambda x: x["score"], reverse=True)

    monday = today + timedelta(days=(7 - today.weekday()) % 7)

    # ── Full ranked table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  ALL STOCKS RANKED — HIGHEST TO LOWEST CONVICTION")
    print(f"  Week of {monday.strftime('%B %d')} | Expiry {exp_date.strftime('%b %d')} | VIX={vix:.1f} | SPY {spy_mom:+.1%} ({regime})")
    print(f"{'='*80}")
    print(f"  {'Rk':<3} {'Ticker':<6} {'Price':>7}  {'Safe zone':^21}  "
          f"{'Credit/sh':>9} {'MaxLoss/sh':>10} {'BE%':>6}  {'Contracts':>9}  {'Collect':>8}  {'Max Risk':>8}  {'Why'}")
    print(f"  {'─'*3} {'─'*6} {'─'*7}  {'─'*21}  {'─'*9} {'─'*10} {'─'*6}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*20}")

    conviction_labels = [
        "Highest conviction",
        "High conviction",
        "High conviction",
        "Solid",
        "Solid",
        "Moderate",
        "Moderate",
        "Lower",
        "Lower",
        "Lowest conviction",
    ]

    for i, r in enumerate(recs):
        flag  = " ⚠EARN" if r["earnings"] else ""
        label = conviction_labels[min(i, len(conviction_labels)-1)]
        # Round strikes to nearest $0.50 for display
        sell_put  = round(r["lo"]  / 0.5) * 0.5
        buy_put   = sell_put - r["width"]
        sell_call = round(r["hi"]  / 0.5) * 0.5
        buy_call  = sell_call + r["width"]
        r["sell_put"]  = sell_put
        r["buy_put"]   = buy_put
        r["sell_call"] = sell_call
        r["buy_call"]  = buy_call
        print(f"  {i+1:<3} {r['ticker']:<6} ${r['price']:>6.2f}  "
              f"${sell_put:>6.2f}–${sell_call:<6.2f}  "
              f"${r['credit']:>7.3f}    ${r['max_loss']:>7.3f}   "
              f"{r['be_win_rate']:>5.1f}%  "
              f"{r['n_contracts']:>4}x      "
              f"${r['actual_cred']:>6.0f}    "
              f"${r['actual_risk']:>6.0f}  "
              f"{label}{flag}")

    # ── Exact order instructions ──────────────────────────────────────────
    top = [r for r in recs if not r["earnings"]][:5]   # top 5 without earnings risk

    print(f"\n{'='*80}")
    print(f"  EXACT WEBULL ORDER INSTRUCTIONS — WEEK OF {monday.strftime('%B %d, %Y').upper()}")
    print(f"  All expire: {exp_date.strftime('%A %B %d, %Y')} | Account: ${ACCOUNT:,.0f} | Risk/trade: {int(RISK_PCT*100)}%")
    print(f"{'='*80}")

    entry_days = [
        (monday,                     "Monday",    "Open 9:30–10:30am — most liquid"),
        (monday + timedelta(days=1), "Tuesday",   "Open 9:30–10:30am"),
        (monday + timedelta(days=1), "Tuesday",   "After first trade fills"),
        (monday + timedelta(days=2), "Wednesday", "Open 9:30–10:30am"),
        (monday + timedelta(days=2), "Wednesday", "After first Wed trade fills"),
    ]

    for i, r in enumerate(top):
        entry_date, day_name, timing = entry_days[i]
        # Credit range: mid ± 10%
        mid       = round(r["credit"], 2)
        lim_start = round(mid * 1.05, 2)   # ask for 5% more than mid first
        lim_floor = round(mid * 0.85, 2)   # floor: 85% of mid, don't go lower

        print(f"\n  ── TRADE #{i+1}: {r['ticker']} ──────────────────────────────────────────")
        print(f"  Enter    : {day_name} {entry_date.strftime('%b %d')}  ({timing})")
        print(f"  Expire   : {exp_date.strftime('%A %b %d')}")
        print(f"  Conviction: {conviction_labels[i]}")
        print(f"  Stock now: ${r['price']:.2f}")
        print(f"")
        print(f"  WEBULL ORDER — set each field exactly:")
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  Strategy    : Iron Condor                              │")
        print(f"  │  Expiration  : {exp_date.strftime('%b %d, %Y')} (weekly)                    │")
        print(f"  │  Side        : SELL  ← critical, must be Sell           │")
        print(f"  │                                                         │")
        print(f"  │  Leg 1  Buy  put   ${r['buy_put']:>7.2f}                           │")
        print(f"  │  Leg 2  Sell put   ${r['sell_put']:>7.2f}  ← short put strike        │")
        print(f"  │  Leg 3  Sell call  ${r['sell_call']:>7.2f}  ← short call strike       │")
        print(f"  │  Leg 4  Buy  call  ${r['buy_call']:>7.2f}                           │")
        print(f"  │                                                         │")
        print(f"  │  Contracts   : {r['n_contracts']}                                        │")
        print(f"  │  Order Type  : LIMIT                                    │")
        print(f"  │  Limit Price : {lim_start:.2f}  (lower to {lim_floor:.2f} if no fill in 10 min) │")
        print(f"  │  Time-in-Force: Day                                     │")
        print(f"  └─────────────────────────────────────────────────────────┘")
        print(f"  If filled at {lim_start:.2f}: collect ${lim_start*100*r['n_contracts']:,.0f}  |  "
              f"Max loss: ${r['actual_risk']:,.0f}")
        print(f"  Win if {r['ticker']} stays between ${r['sell_put']:.2f} and ${r['sell_call']:.2f} by {exp_date.strftime('%b %d')}")
        if r["earnings"]:
            print(f"  ⚠  {r['earnings']} — SKIP THIS TRADE")

    print(f"\n{'='*80}")
    print(f"  MANAGEMENT RULES (same for all trades):")
    print(f"  • 50% profit target: if your credit was $X, close when you can buy back for $X/2")
    print(f"  • 50% loss limit: if your credit was $X, close if position costs $2X to close")
    print(f"  • VIX spike >25 mid-week: close all positions immediately, take whatever P&L")
    print(f"  • Thursday: if still open and stock is comfortably inside bands, let expire Friday")
    print(f"  • Thursday: if stock is within $2 of a short strike, close — don't gamble on Friday")
    print(f"{'='*80}")
    print(f"\n  ⚠  PAPER TRADE FIRST. Verify earnings dates on Webull before entering.")
    print(f"  These are 15-min delayed quotes. Actual fills will differ slightly.\n")


if __name__ == "__main__":
    main()
