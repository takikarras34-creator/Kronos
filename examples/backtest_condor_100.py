"""
Iron condor backtest — baseline (no Kronos shift).

Uses realized-volatility bands centered on the stock price at each cutoff.
Runs fast since no model inference is needed.

Typical usage
─────────────
# Quick 100-trade calm-VIX run (default)
python backtest_condor_100.py

# Date-range run with 100 cutoffs, all approved tickers
python backtest_condor_100.py --start 2022-01-01 --end 2026-05-30 --n 100 --all-tickers

# Realistic fill prices (bid instead of mid, regulatory fees)
python backtest_condor_100.py --start 2022-01-01 --end 2026-05-30 --n 100 --all-tickers --realistic

# Specific tickers, custom cutoffs
python backtest_condor_100.py --start 2023-01-01 --end 2026-05-30 --n 50 --tickers PLTR,NVDA,SHOP,SPY,COIN
"""

import sys, warnings, os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

TICKERS   = ["PLTR","SNAP","SOFI","SHOP","NVDA","AAPL","MSFT","GOOGL","META","AMZN"]
ACCOUNT   = 37_000
RISK_PCT  = 0.07
RISK_FREE = 0.05

# Realistic fill factors: bid-to-mid ratio by options liquidity tier
# Tier 1 — index ETFs, tightest spreads
# Tier 2 — liquid mega-cap, moderate spreads
# Tier 3 — mid-cap growth, wider spreads
# Tier 4 — small/volatile, widest spreads
FILL_FACTORS = {
    "SPY":   0.90, "QQQ":  0.90,
    "AAPL":  0.82, "MSFT": 0.82, "META":  0.82, "AMZN": 0.82,
    "NVDA":  0.82, "GOOG": 0.82, "GOOGL": 0.82, "AMD":  0.82,
    "NFLX":  0.82, "TSLA": 0.82,
    "PLTR":  0.74, "COIN": 0.74, "SHOP":  0.74, "UBER": 0.74,
    "NOW":   0.74, "ANET": 0.74, "CRM":   0.74, "PYPL": 0.74,
    "DASH":  0.74, "RDDT": 0.74, "CVNA":  0.74,
    "SMCI":  0.63, "SNAP": 0.63, "SOFI":  0.63, "MARA": 0.63,
    "DKNG":  0.63, "RIVN": 0.63,
}
FEE_PER_LEG   = 0.65   # regulatory fee per contract per leg (Webull: $0 commission)
LEGS          = 4
T_DAYS    = 5       # options expiry (trading days)
PRED_DAYS = 3       # how far ahead we check the actual price
VOL_WIN   = 20      # realized vol lookback
Z         = 1.28    # 80% CI = 10th-90th pct

APPROVED_TICKERS_FILE = os.path.join(os.path.dirname(__file__), "..", "approved_tickers.txt")


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, kind="call"):
    if T <= 0 or sigma <= 0:
        return max(S-K, 0) if kind=="call" else max(K-S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if kind == "call":
        return float(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))
    return float(K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))


def price_condor(S, lo, hi, T, r, iv, width):
    sc = bs_price(S, hi,       T, r, iv, "call")
    lc = bs_price(S, hi+width, T, r, iv, "call")
    sp = bs_price(S, lo,       T, r, iv, "put")
    lp = bs_price(S, lo-width, T, r, iv, "put")
    credit   = (sc - lc) + (sp - lp)
    max_loss = width - credit
    return round(credit, 3), round(max_loss, 3)


def condor_pnl(S_exp, lo, hi, width, credit):
    if lo <= S_exp <= hi:
        return credit
    elif S_exp > hi:
        return credit - min(S_exp - hi, width)
    else:
        return credit - min(lo - S_exp, width)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_daily(ticker, start="2021-01-01"):
    raw = yf.download(ticker, start=start, interval="1d",
                      auto_adjust=True, progress=False)
    raw = raw.dropna()
    df = pd.DataFrame({
        "date":   pd.to_datetime(raw.index.tz_localize(None) if raw.index.tz else raw.index),
        "close":  raw["Close"].values.flatten(),
        "high":   raw["High"].values.flatten(),
        "low":    raw["Low"].values.flatten(),
        "open":   raw["Open"].values.flatten(),
        "volume": raw["Volume"].values.flatten(),
    })
    return df.reset_index(drop=True)


def load_approved_tickers():
    """Load tickers from approved_tickers.txt, skipping comments and blank lines."""
    tickers = []
    try:
        with open(APPROVED_TICKERS_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    tickers.append(line)
    except FileNotFoundError:
        print(f"  Warning: {APPROVED_TICKERS_FILE} not found, using default list.")
        return TICKERS
    return tickers


def load_vix(start="2021-01-01", end=None):
    end_str = end or (pd.Timestamp.today() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    vix = yf.download("^VIX", start=start, end=end_str,
                      interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    return vix


def get_calm_cutoffs(n=10, vix_max=22, start="2021-01-01"):
    """Return n non-overlapping cutoff dates where VIX < vix_max, most recent first."""
    vix = load_vix(start=start)
    vix = vix.iloc[:-PRED_DAYS]
    calm = vix[vix < vix_max].sort_index(ascending=False)

    selected, last = [], pd.Timestamp("1900-01-01")
    for date, val in calm.items():
        ts = pd.Timestamp(date).tz_localize(None) if date.tzinfo else pd.Timestamp(date)
        if (last == pd.Timestamp("1900-01-01")) or abs((ts - last).days) >= 7:
            selected.append((ts, round(val, 1)))
            last = ts
        if len(selected) == n:
            break
    return selected


def get_date_range_cutoffs(start_str, end_str, n=20, spacing_days=5):
    """
    Return up to n evenly-spaced trading-day cutoffs within [start_str, end_str].
    Each cutoff must have at least PRED_DAYS of actual data after it.
    """
    vix = load_vix(start=start_str, end=(pd.Timestamp(end_str) + pd.Timedelta(days=10)).strftime("%Y-%m-%d"))

    start_ts = pd.Timestamp(start_str)
    end_ts   = pd.Timestamp(end_str)

    in_window = vix[(vix.index >= start_ts) & (vix.index <= end_ts)]
    in_window = in_window.iloc[:-PRED_DAYS] if len(in_window) > PRED_DAYS else in_window

    if len(in_window) == 0:
        return []

    dates = in_window.index.to_list()
    step  = max(1, len(dates) // n)
    candidates = dates[::step][:n]

    selected, last = [], pd.Timestamp("1900-01-01")
    for d in candidates:
        ts = pd.Timestamp(d).tz_localize(None) if d.tzinfo else pd.Timestamp(d)
        if abs((ts - last).days) >= spacing_days or last == pd.Timestamp("1900-01-01"):
            v = float(in_window.loc[d])
            selected.append((ts, round(v, 1)))
            last = ts
    return selected


def vix_label(v):
    if v < 20:   return "CALM"
    if v < 30:   return "ELEVATED"
    return "HIGH"


def spy_regime(cutoff_ts, spy_df):
    """Return SPY 20-day momentum regime at cutoff date."""
    sub = spy_df[spy_df["date"] <= cutoff_ts]
    if len(sub) < 22:
        return "UNKNOWN"
    mom = float(sub["close"].iloc[-1] / sub["close"].iloc[-22] - 1)
    if mom > 0.02:   return "BULL"
    if mom < -0.02:  return "BEAR"
    return "NEUTRAL"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(date_start=None, date_end=None, tickers=None, n_cutoffs=20, realistic=False, account_size=None, max_contracts=5, risk_pct=None):
    run_tickers   = tickers if tickers else TICKERS
    starting_acct = account_size if account_size else ACCOUNT
    trade_risk    = risk_pct if risk_pct is not None else RISK_PCT

    # Determine data fetch start — pull extra history for vol calculation
    data_start = "2021-01-01" if not date_start else (
        pd.Timestamp(date_start) - pd.Timedelta(days=90)
    ).strftime("%Y-%m-%d")
    # Clamp to earliest yfinance data that's reliable
    if data_start < "2015-01-01":
        data_start = "2015-01-01"

    fill_mode = "REALISTIC (bid prices + fees)" if realistic else "THEORETICAL (mid prices)"
    print(f"\n{'='*72}")
    print(f"  IRON CONDOR BACKTEST — BASELINE (no Kronos shift)")
    print(f"  Tickers : {len(run_tickers)}  ({', '.join(run_tickers[:6])}{'…' if len(run_tickers) > 6 else ''})")
    print(f"  Account : ${starting_acct:,.0f}  |  Risk/trade: {trade_risk*100:.1f}%  |  Z={Z}  |  Max contracts: {max_contracts}")
    print(f"  Fill mode: {fill_mode}")
    print(f"{'='*72}")

    print(f"Fetching price data for {len(run_tickers)} tickers …")
    data = {}
    for t in run_tickers:
        df = load_daily(t, start=data_start)
        if len(df) >= VOL_WIN + 10:
            data[t] = df
        else:
            print(f"  Skipping {t} — insufficient history ({len(df)} rows)")
    valid_tickers = list(data.keys())
    print(f"  {len(valid_tickers)}/{len(run_tickers)} tickers have sufficient data.")

    # Also load SPY for regime detection (always needed)
    if "SPY" not in data:
        spy_df = load_daily("SPY", start=data_start)
    else:
        spy_df = data["SPY"]

    if date_start and date_end:
        label = f"{date_start} → {date_end}"
        print(f"Selecting {n_cutoffs} cutoff dates in range {label} …")
        cutoffs = get_date_range_cutoffs(date_start, date_end, n=n_cutoffs, spacing_days=5)
        if not cutoffs:
            print("Could not find any trading dates in that range."); return
    else:
        label = "calm VIX (<22) — most recent periods"
        print(f"Finding {n_cutoffs} calm-VIX (< 22) cutoff dates …")
        cutoffs = get_calm_cutoffs(n=n_cutoffs, vix_max=22)
        if not cutoffs:
            print("Could not find enough calm periods."); return

    print(f"  {len(cutoffs)} cutoffs selected.")
    print(f"  First: {cutoffs[0][0].date()}  VIX={cutoffs[0][1]}")
    print(f"  Last : {cutoffs[-1][0].date()}  VIX={cutoffs[-1][1]}")

    trade_num = 0
    risk_label = f"{int(RISK_PCT*100)}% Risk"
    print(f"\n{'─'*80}")
    print(f"  {'#':<5} {'Date':<12} {'Ticker':<6} {'VIX':>5} {'Regime':>7}  "
          f"{'Acct Before':>12} {'Result':>6} {'P&L':>8} {'Acct After':>11}")
    print(f"{'─'*80}")

    trades  = []
    account = starting_acct

    for cutoff_ts, vix_val in cutoffs:
        period_trades = []
        regime = spy_regime(cutoff_ts, spy_df)

        for ticker in valid_tickers:
            df     = data[ticker]
            df_in  = df[df["date"] <= cutoff_ts].copy()
            df_act = df[df["date"]  > cutoff_ts].head(PRED_DAYS)

            if len(df_in) < VOL_WIN + 5 or len(df_act) < PRED_DAYS:
                continue

            S     = float(df_in["close"].iloc[-1])
            S_exp = float(df_act["close"].iloc[-1])

            closes    = df_in["close"].values[-(VOL_WIN+1):]
            rets      = np.diff(closes) / closes[:-1]
            daily_vol = np.std(rets)
            lo = S * (1 - Z * daily_vol * np.sqrt(PRED_DAYS))
            hi = S * (1 + Z * daily_vol * np.sqrt(PRED_DAYS))

            iv    = daily_vol * np.sqrt(252) * 1.15
            if   S <  20:  width = 0.5
            elif S <  50:  width = 1.0
            elif S < 100:  width = 2.5
            elif S < 200:  width = 4.0
            else:          width = 5.0
            T_exp = T_DAYS / 252

            credit, max_loss = price_condor(S, lo, hi, T_exp, RISK_FREE, iv, width)
            if credit <= 0 or max_loss <= 0:
                continue

            # Apply realistic fill: bid price + regulatory fees
            if realistic:
                fill_f       = FILL_FACTORS.get(ticker, 0.74)
                fill_credit  = credit * fill_f
                fill_max_loss = width - fill_credit
            else:
                fill_credit   = credit
                fill_max_loss = max_loss

            account_before = account
            max_risk       = account * trade_risk
            n_contracts    = max(1, min(max_contracts, int(max_risk / (fill_max_loss * 100))))
            if n_contracts < 1:
                continue
            actual_risk    = n_contracts * fill_max_loss * 100
            fees           = FEE_PER_LEG * LEGS * n_contracts if realistic else 0.0

            pnl_ps  = condor_pnl(S_exp, lo, hi, width, fill_credit)
            pnl     = pnl_ps * 100 * n_contracts - fees
            result  = "WIN" if pnl > 0 else "LOSS"
            account += pnl
            trade_num += 1

            print(f"  {trade_num:<5} {str(cutoff_ts.date()):<12} {ticker:<6} "
                  f"{vix_val:>5.1f} {regime:>7}  "
                  f"${account_before:>10,.0f}  "
                  f"{'WIN' if pnl>0 else 'LOSS':>6}  "
                  f"${pnl:>+7,.0f}  "
                  f"${account:>10,.0f}")

            period_trades.append({
                "date": cutoff_ts.date(), "ticker": ticker,
                "vix": vix_val, "vix_bucket": vix_label(vix_val),
                "spy_regime": regime,
                "start": S, "lo": lo, "hi": hi, "S_exp": S_exp,
                "credit": fill_credit * 100 * n_contracts,
                "risk": actual_risk, "pnl": pnl,
                "result": result, "account": account,
            })

        trades.extend(period_trades)
        wins  = sum(1 for t in period_trades if t["result"] == "WIN")
        p_pnl = sum(t["pnl"] for t in period_trades)
        print(f"{'─'*80}  ← {str(cutoff_ts.date())} end  "
              f"{wins}/{len(period_trades)} wins  P&L ${p_pnl:+,.0f}")

    if not trades:
        print("No trades completed."); return

    # ── Summary ───────────────────────────────────────────────────────────────
    wins_list   = [t for t in trades if t["result"] == "WIN"]
    losses_list = [t for t in trades if t["result"] == "LOSS"]
    total_pnl    = sum(t["pnl"] for t in trades)
    total_credit = sum(t["credit"] for t in trades)
    win_rate     = len(wins_list) / len(trades) * 100

    running = [t["account"] for t in trades]
    max_drawdown = 0.0
    peak = starting_acct
    for val in running:
        if val > peak: peak = val
        dd = (peak - val) / peak * 100
        if dd > max_drawdown: max_drawdown = dd

    weeks_covered = len(cutoffs)
    annual_factor = 52 / weeks_covered
    annual_pnl    = total_pnl * annual_factor
    annual_roi    = annual_pnl / starting_acct * 100

    avg_win  = np.mean([t["pnl"] for t in wins_list]) if wins_list else 0
    avg_loss = np.mean([t["pnl"] for t in losses_list]) if losses_list else 0
    be_wr    = abs(avg_loss) / (avg_win + abs(avg_loss)) * 100 if wins_list and losses_list else 0

    print(f"\n{'='*72}")
    print(f"  IRON CONDOR BASELINE — {label.upper()}")
    print(f"{'='*72}")
    print(f"  Starting account   : ${starting_acct:>10,.0f}")
    print(f"  Ending account     : ${account:>10,.0f}")
    print(f"  Total P&L          : ${total_pnl:>+10,.0f}")
    print(f"  Total return       : {total_pnl/starting_acct*100:>+9.2f}%")
    print(f"{'─'*72}")
    print(f"  Trades             : {len(trades)}  ({len(wins_list)}W / {len(losses_list)}L)")
    print(f"  Win rate           : {win_rate:.1f}%")
    print(f"  Avg win            : ${avg_win:>+,.0f}")
    print(f"  Avg loss           : ${avg_loss:>+,.0f}" if losses_list else "  Avg loss           : N/A")
    print(f"  Largest single win : ${max(t['pnl'] for t in trades):>+,.0f}")
    print(f"  Largest single loss: ${min(t['pnl'] for t in trades):>+,.0f}")
    print(f"  Max drawdown       : {max_drawdown:.1f}%")
    print(f"  Breakeven win rate : {be_wr:.1f}%")
    print(f"  Total premium sold : ${total_credit:>10,.0f}")
    print(f"{'─'*72}")
    print(f"  Annualised P&L est : ${annual_pnl:>+10,.0f}  ({annual_roi:+.1f}% ROI)")

    # ── VIX regime breakdown ──────────────────────────────────────────────────
    print(f"\n  VIX REGIME BREAKDOWN:")
    print(f"  {'Regime':<12} {'Trades':>7} {'Wins':>6} {'Win%':>6} {'Avg Win':>9} {'Avg Loss':>9} {'P&L':>10}")
    print(f"  {'─'*64}")
    for bucket in ["CALM", "ELEVATED", "HIGH"]:
        bt = [t for t in trades if t["vix_bucket"] == bucket]
        if not bt: continue
        bw = [t for t in bt if t["result"] == "WIN"]
        bl = [t for t in bt if t["result"] == "LOSS"]
        wr = len(bw) / len(bt) * 100
        aw = np.mean([t["pnl"] for t in bw]) if bw else 0
        al = np.mean([t["pnl"] for t in bl]) if bl else 0
        bp = sum(t["pnl"] for t in bt)
        print(f"  {bucket:<12} {len(bt):>7} {len(bw):>6} {wr:>5.0f}%  "
              f"${aw:>+7,.0f}   ${al:>+7,.0f}   ${bp:>+8,.0f}")

    # ── SPY regime breakdown ───────────────────────────────────────────────────
    print(f"\n  SPY REGIME BREAKDOWN:")
    print(f"  {'Regime':<10} {'Trades':>7} {'Wins':>6} {'Win%':>6} {'P&L':>10}")
    print(f"  {'─'*44}")
    for regime_name in ["BULL", "NEUTRAL", "BEAR"]:
        rt = [t for t in trades if t["spy_regime"] == regime_name]
        if not rt: continue
        rw = [t for t in rt if t["result"] == "WIN"]
        rp = sum(t["pnl"] for t in rt)
        wr = len(rw) / len(rt) * 100
        print(f"  {regime_name:<10} {len(rt):>7} {len(rw):>6} {wr:>5.0f}%  ${rp:>+8,.0f}")

    # ── Per-ticker breakdown ───────────────────────────────────────────────────
    print(f"\n  PER-TICKER BREAKDOWN:")
    print(f"  {'Ticker':<8} {'Trades':>7} {'Wins':>6} {'Win%':>6} {'Avg Win':>9} {'Avg Loss':>9} {'P&L':>10}")
    print(f"  {'─'*62}")
    ticker_pnls = []
    for ticker in sorted(set(t["ticker"] for t in trades)):
        tt = [t for t in trades if t["ticker"] == ticker]
        tw = [t for t in tt if t["result"] == "WIN"]
        tl = [t for t in tt if t["result"] == "LOSS"]
        tp = sum(t["pnl"] for t in tt)
        wr = len(tw) / len(tt) * 100
        aw = np.mean([t["pnl"] for t in tw]) if tw else 0
        al = np.mean([t["pnl"] for t in tl]) if tl else 0
        ticker_pnls.append((ticker, tp))
        print(f"  {ticker:<8} {len(tt):>7} {len(tw):>6} {wr:>5.0f}%  "
              f"${aw:>+7,.0f}   ${al:>+7,.0f}   ${tp:>+8,.0f}")

    best  = max(ticker_pnls, key=lambda x: x[1])
    worst = min(ticker_pnls, key=lambda x: x[1])
    print(f"\n  Best ticker : {best[0]}  (${best[1]:+,.0f})")
    print(f"  Worst ticker: {worst[0]}  (${worst[1]:+,.0f})")

    # ── Friction note ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    if realistic:
        print(f"  Fill discount (bid vs mid) and regulatory fees already included above.")
        print(f"  These results reflect what you would actually collect at the bid.")
    else:
        friction = len(trades) * 50
        print(f"  Real friction (~$50/trade in spreads) NOT modelled above.")
        print(f"  Estimated friction ({len(trades)} trades × $50): -${friction:,.0f}")
        print(f"  P&L after friction : ${total_pnl - friction:+,.0f}  "
              f"({(total_pnl-friction)/starting_acct*100:+.1f}%)")
        print(f"  Run with --realistic for bid-price fills and per-ticker fill factors.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Iron condor baseline backtest")
    p.add_argument("--start",        type=str, default=None,  help="Start date YYYY-MM-DD")
    p.add_argument("--end",          type=str, default=None,  help="End date YYYY-MM-DD")
    p.add_argument("--n",            type=int, default=20,    help="Number of cutoff dates (default 20)")
    p.add_argument("--tickers",      type=str, default=None,  help="Comma-separated tickers, e.g. PLTR,NVDA,SPY")
    p.add_argument("--all-tickers",  action="store_true",     help="Use all tickers from approved_tickers.txt")
    p.add_argument("--realistic",    action="store_true",     help="Apply bid-price fill factors and regulatory fees")
    p.add_argument("--account",       type=float, default=None, help="Starting account size in dollars (default 37000)")
    p.add_argument("--max-contracts", type=int,   default=5,   help="Max contracts per trade (default 5; raise for large accounts)")
    p.add_argument("--risk",          type=float, default=None, help="Risk per trade as a decimal, e.g. 0.03 for 3%% (default 0.07)")
    args = p.parse_args()

    if args.all_tickers:
        ticker_list = load_approved_tickers()
    elif args.tickers:
        ticker_list = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        ticker_list = None  # uses default TICKERS constant

    main(
        date_start=args.start,
        date_end=args.end,
        tickers=ticker_list,
        n_cutoffs=args.n,
        realistic=args.realistic,
        account_size=args.account,
        max_contracts=args.max_contracts,
        risk_pct=args.risk,
    )
