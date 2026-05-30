"""
Iron condor backtest — Kronos center-shifted vs. vol-only baseline.

Runs two strategies side-by-side on the same dates and tickers:
  BASELINE : condor centered on current price S  (pure realized-vol bands)
  KRONOS   : condor center shifted to Kronos's predicted price in N days

The band *width* is identical in both versions — only the center moves.
Example: if SHOP is at $118 and Kronos predicts $126 by Wednesday,
the Kronos condor box slides up $8, giving more call-side cushion.

Shift is capped at ±MAX_SHIFT (default 4%) to prevent extreme predictions
from flipping the condor completely.

Usage:
    cd examples && python3 backtest_condor_kronos.py
    cd examples && python3 backtest_condor_kronos.py --start 2024-08-01 --end 2024-09-01
    cd examples && python3 backtest_condor_kronos.py --samples 10  # more accurate, slower
"""

import sys, warnings, argparse
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

TICKERS      = ["PLTR","SNAP","SOFI","SHOP","NVDA","AAPL","MSFT","GOOGL","META","AMZN"]
ACCOUNT      = 37_000
RISK_PCT     = 0.07
RISK_FREE    = 0.05
T_DAYS       = 5        # options expiry (trading days)
PRED_DAYS    = 3        # how far ahead we check actual price + Kronos target
VOL_WIN      = 20
Z            = 1.28     # 80% CI
MAX_SHIFT    = 0.04     # cap center shift at ±4% of stock price
KRONOS_T     = 0.7      # temperature — lower = more conservative predictions
DEFAULT_SAMPLES = 5     # Kronos sample count; increase for accuracy at cost of time


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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_daily(ticker):
    raw = yf.download(ticker, start="2023-01-01", interval="1d",
                      auto_adjust=True, progress=False)
    raw = raw.dropna()
    df = pd.DataFrame({
        "date":       pd.to_datetime(raw.index.tz_localize(None) if raw.index.tz else raw.index),
        "timestamps": pd.to_datetime(raw.index.tz_localize(None) if raw.index.tz else raw.index),
        "open":   raw["Open"].values.flatten(),
        "high":   raw["High"].values.flatten(),
        "low":    raw["Low"].values.flatten(),
        "close":  raw["Close"].values.flatten(),
        "volume": raw["Volume"].values.flatten(),
        "amount": raw["Open"].values.flatten() * raw["Volume"].values.flatten(),
    })
    return df.reset_index(drop=True)


# ── Kronos prediction ─────────────────────────────────────────────────────────

def get_kronos_center(predictor, df, cutoff_ts, pred_days, sample_count):
    """
    Slice df to cutoff_ts, run Kronos on daily bars, return predicted close
    at pred_days business days out. Returns None on failure.
    """
    df_in = df[df["date"] <= cutoff_ts].copy()
    if len(df_in) < 60:
        return None

    df_input = df_in[["open","high","low","close","volume","amount"]].tail(400).reset_index(drop=True)
    x_ts     = df_in["timestamps"].tail(400).reset_index(drop=True)

    y_ts = pd.Series(pd.bdate_range(start=cutoff_ts, periods=pred_days + 1)[1:])

    try:
        pred_df = predictor.predict(
            df=df_input,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_days,
            T=KRONOS_T,
            top_p=0.9,
            sample_count=sample_count,
            verbose=False,
        )
        return float(pred_df["close"].iloc[-1])
    except Exception:
        return None


# ── Cutoff date helpers ───────────────────────────────────────────────────────

def get_calm_cutoffs(n=10, vix_max=22):
    vix = yf.download("^VIX", start="2024-06-01", interval="1d",
                      auto_adjust=True, progress=False)["Close"].squeeze().dropna()
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


def get_date_range_cutoffs(start_str, end_str, n=5, spacing_days=5):
    vix = yf.download("^VIX", start=start_str,
                      end=(pd.Timestamp(end_str) + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                      interval="1d", auto_adjust=True, progress=False)["Close"].squeeze().dropna()
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


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(label, trades, account_end):
    if not trades:
        print(f"\n  {label}: no trades")
        return
    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_pnl    = sum(t["pnl"] for t in trades)
    win_rate     = len(wins) / len(trades) * 100
    total_credit = sum(t["credit"] for t in trades)

    running = [ACCOUNT] + [t["account"] for t in trades]
    max_dd, peak = 0.0, ACCOUNT
    for val in running:
        peak = max(peak, val)
        dd   = (peak - val) / peak * 100
        max_dd = max(max_dd, dd)

    weeks         = len(set(t["date"] for t in trades))
    annual_factor = 52 / max(weeks, 1)

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Trades       : {len(trades)}  ({len(wins)}W / {len(losses)}L)   "
          f"Win rate: {win_rate:.1f}%")
    print(f"  Total P&L    : ${total_pnl:>+10,.0f}   ({total_pnl/ACCOUNT*100:+.2f}%)")
    print(f"  Ending acct  : ${account_end:>10,.0f}")
    if wins:
        print(f"  Avg win      : ${np.mean([t['pnl'] for t in wins]):>+10,.0f}")
    if losses:
        print(f"  Avg loss     : ${np.mean([t['pnl'] for t in losses]):>+10,.0f}")
    print(f"  Max drawdown : {max_dd:.1f}%")
    print(f"  Total premium: ${total_credit:>10,.0f}")
    print(f"  Ann. P&L est : ${total_pnl * annual_factor:>+10,.0f}")
    if wins and losses:
        be = abs(np.mean([t['pnl'] for t in losses])) / \
             (np.mean([t['pnl'] for t in wins]) + abs(np.mean([t['pnl'] for t in losses]))) * 100
        print(f"  BE win rate  : {be:.1f}%")

    print(f"\n  Per-ticker:")
    print(f"  {'Ticker':<8} {'T':>4} {'W':>4} {'Win%':>6} {'P&L':>10}")
    print(f"  {'─'*36}")
    for ticker in sorted(set(t["ticker"] for t in trades)):
        tt = [t for t in trades if t["ticker"] == ticker]
        if not tt:
            continue
        tw = sum(1 for t in tt if t["result"] == "WIN")
        tp = sum(t["pnl"] for t in tt)
        print(f"  {ticker:<8} {len(tt):>4} {tw:>4} {tw/len(tt)*100:>5.0f}%  ${tp:>+8,.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(date_start=None, date_end=None, sample_count=DEFAULT_SAMPLES,
         tickers=None, n_cutoffs=None):
    run_tickers = tickers if tickers else TICKERS

    print(f"\n{'='*70}")
    print(f"  KRONOS IRON CONDOR BACKTEST — A/B COMPARISON")
    print(f"  BASELINE (symmetric) vs KRONOS-SHIFTED")
    print(f"  Tickers: {', '.join(run_tickers)}")
    print(f"  Kronos samples per prediction: {sample_count}  |  Shift cap: ±{MAX_SHIFT*100:.0f}%")
    print(f"{'='*70}")

    print("\nLoading Kronos model …")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    predictor = KronosPredictor(model, tokenizer, max_context=2048)
    print("Model loaded.\n")

    print("Fetching daily price data for all tickers …")
    data = {t: load_daily(t) for t in run_tickers}

    if date_start and date_end:
        label    = f"{date_start} → {date_end}"
        n        = n_cutoffs if n_cutoffs else 20
        print(f"Selecting {n} cutoff dates in range {label} …")
        cutoffs = get_date_range_cutoffs(date_start, date_end, n=n, spacing_days=5)
        if not cutoffs:
            print("No cutoffs found."); return
    else:
        label = "calm VIX (<22)"
        n     = n_cutoffs if n_cutoffs else 10
        print(f"Finding {n} calm-VIX cutoff dates …")
        cutoffs = get_calm_cutoffs(n=n, vix_max=22)
        if not cutoffs:
            print("No calm periods found."); return

    for ts, v in cutoffs:
        print(f"  {ts.date()}  VIX={v}")

    n_predictions = len(run_tickers) * len(cutoffs)
    est_minutes   = n_predictions * sample_count * 2 / 60
    print(f"\n{n_predictions} Kronos predictions to run (~{est_minutes:.0f} min at "
          f"{sample_count} samples each) …\n")

    # ── Per-trade loop ────────────────────────────────────────────
    b_trades, b_account = [], ACCOUNT   # baseline
    k_trades, k_account = [], ACCOUNT   # Kronos-shifted

    COL = 110
    print(f"{'─'*COL}")
    print(f"  {'#':<4} {'Date':<12} {'Ticker':<6} "
          f"{'S':>7}  {'K-pred':>7}  {'Shift':>7}  "
          f"{'BASE lo─hi':^21}  {'KRON lo─hi':^21}  "
          f"{'S_exp':>7}  {'B':>4} {'K':>4}")
    print(f"{'─'*COL}")

    trade_num = 0
    for cutoff_ts, vix_val in cutoffs:
        period_b, period_k = [], []

        for ticker in run_tickers:
            df     = data[ticker]
            df_in  = df[df["date"] <= cutoff_ts].copy()
            df_act = df[df["date"]  > cutoff_ts].head(PRED_DAYS)

            if len(df_in) < VOL_WIN + 5 or len(df_act) < PRED_DAYS:
                continue

            S     = float(df_in["close"].iloc[-1])
            S_exp = float(df_act["close"].iloc[-1])

            # Realized vol → band half-width
            closes    = df_in["close"].values[-(VOL_WIN+1):]
            rets      = np.diff(closes) / closes[:-1]
            daily_vol = np.std(rets)
            iv        = daily_vol * np.sqrt(252) * 1.15
            half_band = S * Z * daily_vol * np.sqrt(PRED_DAYS)

            # Spread width by price tier
            if   S <  20:  width = 0.5
            elif S <  50:  width = 1.0
            elif S < 100:  width = 2.5
            elif S < 200:  width = 4.0
            else:          width = 5.0
            T_exp = T_DAYS / 252

            # ── BASELINE: center = S ─────────────────────────────
            b_lo = S - half_band
            b_hi = S + half_band
            b_credit, b_maxloss = price_condor(S, b_lo, b_hi, T_exp, RISK_FREE, iv, width)
            if b_credit <= 0 or b_maxloss <= 0:
                continue

            # ── KRONOS: center = capped prediction ───────────────
            kronos_raw = get_kronos_center(predictor, df, cutoff_ts,
                                           pred_days=PRED_DAYS,
                                           sample_count=sample_count)
            if kronos_raw is not None:
                raw_shift    = kronos_raw - S
                capped_shift = float(np.clip(raw_shift, -MAX_SHIFT * S, MAX_SHIFT * S))
                k_center     = S + capped_shift
            else:
                k_center     = S          # fallback to baseline
                capped_shift = 0.0

            shift_pct = capped_shift / S * 100
            k_lo = k_center - half_band
            k_hi = k_center + half_band
            k_credit, k_maxloss = price_condor(S, k_lo, k_hi, T_exp, RISK_FREE, iv, width)
            if k_maxloss <= 0:
                k_credit, k_maxloss = b_credit, b_maxloss   # fallback

            # ── Position sizing & P&L ────────────────────────────
            b_n    = max(1, min(5, int(b_account * RISK_PCT / (b_maxloss * 100))))
            b_pnl  = condor_pnl(S_exp, b_lo, b_hi, width, b_credit) * 100 * b_n
            b_account += b_pnl

            k_n    = max(1, min(5, int(k_account * RISK_PCT / (k_maxloss * 100))))
            k_pnl  = condor_pnl(S_exp, k_lo, k_hi, width, k_credit) * 100 * k_n
            k_account += k_pnl

            b_result = "WIN" if b_pnl > 0 else "LOSS"
            k_result = "WIN" if k_pnl > 0 else "LOSS"
            trade_num += 1

            print(f"  {trade_num:<4} {str(cutoff_ts.date()):<12} {ticker:<6} "
                  f"${S:>6.2f}  "
                  f"${k_center:>6.2f}  "
                  f"{shift_pct:>+6.1f}%  "
                  f"${b_lo:>6.1f}─${b_hi:<7.1f} "
                  f"${k_lo:>6.1f}─${k_hi:<7.1f} "
                  f"${S_exp:>6.2f}  "
                  f"{'W' if b_result=='WIN' else 'L':>4} "
                  f"{'W' if k_result=='WIN' else 'L':>4}")

            period_b.append({"date": cutoff_ts.date(), "ticker": ticker,
                              "pnl": b_pnl, "result": b_result,
                              "account": b_account,
                              "credit": b_credit * 100 * b_n})
            period_k.append({"date": cutoff_ts.date(), "ticker": ticker,
                              "pnl": k_pnl, "result": k_result,
                              "account": k_account,
                              "credit": k_credit * 100 * k_n})

        b_trades.extend(period_b)
        k_trades.extend(period_k)
        bw = sum(1 for t in period_b if t["result"] == "WIN")
        kw = sum(1 for t in period_k if t["result"] == "WIN")
        bp = sum(t["pnl"] for t in period_b)
        kp = sum(t["pnl"] for t in period_k)
        print(f"{'─'*COL}  ← period end  "
              f"BASE {bw}/{len(period_b)} wins ${bp:+,.0f}  |  "
              f"KRON {kw}/{len(period_k)} wins ${kp:+,.0f}")

    # ── Summaries ─────────────────────────────────────────────────
    print_summary(f"BASELINE — {label.upper()}", b_trades, b_account)
    print_summary(f"KRONOS-SHIFTED — {label.upper()}", k_trades, k_account)

    # ── Head-to-head delta ────────────────────────────────────────
    b_pnl = sum(t["pnl"] for t in b_trades)
    k_pnl = sum(t["pnl"] for t in k_trades)
    b_wr  = sum(1 for t in b_trades if t["result"]=="WIN") / max(len(b_trades),1) * 100
    k_wr  = sum(1 for t in k_trades if t["result"]=="WIN") / max(len(k_trades),1) * 100

    print(f"\n{'='*65}")
    print(f"  HEAD-TO-HEAD DELTA")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {'Baseline':>12} {'Kronos':>12} {'Delta':>12}")
    print(f"  {'─'*58}")
    print(f"  {'Total P&L':<22} ${b_pnl:>+10,.0f}  ${k_pnl:>+10,.0f}  ${k_pnl-b_pnl:>+10,.0f}")
    print(f"  {'Win rate':<22} {b_wr:>11.1f}%  {k_wr:>11.1f}%  {k_wr-b_wr:>+11.1f}%")
    print(f"  {'Ending account':<22} ${b_account:>10,.0f}  ${k_account:>10,.0f}  "
          f"${k_account-b_account:>+10,.0f}")

    if k_pnl > b_pnl:
        verdict = f"KRONOS WINS  +${k_pnl-b_pnl:,.0f} vs baseline"
    elif b_pnl > k_pnl:
        verdict = f"BASELINE WINS  Kronos underperformed by ${b_pnl-k_pnl:,.0f}"
    else:
        verdict = "TIE"

    print(f"\n  Verdict: {verdict}")
    print(f"\n  Friction note: ~$50/trade not deducted above.")
    print(f"  Shift cap was ±{MAX_SHIFT*100:.0f}% — large Kronos predictions were clamped.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start",    type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end",      type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument("--samples",  type=int, default=DEFAULT_SAMPLES,
                   help=f"Kronos samples per prediction (default {DEFAULT_SAMPLES})")
    p.add_argument("--tickers",  type=str, default=None,
                   help="Comma-separated ticker list, e.g. SPY,PLTR,SHOP,NVDA,COIN")
    p.add_argument("--n",        type=int, default=None,
                   help="Number of cutoff dates (default 20 for date range, 10 for calm mode)")
    args = p.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    main(date_start=args.start, date_end=args.end, sample_count=args.samples,
         tickers=tickers, n_cutoffs=args.n)
