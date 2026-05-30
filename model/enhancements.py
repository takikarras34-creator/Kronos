"""
Kronos prediction enhancements:
  - Earnings calendar masking
  - SPY market-regime beta correction
  - Rolling 5-day prediction
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ── 1. Earnings calendar ──────────────────────────────────────────────────────

def get_earnings_dates(ticker: str, window_days: int = 40) -> list:
    """Return earnings dates for `ticker` within the next `window_days`."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return []
        # yfinance returns a dict with 'Earnings Date' as a list or a single date
        dates = cal.get("Earnings Date", [])
        if isinstance(dates, pd.Timestamp):
            dates = [dates]
        return [pd.Timestamp(d) for d in dates]
    except Exception:
        return []


def check_earnings_in_window(ticker: str, pred_start: pd.Timestamp,
                              pred_end: pd.Timestamp) -> tuple:
    """
    Returns (has_earnings: bool, dates: list).
    Warns if an earnings print falls inside the prediction window.
    """
    dates = get_earnings_dates(ticker)
    hits = [d for d in dates if pred_start <= d <= pred_end]
    return bool(hits), hits


# ── 2. SPY regime + beta correction ──────────────────────────────────────────

def _fetch_returns(ticker: str, period: str = "1y") -> pd.Series:
    data = yf.download(ticker, period=period, interval="1d",
                       auto_adjust=True, progress=False)
    close = data["Close"].squeeze().dropna()
    return close.pct_change().dropna()


def compute_beta(stock_ticker: str, market_ticker: str = "SPY",
                 period: str = "1y") -> float:
    """Compute trailing 1-year beta of stock vs market."""
    try:
        r_stock  = _fetch_returns(stock_ticker, period)
        r_market = _fetch_returns(market_ticker, period)
        aligned  = pd.concat([r_stock, r_market], axis=1).dropna()
        aligned.columns = ["stock", "market"]
        cov = np.cov(aligned["stock"], aligned["market"])
        return float(cov[0, 1] / cov[1, 1])
    except Exception:
        return 1.0


def get_spy_momentum(lookback_days: int = 20) -> float:
    """
    Returns SPY's return over the last `lookback_days` trading days.
    Positive = bullish regime, negative = bearish.
    """
    try:
        data = yf.download("SPY", period="60d", interval="1d",
                           auto_adjust=True, progress=False)
        close = data["Close"].squeeze().dropna()
        if len(close) < lookback_days + 1:
            return 0.0
        return float(close.iloc[-1] / close.iloc[-lookback_days - 1] - 1)
    except Exception:
        return 0.0


def apply_regime_correction(pred_df: pd.DataFrame, cutoff_close: float,
                             spy_momentum: float, beta: float,
                             scale: float = 0.5) -> pd.DataFrame:
    """
    Nudge the predicted close (and OHLC) toward the market direction.

    scale controls how aggressively to apply the correction (0 = off, 1 = full).
    With scale=0.5 we apply half the beta-adjusted market move as a correction.
    """
    pred_df = pred_df.copy()
    n = len(pred_df)
    # Linear ramp so correction builds over the forecast horizon
    ramp = np.linspace(0, 1, n)
    correction = cutoff_close * spy_momentum * beta * scale * ramp

    for col in ["open", "high", "low", "close"]:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col] + correction
        if f"{col}_lo" in pred_df.columns:
            pred_df[f"{col}_lo"] = pred_df[f"{col}_lo"] + correction
        if f"{col}_hi" in pred_df.columns:
            pred_df[f"{col}_hi"] = pred_df[f"{col}_hi"] + correction

    return pred_df


# ── 3. VIX volatility regime ─────────────────────────────────────────────────

def get_vix_level() -> float:
    """Returns current VIX. >30 = high fear, 15-20 = normal, <15 = complacent."""
    try:
        data = yf.download("^VIX", period="5d", interval="1d",
                           auto_adjust=True, progress=False)
        close = data["Close"].squeeze().dropna()
        return float(close.iloc[-1]) if len(close) > 0 else 20.0
    except Exception:
        return 20.0


def apply_vix_band_adjustment(pred_df: pd.DataFrame, vix: float) -> pd.DataFrame:
    """
    Widen confidence bands when VIX is elevated.
    VIX 20 = baseline. Each 5 points above 20 widens bands ~10%.
    """
    if vix <= 20:
        return pred_df
    factor = 1.0 + (vix - 20) / 50   # VIX 30 → 1.2×, VIX 40 → 1.4×
    pred_df = pred_df.copy()
    for col in ["open", "high", "low", "close"]:
        if f"{col}_lo" in pred_df.columns:
            mid  = pred_df[col]
            half = (pred_df[f"{col}_hi"] - pred_df[f"{col}_lo"]) / 2
            pred_df[f"{col}_lo"] = mid - half * factor
            pred_df[f"{col}_hi"] = mid + half * factor
    return pred_df


# ── 4. Analyst consensus ──────────────────────────────────────────────────────

def get_analyst_consensus(ticker: str) -> dict:
    """
    Returns analyst mean price target and recommendation from yfinance.
    recommendation_mean: 1=strong buy → 5=strong sell
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "target_mean":          info.get("targetMeanPrice"),
            "target_low":           info.get("targetLowPrice"),
            "target_high":          info.get("targetHighPrice"),
            "recommendation_mean":  info.get("recommendationMean"),
            "num_analysts":         info.get("numberOfAnalystOpinions", 0),
        }
    except Exception:
        return {"target_mean": None, "recommendation_mean": None, "num_analysts": 0}


def apply_analyst_correction(pred_df: pd.DataFrame, cutoff_close: float,
                              analyst: dict, scale: float = 0.3) -> pd.DataFrame:
    """
    Blend analyst consensus price target into the Kronos forecast.
    scale=0.3 means we apply 30% weight toward the analyst direction.
    Requires at least 3 analyst opinions to activate.
    """
    target = analyst.get("target_mean")
    if target is None or (analyst.get("num_analysts") or 0) < 3:
        return pred_df

    pred_df = pred_df.copy()
    n    = len(pred_df)
    ramp = np.linspace(0, 1, n)

    analyst_move = target / cutoff_close - 1
    kronos_move  = pred_df["close"].iloc[-1] / cutoff_close - 1
    delta        = (analyst_move - kronos_move) * scale
    correction   = cutoff_close * delta * ramp

    for col in ["open", "high", "low", "close"]:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col] + correction
        if f"{col}_lo" in pred_df.columns:
            pred_df[f"{col}_lo"] = pred_df[f"{col}_lo"] + correction
        if f"{col}_hi" in pred_df.columns:
            pred_df[f"{col}_hi"] = pred_df[f"{col}_hi"] + correction
    return pred_df


# ── 5. Earnings surprise history ──────────────────────────────────────────────

def get_earnings_surprise(ticker: str) -> float:
    """
    Returns the most recent quarterly earnings surprise as a fraction.
    +0.10 = beat estimates by 10%, -0.05 = missed by 5%, 0.0 = no data.
    """
    try:
        t = yf.Ticker(ticker)
        hist = getattr(t, "earnings_history", None)
        if hist is None or len(hist) == 0:
            # Fallback: quarterly_earnings
            qe = getattr(t, "quarterly_earnings", None)
            if qe is None or len(qe) == 0:
                return 0.0
            # quarterly_earnings has no estimate, so can't compute surprise
            return 0.0
        latest   = hist.sort_index().iloc[-1]
        estimate = latest.get("epsEstimate") or latest.get("EPS Estimate")
        actual   = latest.get("epsActual")   or latest.get("EPS Actual")
        if estimate and estimate != 0 and actual is not None:
            return float((actual - estimate) / abs(estimate))
        return 0.0
    except Exception:
        return 0.0


def apply_earnings_surprise_correction(pred_df: pd.DataFrame, cutoff_close: float,
                                        surprise: float, scale: float = 0.2) -> pd.DataFrame:
    """
    Companies that beat estimates tend to keep beating; those that miss tend to miss.
    Apply a small directional nudge based on the last earnings surprise.
    Ignored if surprise magnitude < 2%.
    """
    if abs(surprise) < 0.02:
        return pred_df
    pred_df = pred_df.copy()
    n    = len(pred_df)
    ramp = np.linspace(0, 1, n)
    # Cap surprise at ±30% to avoid outlier earnings dominating
    surprise_capped = float(np.clip(surprise, -0.30, 0.30))
    correction = cutoff_close * surprise_capped * scale * ramp

    for col in ["open", "high", "low", "close"]:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col] + correction
        if f"{col}_lo" in pred_df.columns:
            pred_df[f"{col}_lo"] = pred_df[f"{col}_lo"] + correction
        if f"{col}_hi" in pred_df.columns:
            pred_df[f"{col}_hi"] = pred_df[f"{col}_hi"] + correction
    return pred_df


# ── 6. News sentiment ─────────────────────────────────────────────────────────

_POS_WORDS = {
    "beat", "beats", "record", "surge", "soar", "rally", "upgrade", "buy",
    "outperform", "strong", "growth", "profit", "gain", "rise", "raised",
    "positive", "bullish", "exceed", "exceeds", "upside", "expand", "boost",
}
_NEG_WORDS = {
    "miss", "misses", "missed", "fall", "drop", "decline", "downgrade", "sell",
    "underperform", "weak", "loss", "cut", "lower", "negative", "bearish",
    "warn", "warning", "lawsuit", "investigation", "layoff", "layoffs", "downside",
    "crash", "concern", "concerns", "risk", "risks", "halt",
}


def get_news_sentiment(ticker: str, finnhub_key: str = None,
                       max_articles: int = 20) -> float:
    """
    Returns sentiment score from -1.0 (very negative) to +1.0 (very positive).

    If finnhub_key is provided, uses Finnhub's news-sentiment endpoint.
    Otherwise falls back to yfinance headlines + keyword scoring.
    """
    if finnhub_key:
        try:
            import requests
            url = (f"https://finnhub.io/api/v1/news-sentiment"
                   f"?symbol={ticker}&token={finnhub_key}")
            data = requests.get(url, timeout=5).json()
            bull = data.get("sentiment", {}).get("bullishPercent", 0.5)
            return float(bull) * 2 - 1   # 0–1 → -1 to +1
        except Exception:
            pass

    try:
        raw_news = yf.Ticker(ticker).news or []
        scores   = []
        for article in raw_news[:max_articles]:
            # yfinance ≥0.2.50 nests content; older versions keep flat keys
            content = article.get("content", article)
            title   = content.get("title", "") + " " + content.get("summary", "")
            words   = set(title.lower().split())
            pos = len(words & _POS_WORDS)
            neg = len(words & _NEG_WORDS)
            if pos + neg > 0:
                scores.append((pos - neg) / (pos + neg))
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def apply_sentiment_correction(pred_df: pd.DataFrame, cutoff_close: float,
                                sentiment: float, scale: float = 0.15) -> pd.DataFrame:
    """
    Small directional nudge from recent news. scale is kept low (0.15) because
    keyword sentiment is noisy and Kronos already captures price momentum.
    Ignored if |sentiment| < 0.1.
    """
    if abs(sentiment) < 0.1:
        return pred_df
    pred_df = pred_df.copy()
    n    = len(pred_df)
    ramp = np.linspace(0, 1, n)
    correction = cutoff_close * sentiment * scale * ramp

    for col in ["open", "high", "low", "close"]:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col] + correction
        if f"{col}_lo" in pred_df.columns:
            pred_df[f"{col}_lo"] = pred_df[f"{col}_lo"] + correction
        if f"{col}_hi" in pred_df.columns:
            pred_df[f"{col}_hi"] = pred_df[f"{col}_hi"] + correction
    return pred_df


# ── 7. Fed meeting calendar ───────────────────────────────────────────────────

# FOMC meeting dates — released annually at federalreserve.gov
_FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]


def check_fed_meetings_in_window(pred_start: pd.Timestamp,
                                  pred_end: pd.Timestamp) -> tuple:
    """Returns (has_meeting: bool, dates: list[Timestamp])."""
    dates = [pd.Timestamp(d) for d in _FOMC_DATES]
    hits  = [d for d in dates if pred_start <= d <= pred_end]
    return bool(hits), hits


# ── 8. Rolling 5-day predictions ─────────────────────────────────────────────

def predict_rolling(predictor, df_full: pd.DataFrame, step: int = 5,
                    total_days: int = 25, lookback: int = 400,
                    T: float = 0.7, top_p: float = 0.9,
                    sample_count: int = 10, use_confidence: bool = True,
                    verbose: bool = False) -> pd.DataFrame:
    """
    Chain `total_days // step` short forecasts of length `step`.

    Each step uses the most recent `lookback` bars of *known* data, so error
    doesn't compound across the full horizon the way a single long forecast does.

    Returns a DataFrame with the same columns as predict_with_confidence.
    """
    cols = ["open", "high", "low", "close", "volume", "amount"]
    pieces = []

    df_known = df_full[cols].copy()
    ts_known = df_full["timestamps"].copy()

    n_steps = total_days // step
    for i in range(n_steps):
        x_df = df_known.tail(lookback).reset_index(drop=True)
        x_ts = ts_known.tail(lookback).reset_index(drop=True)

        last_ts = pd.Timestamp(x_ts.iloc[-1])
        y_ts    = pd.Series(pd.bdate_range(start=last_ts, periods=step + 1)[1:])

        if use_confidence:
            chunk = predictor.predict_with_confidence(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=step, T=T, top_p=top_p,
                sample_count=sample_count, verbose=verbose,
            )
        else:
            chunk = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=step, T=T, top_p=top_p,
                sample_count=sample_count, verbose=verbose,
            )

        pieces.append(chunk)

        # Append median predictions as "known" data for the next step
        new_rows = pd.DataFrame({
            "timestamps": y_ts.values,
            **{c: chunk[c].values for c in cols},
        })
        df_known = pd.concat(
            [df_known, new_rows[cols]], ignore_index=True
        )
        ts_known = pd.concat(
            [ts_known, new_rows["timestamps"]], ignore_index=True
        )

    return pd.concat(pieces)
