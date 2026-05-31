"""
Iron Condor Trade Monitor — Railway deployment
───────────────────────────────────────────────
Runs 24/7 on Railway. Checks prices every 5 minutes during market hours.
Emails you when a position gets close to a short strike.

RAILWAY SETUP
─────────────
Set these environment variables in your Railway dashboard:

  EMAIL_APP_PASSWORD   → your Google App Password (16 chars)
  EMAIL_TO             → taki.karras34@gmail.com  (already set as default)

  TRADE_1  → SPY,746.50,766.50,Jun 05
  TRADE_2  → QQQ,722.00,754.50,Jun 05
  TRADE_3  → ANET,145.00,174.00,Jun 05
  TRADE_4  → DASH,148.50,170.00,Jun 05
  TRADE_5  → NVDA,200.50,221.50,Jun 05

  Format:  TICKER,sell_put,sell_call,expiry_label

UPDATING EACH MONDAY (60 seconds)
───────────────────────────────────
1. Open Railway dashboard → your Kronos project → Variables
2. Update TRADE_1 through TRADE_5 with new strikes
3. Railway auto-restarts — done.
"""

import os, smtplib, time, warnings, sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date
from zoneinfo import ZoneInfo
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG FROM ENV VARS ──────────────────────────────────────────────────────
EMAIL_TO     = os.environ.get("EMAIL_TO", "taki.karras34@gmail.com")
EMAIL_FROM   = os.environ.get("EMAIL_FROM", "taki.karras34@gmail.com")
APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD", "")

DANGER_BUFFER   = float(os.environ.get("DANGER_BUFFER",   "2.0"))
CRITICAL_BUFFER = float(os.environ.get("CRITICAL_BUFFER", "1.0"))
CHECK_INTERVAL  = int(os.environ.get("CHECK_INTERVAL",    "300"))   # seconds
ALERT_COOLDOWN  = int(os.environ.get("ALERT_COOLDOWN",    "1800"))  # 30 min


def _load_trades() -> list[dict]:
    """
    Load trades from TRADE_1 … TRADE_5 env vars.
    Format: TICKER,sell_put,sell_call,expiry_label
    Falls back to this week's hardcoded values if no env vars set.
    """
    trades = []
    for i in range(1, 11):
        val = os.environ.get(f"TRADE_{i}")
        if not val:
            break
        parts = [p.strip() for p in val.split(",")]
        if len(parts) < 3:
            continue
        trades.append({
            "ticker":    parts[0].upper(),
            "sell_put":  float(parts[1]),
            "sell_call": float(parts[2]),
            "expiry":    parts[3] if len(parts) > 3 else "Friday",
        })

    # Fallback: hardcoded defaults (update here if not using Railway env vars)
    if not trades:
        trades = [
            {"ticker": "SPY",  "sell_put": 746.50, "sell_call": 766.50, "expiry": "Jun 05"},
            {"ticker": "QQQ",  "sell_put": 722.00, "sell_call": 754.50, "expiry": "Jun 05"},
            {"ticker": "ANET", "sell_put": 145.00, "sell_call": 174.00, "expiry": "Jun 05"},
            {"ticker": "DASH", "sell_put": 148.50, "sell_call": 170.00, "expiry": "Jun 05"},
            {"ticker": "NVDA", "sell_put": 200.50, "sell_call": 221.50, "expiry": "Jun 05"},
        ]
    return trades


# ── INTERNALS ─────────────────────────────────────────────────────────────────
EASTERN      = ZoneInfo("America/New_York")
MARKET_OPEN  = (9, 30)
MARKET_CLOSE = (16, 0)

_last_alert: dict[str, float] = {}


def is_market_open() -> bool:
    now = datetime.now(EASTERN)
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t < MARKET_CLOSE


def get_price(ticker: str) -> float | None:
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


def send_email(subject: str, body: str) -> bool:
    if not APP_PASSWORD:
        print(f"  [EMAIL SKIPPED — EMAIL_APP_PASSWORD not set]  {subject}")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_FROM, APP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        return True
    except Exception as e:
        print(f"  [EMAIL ERROR] {e}")
        return False


def check_trade(trade: dict, price: float) -> str | None:
    put_dist  = price - trade["sell_put"]
    call_dist = trade["sell_call"] - price
    min_dist  = min(put_dist, call_dist)
    if min_dist <= CRITICAL_BUFFER:
        return "CRITICAL"
    if min_dist <= DANGER_BUFFER:
        return "WARNING"
    return None


def build_snapshot(trades: list[dict], prices: dict) -> str:
    lines = ["\nAll positions right now:"]
    for t in trades:
        p = prices.get(t["ticker"])
        if not p:
            continue
        pd = p - t["sell_put"]
        cd = t["sell_call"] - p
        flag = " ⚠" if min(pd, cd) <= DANGER_BUFFER else ""
        lines.append(f"  {t['ticker']:<6} ${p:>7.2f}  safe: ${t['sell_put']:.2f}–${t['sell_call']:.2f}{flag}")
    return "\n".join(lines)


def build_alert(trade: dict, price: float, level: str, snapshot: str) -> tuple[str, str]:
    put_dist  = price - trade["sell_put"]
    call_dist = trade["sell_call"] - price
    min_dist  = min(put_dist, call_dist)
    side      = "PUT SIDE — price falling" if put_dist < call_dist else "CALL SIDE — price rising"
    emoji     = "🚨" if level == "CRITICAL" else "⚠️"
    action    = "CLOSE NOW — within $1 of strike." if level == "CRITICAL" else "Watch closely. Consider closing if it keeps moving."

    subject = f"{emoji} Kronos Alert: {trade['ticker']} {level} — ${price:.2f}"
    body = f"""
{emoji} IRON CONDOR ALERT — {trade['ticker']} {level}
{'='*50}

Time         : {datetime.now(EASTERN).strftime('%I:%M %p ET')}
Stock        : {trade['ticker']}
Current price: ${price:.2f}
Expiry       : {trade['expiry']}

Safe zone    : ${trade['sell_put']:.2f} – ${trade['sell_call']:.2f}
Threatened   : {side}
Distance to strike: ${min_dist:.2f}

ACTION: {action}

─────────────────────────────────────────────
To close on Webull:
  Find the {trade['ticker']} iron condor position
  Tap → Close Position → Market order
─────────────────────────────────────────────
{snapshot}
"""
    return subject, body


def morning_summary(trades: list[dict], prices: dict) -> None:
    lines = [
        f"Good morning! Here are your open iron condor positions as of 9:30am ET.\n",
        f"  {'Ticker':<6}  {'Price':>7}  {'Safe Zone':^24}  {'Put Dist':>9}  {'Call Dist':>10}  Status",
        f"  {'─'*6}  {'─'*7}  {'─'*24}  {'─'*9}  {'─'*10}  {'─'*8}",
    ]
    for t in trades:
        p = prices.get(t["ticker"])
        if p is None:
            lines.append(f"  {t['ticker']:<6}  price unavailable")
            continue
        pd = p - t["sell_put"]
        cd = t["sell_call"] - p
        status = "✅ Safe" if min(pd, cd) > DANGER_BUFFER else ("⚠️  Watch" if min(pd, cd) > CRITICAL_BUFFER else "🚨 CLOSE")
        lines.append(f"  {t['ticker']:<6}  ${p:>6.2f}  ${t['sell_put']:>6.2f} – ${t['sell_call']:<7.2f}  +${pd:>6.2f}   +${cd:>6.2f}   {status}")

    lines.append(f"\nAll expire: {trades[0]['expiry']} — check again Thursday at 3pm ET.")
    lines.append("Reply to this email anytime to run a new analysis.")
    send_email("📊 Kronos Morning Summary", "\n".join(lines))
    print("  Morning summary email sent.")


def send_test_email(trades: list[dict]) -> None:
    body = f"""
✅ Kronos Monitor is live and working!

This is a test email confirming your trade monitor is running on Railway.

Watching these positions (expiry {trades[0]['expiry']}):
"""
    for t in trades:
        body += f"  {t['ticker']:<6}  safe zone: ${t['sell_put']:.2f} – ${t['sell_call']:.2f}\n"

    body += f"""
You'll receive:
  📊 Morning summary every trading day at 9:30am ET
  ⚠️  Warning if any stock gets within $2 of a strike
  🚨 Critical alert if any stock gets within $1 — close immediately

To update next week's strikes, go to Railway → Variables and edit TRADE_1 through TRADE_5.
"""
    ok = send_email("✅ Kronos Monitor is live!", body)
    print(f"  Test email {'sent ✉' if ok else 'FAILED — check EMAIL_APP_PASSWORD'}")


def run():
    trades = _load_trades()

    print(f"\n{'='*60}")
    print(f"  KRONOS TRADE MONITOR")
    print(f"  Watching : {', '.join(t['ticker'] for t in trades)}")
    print(f"  Alerts to: {EMAIL_TO}")
    print(f"  Buffers  : warning ${DANGER_BUFFER:.0f}  /  critical ${CRITICAL_BUFFER:.0f}")
    print(f"  Interval : every {CHECK_INTERVAL//60} min during market hours")
    if not APP_PASSWORD:
        print(f"\n  ⚠  EMAIL_APP_PASSWORD not set — alerts will log here only.")
    print(f"{'='*60}\n")

    if os.environ.get("SEND_TEST_EMAIL", "").lower() == "true":
        print("  Sending test email…")
        send_test_email(trades)

    morning_sent_date = None

    while True:
        now_et = datetime.now(EASTERN)
        trades = _load_trades()   # reload each cycle so env var changes take effect

        if is_market_open():
            print(f"[{now_et.strftime('%H:%M ET')}] Checking {len(trades)} positions…")

            prices = {t["ticker"]: get_price(t["ticker"]) for t in trades}

            # Morning summary once per day
            today = date.today()
            if morning_sent_date != today and now_et.hour == 9 and now_et.minute >= 30:
                morning_summary(trades, prices)
                morning_sent_date = today

            snapshot = build_snapshot(trades, prices)

            for trade in trades:
                ticker = trade["ticker"]
                price  = prices.get(ticker)
                if price is None:
                    print(f"  {ticker}: price unavailable")
                    continue

                level    = check_trade(trade, price)
                put_d    = price - trade["sell_put"]
                call_d   = trade["sell_call"] - price
                status   = f"${price:.2f}  put +${put_d:.2f}  call +${call_d:.2f}"

                if level:
                    last      = _last_alert.get(ticker, 0)
                    cooldown_ok = (time.time() - last) > ALERT_COOLDOWN
                    if cooldown_ok or level == "CRITICAL":
                        subject, body = build_alert(trade, price, level, snapshot)
                        ok = send_email(subject, body)
                        _last_alert[ticker] = time.time()
                        icon = "🚨" if level == "CRITICAL" else "⚠️ "
                        print(f"  {icon} {ticker}: {status}  [{level}] {'✉ sent' if ok else '✉ failed'}")
                    else:
                        print(f"  ⏸  {ticker}: {status}  [{level}] — cooldown")
                else:
                    print(f"  ✅ {ticker}: {status}")

        else:
            next_open = "Monday 9:30am ET" if now_et.weekday() >= 4 else "tomorrow 9:30am ET"
            print(f"[{now_et.strftime('%H:%M ET %a')}] Market closed — next open: {next_open}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        sys.exit(0)
