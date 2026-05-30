"""
Kronos Weekly Iron Condor Report Generator
==========================================
Run:    python3 weekly_report.py
Output: Kronos_Weekly_Report_YYYY-MM-DD.pdf
"""

import io, sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
from scipy.stats import norm
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.utils import ImageReader

warnings.filterwarnings("ignore")

# ── Settings ───────────────────────────────────────────────────────────────────
ACCOUNT    = 37_000
RISK_PCT   = 0.07
RISK_FREE  = 0.05
T_DAYS     = 5
PRED_DAYS  = 3
VOL_WIN    = 20
Z          = 1.28
MIN_CREDIT = 300

# ── Design tokens ──────────────────────────────────────────────────────────────
# Primary palette
INK        = HexColor("#0d1b2a")   # near-black for text
NAVY       = HexColor("#0a2540")   # dark navy for headers
BLUE       = HexColor("#1a56db")   # accent blue
BLUE_LIGHT = HexColor("#e8f0fe")   # light blue tint
GREEN      = HexColor("#0d6b3a")   # dark green
GREEN_LIGHT= HexColor("#d1fae5")
RED        = HexColor("#b91c1c")
RED_LIGHT  = HexColor("#fee2e2")
AMBER      = HexColor("#b45309")
AMBER_LIGHT= HexColor("#fef3c7")
SILVER     = HexColor("#f1f5f9")   # page background tint
RULE       = HexColor("#e2e8f0")   # divider lines
MUTED      = HexColor("#64748b")   # secondary text
WHITE      = colors.white
BLACK      = colors.black

# Page geometry
W, H   = letter
MAR    = 0.45 * inch   # margin
CWIDTH = W - 2 * MAR   # content width


# ── Helpers ────────────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, kind="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if kind == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def price_condor(S, lo, hi, T, r, iv, width):
    credit = (
        bs_price(S, hi, T, r, iv, "call") - bs_price(S, hi + width, T, r, iv, "call") +
        bs_price(S, lo, T, r, iv, "put")  - bs_price(S, lo - width, T, r, iv, "put")
    )
    return round(credit, 3), round(width - credit, 3)


def load_ticker(sym):
    try:
        raw = yf.download(sym, period="90d", interval="1d",
                          auto_adjust=True, progress=False).dropna()
        if len(raw) < 25:
            return None
        return pd.DataFrame({
            "date":  pd.to_datetime(raw.index.tz_localize(None) if raw.index.tz else raw.index),
            "close": raw["Close"].values.flatten(),
            "high":  raw["High"].values.flatten(),
            "low":   raw["Low"].values.flatten(),
        }).reset_index(drop=True)
    except Exception:
        return None


def next_monday():
    t = date.today()
    d = (7 - t.weekday()) % 7
    return t + timedelta(days=d if d else 7)


def next_friday():
    t = date.today()
    d = 4 - t.weekday()
    return t + timedelta(days=d if d > 0 else d + 7)


def spread_width(S):
    if   S <  20:  return 0.5
    elif S <  50:  return 1.0
    elif S < 100:  return 2.5
    elif S < 200:  return 4.0
    else:          return 5.0


# ── Chart ──────────────────────────────────────────────────────────────────────
def make_chart(sym, df, sell_put, sell_call, price):
    hist = df.tail(60).copy()
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    # Candlestick-style via filled area between high/low
    ax.fill_between(hist["date"], hist["low"], hist["high"],
                    alpha=0.08, color="#1a56db")
    ax.plot(hist["date"], hist["close"],
            color="#1a56db", linewidth=2.0, zorder=4, solid_capstyle="round")

    # Safe zone band
    ax.axhspan(sell_put, sell_call, alpha=0.12, color="#0d6b3a", zorder=1)
    ax.axhline(sell_put,  color="#b91c1c", linestyle="--",
               linewidth=1.1, zorder=3, label=f"${sell_put:.2f}")
    ax.axhline(sell_call, color="#b91c1c", linestyle="--",
               linewidth=1.1, zorder=3, label=f"${sell_call:.2f}")

    # Current price dot
    ax.scatter([hist["date"].iloc[-1]], [price],
               color="#0a2540", s=40, zorder=5)

    # Labels on right edge
    xlim = ax.get_xlim()
    x_right = hist["date"].iloc[-1]
    ax.annotate(f"  ${sell_call:.2f}", xy=(x_right, sell_call),
                fontsize=6.5, color="#b91c1c", va="center")
    ax.annotate(f"  ${sell_put:.2f}",  xy=(x_right, sell_put),
                fontsize=6.5, color="#b91c1c", va="center")

    ax.set_xlim(left=hist["date"].iloc[0])
    ax.set_title(f"{sym}  ·  60-day price history  ·  red dashes = short strikes  ·  green = safe zone",
                 fontsize=7, color="#64748b", pad=4, loc="left")
    ax.grid(True, alpha=0.18, color="#cbd5e1", linewidth=0.7)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#e2e8f0")
    ax.tick_params(labelsize=6.5, color="#e2e8f0")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=20)
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Drawing primitives ─────────────────────────────────────────────────────────
def rule(c, y, x0=None, x1=None, color=None, lw=0.5):
    c.setStrokeColor(color or RULE)
    c.setLineWidth(lw)
    c.line(x0 or MAR, y, x1 or W - MAR, y)


def label_value(c, x, y, label, value, label_color=None, value_color=None,
                label_size=7, value_size=8.5, gap=0.0):
    c.setFillColor(label_color or MUTED)
    c.setFont("Helvetica", label_size)
    c.drawString(x, y + value_size * 0.95 + 1, label.upper())
    c.setFillColor(value_color or INK)
    c.setFont("Helvetica-Bold", value_size)
    c.drawString(x + gap, y, value)


def badge(c, x, y, text, bg, fg=WHITE, w=None, h=0.22 * inch, r=3, size=7.5):
    tw = c.stringWidth(text, "Helvetica-Bold", size)
    bw = w or tw + 14
    c.setFillColor(bg)
    c.roundRect(x, y, bw, h, r, fill=1, stroke=0)
    c.setFillColor(fg)
    c.setFont("Helvetica-Bold", size)
    c.drawCentredString(x + bw / 2, y + h * 0.27, text)
    return bw


def page_number(c, n):
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7)
    c.drawRightString(W - MAR, 0.22 * inch, f"Page {n}")


# ── Page 1: Cover ──────────────────────────────────────────────────────────────
def page_cover(c, today, monday, exp_date, vix, spy_mom, regime,
               n_tickers, n_viable, top5):

    # ── Full-bleed hero ───────────────────────────────────────────────────────
    hero_h = 3.0 * inch
    c.setFillColor(NAVY)
    c.rect(0, H - hero_h, W, hero_h, fill=1, stroke=0)

    # Subtle diagonal stripe texture
    c.setStrokeColor(Color(1, 1, 1, alpha=0.03))
    c.setLineWidth(18)
    for i in range(-5, 25):
        c.line(i * 36, H - hero_h, i * 36 + hero_h, H)

    # Thin blue accent bar at top
    c.setFillColor(BLUE)
    c.rect(0, H - 0.06 * inch, W, 0.06 * inch, fill=1, stroke=0)

    # KRONOS wordmark
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 36)
    c.drawString(MAR, H - 1.05 * inch, "KRONOS")
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 36)
    c.drawString(MAR + c.stringWidth("KRONOS", "Helvetica-Bold", 36) + 6,
                 H - 1.05 * inch, "·")

    # Subtitle
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 11)
    c.drawString(MAR, H - 1.38 * inch, "WEEKLY IRON CONDOR REPORT")

    # Date strip
    c.setFillColor(Color(1, 1, 1, alpha=0.08))
    c.rect(0, H - hero_h, W, 0.68 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MAR, H - hero_h + 0.40 * inch,
                 f"Week of {monday.strftime('%B %d, %Y')}")
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 9)
    c.drawString(MAR, H - hero_h + 0.20 * inch,
                 f"Options expiry {exp_date.strftime('%A, %B %d')}  ·  "
                 f"Generated {today.strftime('%b %d, %Y')}")

    # Generated-by tag (right side)
    c.setFillColor(Color(1, 1, 1, alpha=0.15))
    tag_w = 1.6 * inch
    c.roundRect(W - MAR - tag_w, H - hero_h + 0.16 * inch,
                tag_w, 0.38 * inch, 4, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawCentredString(W - MAR - tag_w / 2, H - hero_h + 0.35 * inch, "POWERED BY KRONOS AI")
    c.setFont("Helvetica", 6.5)
    c.drawCentredString(W - MAR - tag_w / 2, H - hero_h + 0.20 * inch,
                        "Black-Scholes · Realized Vol")

    # ── Market snapshot cards ─────────────────────────────────────────────────
    y_cards = H - hero_h - 1.25 * inch
    card_h  = 0.95 * inch
    card_w  = (CWIDTH - 3 * 0.12 * inch) / 4
    gaps    = [MAR + i * (card_w + 0.12 * inch) for i in range(4)]

    vix_bg  = RED   if vix > 30 else AMBER   if vix > 22 else GREEN
    vix_lbl = "HIGH FEAR" if vix > 30 else "ELEVATED" if vix > 22 else "CALM"
    spy_bg  = GREEN if spy_mom > 0.02 else RED if spy_mom < -0.02 else AMBER

    cards = [
        (f"VIX  {vix:.1f}", vix_lbl, vix_bg),
        (f"SPY {spy_mom:+.1%}", f"Market {regime}", spy_bg),
        (f"{n_viable} trades", f"of {n_tickers} analyzed", NAVY),
        (f"${ACCOUNT:,.0f}", f"@ {int(RISK_PCT*100)}% risk / trade", BLUE),
    ]

    for (val, sub, bg), x in zip(cards, gaps):
        # Shadow
        c.setFillColor(Color(0, 0, 0, alpha=0.07))
        c.roundRect(x + 2, y_cards - 2, card_w, card_h, 8, fill=1, stroke=0)
        # Card
        c.setFillColor(bg)
        c.roundRect(x, y_cards, card_w, card_h, 8, fill=1, stroke=0)
        # Top accent line
        c.setFillColor(Color(1, 1, 1, alpha=0.2))
        c.roundRect(x, y_cards + card_h - 0.06 * inch, card_w, 0.06 * inch, 8,
                    fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(x + card_w / 2, y_cards + card_h * 0.50, val)
        c.setFont("Helvetica", 8)
        c.setFillColor(Color(1, 1, 1, alpha=0.80))
        c.drawCentredString(x + card_w / 2, y_cards + card_h * 0.24, sub.upper())

    # ── Top picks this week ───────────────────────────────────────────────────
    y_picks = y_cards - 0.7 * inch
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MAR, y_picks, "TOP TRADES THIS WEEK")
    rule(c, y_picks - 0.08 * inch, lw=0.4)

    x_pill = MAR
    pill_h = 0.34 * inch
    pill_colors = [GREEN, GREEN, HexColor("#15803d"),
                   HexColor("#15803d"), BLUE]
    for i, sym in enumerate(top5[:5]):
        bg  = pill_colors[i]
        txt = f"  #{i+1}  {sym}  "
        pw  = c.stringWidth(txt, "Helvetica-Bold", 10) + 10
        c.setFillColor(Color(0, 0, 0, alpha=0.06))
        c.roundRect(x_pill + 1, y_picks - 0.60 * inch - 1, pw, pill_h, 5, fill=1, stroke=0)
        c.setFillColor(bg)
        c.roundRect(x_pill, y_picks - 0.60 * inch, pw, pill_h, 5, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(x_pill + pw / 2, y_picks - 0.42 * inch, txt.strip())
        x_pill += pw + 0.1 * inch

    # ── Contents card ─────────────────────────────────────────────────────────
    y_cont = y_picks - 1.22 * inch
    c.setFillColor(SILVER)
    c.roundRect(MAR, y_cont - 1.35 * inch, CWIDTH, 1.4 * inch, 6, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(MAR + 0.2 * inch, y_cont - 0.02 * inch, "WHAT'S IN THIS REPORT")
    rule(c, y_cont - 0.22 * inch, MAR + 0.2 * inch, W - MAR - 0.2 * inch, RULE)

    contents = [
        ("Page 2", "Full rankings — all analyzed stocks sorted highest to lowest conviction"),
        ("Pages 3 – 7", "One page per top trade: 60-day chart, exact Webull order ticket, win condition"),
        ("Last page", "Trade management rules + historical backtest performance reference"),
    ]
    for i, (pg, desc) in enumerate(contents):
        ry = y_cont - 0.52 * inch - i * 0.30 * inch
        c.setFillColor(BLUE)
        c.roundRect(MAR + 0.2 * inch, ry + 0.01 * inch,
                    0.72 * inch, 0.19 * inch, 3, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(MAR + 0.56 * inch, ry + 0.04 * inch, pg)
        c.setFillColor(INK)
        c.setFont("Helvetica", 8.5)
        c.drawString(MAR + 1.06 * inch, ry + 0.04 * inch, desc)

    # Footer
    c.setFillColor(RULE)
    c.rect(0, 0, W, 0.38 * inch, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 7)
    c.drawCentredString(W / 2, 0.14 * inch,
        "This report is generated from historical price data and mathematical models. "
        "Paper trade before risking real capital. Past performance does not guarantee future results.")
    page_number(c, 1)


# ── Page 2: Rankings ───────────────────────────────────────────────────────────
def page_rankings(c, recs, exp_date, page_n):
    # Header
    c.setFillColor(NAVY)
    c.rect(0, H - 0.72 * inch, W, 0.72 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MAR, H - 0.42 * inch, "ALL STOCKS RANKED")
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 8.5)
    c.drawRightString(W - MAR, H - 0.42 * inch,
                      f"{len(recs)} viable trades  ·  Expiry {exp_date.strftime('%b %d')}")

    # Column definitions
    headers = ["#", "Ticker", "Current Price", "Safe Zone  (stay between these)", "Spread", "Credit / sh", "Max Loss / sh", "B/E %", "Conviction"]
    cw      = [i * inch for i in [0.25, 0.55, 0.75, 2.05, 0.52, 0.72, 0.80, 0.52, 1.0]]
    total_w = sum(cw)
    x0      = (W - total_w) / 2
    rh      = 0.225 * inch
    y       = H - 1.02 * inch

    # Table header row
    c.setFillColor(HexColor("#f8fafc"))
    c.rect(x0, y - 0.02 * inch, total_w, rh, fill=1, stroke=0)
    rule(c, y - 0.02 * inch, x0, x0 + total_w, RULE)
    rule(c, y + rh - 0.02 * inch, x0, x0 + total_w, NAVY, lw=1.5)

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 7)
    x = x0
    for h, w in zip(headers, cw):
        c.drawCentredString(x + w / 2, y + 0.062 * inch, h.upper())
        x += w
    y -= rh

    labels    = ["#1 Highest","#2 High","#3 High","#4 Solid","#5 Solid",
                 "Moderate","Moderate","Lower","Lower","Lowest"] + ["—"] * 30
    badge_bgs = [GREEN, GREEN, HexColor("#16a34a"), HexColor("#16a34a"),
                 HexColor("#16a34a"), AMBER, AMBER, MUTED, MUTED, MUTED]

    for i, r in enumerate(recs):
        if y < 0.45 * inch:
            break
        even = i % 2 == 0
        row_bg = HexColor("#f8fafc") if even else WHITE
        c.setFillColor(row_bg)
        c.rect(x0, y - 0.02 * inch, total_w, rh, fill=1, stroke=0)

        # Left accent bar for top 5
        if i < 5:
            accent = badge_bgs[i]
            c.setFillColor(accent)
            c.rect(x0, y - 0.02 * inch, 0.04 * inch, rh, fill=1, stroke=0)

        cells = [
            str(i + 1), r["ticker"], f"${r['price']:.2f}",
            f"${r['sell_put']:.2f}  –  ${r['sell_call']:.2f}",
            f"${r['width']:.1f}",
            f"${r['credit']:.2f}", f"${r['max_loss']:.2f}",
            f"{r['be_win_rate']:.0f}%",
            labels[min(i, len(labels) - 1)],
        ]

        x = x0
        for j, (cell, w) in enumerate(zip(cells, cw)):
            if j == 8:
                bg2 = badge_bgs[min(i, len(badge_bgs) - 1)]
                bw2 = w * 0.88
                c.setFillColor(bg2)
                c.roundRect(x + (w - bw2) / 2, y + 0.025 * inch,
                            bw2, 0.16 * inch, 3, fill=1, stroke=0)
                c.setFillColor(WHITE)
                c.setFont("Helvetica-Bold", 6.5)
                c.drawCentredString(x + w / 2, y + 0.05 * inch, cell)
            elif j == 1:  # ticker bold
                c.setFillColor(NAVY if i < 5 else INK)
                c.setFont("Helvetica-Bold", 8.5)
                c.drawCentredString(x + w / 2, y + 0.062 * inch, cell)
            else:
                c.setFillColor(INK)
                c.setFont("Helvetica-Bold" if i < 5 else "Helvetica", 7.5)
                c.drawCentredString(x + w / 2, y + 0.062 * inch, cell)
            x += w

        rule(c, y - 0.02 * inch, x0, x0 + total_w, RULE, 0.3)
        y -= rh

    # Footer
    rule(c, 0.42 * inch, lw=0.5)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7)
    c.drawString(MAR, 0.26 * inch,
                 "B/E % = minimum win rate to break even long-term  ·  "
                 "Credit & Max Loss are per share  ·  "
                 "Safe Zone = where stock must close by expiry date")
    page_number(c, page_n)


# ── Pages 3-7: Trade pages ─────────────────────────────────────────────────────
def page_trade(c, rank, r, df, entry_date, day_name, exp_date, conv_label, page_n):
    conv_bg = (GREEN if rank <= 2 else HexColor("#16a34a") if rank <= 5 else BLUE)

    # ── Top header bar ────────────────────────────────────────────────────────
    c.setFillColor(NAVY)
    c.rect(0, H - 0.90 * inch, W, 0.90 * inch, fill=1, stroke=0)

    # Rank circle
    c.setFillColor(conv_bg)
    c.circle(MAR + 0.28 * inch, H - 0.44 * inch, 0.26 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(MAR + 0.28 * inch, H - 0.50 * inch, str(rank))

    # Ticker + subtitle
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(MAR + 0.68 * inch, H - 0.48 * inch, r["ticker"])
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 8.5)
    c.drawString(MAR + 0.68 * inch, H - 0.70 * inch,
                 f"Enter {day_name} {entry_date.strftime('%b %d')}   ·   "
                 f"Expire {exp_date.strftime('%a %b %d')}   ·   "
                 f"Last price  ${r['price']:.2f}")

    # Conviction badge (right)
    bw = badge(c, W - MAR - 1.55 * inch, H - 0.66 * inch,
               conv_label.upper() + " CONVICTION",
               conv_bg, w=1.55 * inch, h=0.26 * inch, size=7.5)

    # ── Two-column body ───────────────────────────────────────────────────────
    body_top = H - 1.02 * inch
    col_gap  = 0.18 * inch
    left_w   = CWIDTH * 0.56
    right_w  = CWIDTH - left_w - col_gap
    left_x   = MAR
    right_x  = MAR + left_w + col_gap

    # ── LEFT: chart + win condition ───────────────────────────────────────────
    chart_h = 2.55 * inch
    chart_buf = make_chart(r["ticker"], df, r["sell_put"], r["sell_call"], r["price"])
    img = ImageReader(chart_buf)
    c.drawImage(img, left_x, body_top - chart_h,
                width=left_w, height=chart_h, preserveAspectRatio=True)

    # Win condition card
    wc_y = body_top - chart_h - 0.12 * inch - 0.72 * inch
    c.setFillColor(GREEN_LIGHT)
    c.roundRect(left_x, wc_y, left_w, 0.68 * inch, 5, fill=1, stroke=0)
    c.setFillColor(GREEN)
    c.roundRect(left_x, wc_y + 0.68 * inch - 0.06 * inch,
                left_w, 0.06 * inch, 5, fill=1, stroke=0)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 7.5)
    c.drawCentredString(left_x + left_w / 2, wc_y + 0.49 * inch, "WIN CONDITION")
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(left_x + left_w / 2, wc_y + 0.26 * inch,
                        f"{r['ticker']} closes between")
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(left_x + left_w / 2, wc_y + 0.07 * inch,
                        f"${r['sell_put']:.2f}  and  ${r['sell_call']:.2f}  "
                        f"by {exp_date.strftime('%b %d')}")

    # P&L summary row
    pnl_y   = wc_y - 0.12 * inch - 0.72 * inch
    pnl_bw  = (left_w - 0.08 * inch) / 3
    pnl_items = [
        ("COLLECT IF WIN",  f"${r['actual_cred']:.0f}", GREEN, GREEN_LIGHT),
        ("MAX LOSS",        f"${r['actual_risk']:.0f}", RED,   RED_LIGHT),
        ("BREAKEVEN",       f"{r['be_win_rate']:.0f}% wins", AMBER, AMBER_LIGHT),
    ]
    for i, (lbl, val, fg, bg) in enumerate(pnl_items):
        bx = left_x + i * (pnl_bw + 0.04 * inch)
        c.setFillColor(bg)
        c.roundRect(bx, pnl_y, pnl_bw, 0.68 * inch, 5, fill=1, stroke=0)
        c.setFillColor(fg)
        c.setFont("Helvetica-Bold", 6.5)
        c.drawCentredString(bx + pnl_bw / 2, pnl_y + 0.50 * inch, lbl)
        c.setFillColor(fg)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(bx + pnl_bw / 2, pnl_y + 0.22 * inch, val)

    # ── RIGHT: order ticket ───────────────────────────────────────────────────
    ticket_h = body_top - 0.38 * inch - (H - 0.38 * inch - body_top + 0.38 * inch)
    ticket_h = 4.95 * inch

    c.setFillColor(SILVER)
    c.roundRect(right_x, body_top - ticket_h, right_w, ticket_h, 6, fill=1, stroke=0)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.5)
    c.roundRect(right_x, body_top - ticket_h, right_w, ticket_h, 6, fill=0, stroke=1)

    # Ticket header
    c.setFillColor(NAVY)
    # Top rounded section
    c.roundRect(right_x, body_top - 0.38 * inch, right_w, 0.38 * inch,
                6, fill=1, stroke=0)
    c.rect(right_x, body_top - 0.38 * inch, right_w, 0.2 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawCentredString(right_x + right_w / 2, body_top - 0.22 * inch,
                        "WEBULL ORDER TICKET")

    pad  = 0.14 * inch
    ty   = body_top - 0.55 * inch
    iw   = right_w - 2 * pad

    def tr(label, val, val_color=None, divider=False):
        nonlocal ty
        if divider:
            ty -= 0.06 * inch
            c.setStrokeColor(RULE)
            c.setLineWidth(0.4)
            c.line(right_x + pad, ty, right_x + right_w - pad, ty)
            ty -= 0.08 * inch
            return
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 6.5)
        c.drawString(right_x + pad, ty + 0.105 * inch, label.upper())
        c.setFillColor(val_color or INK)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawRightString(right_x + right_w - pad, ty + 0.09 * inch, val)
        ty -= 0.215 * inch

    tr("Strategy",      "Iron Condor")
    tr("Expiration",    exp_date.strftime("%b %d, %Y") + "  (weekly)")
    tr("Side",          "SELL ← must be Sell",  RED)
    tr("", "", divider=True)
    tr("Leg 1  BUY put",  f"${r['buy_put']:.2f}",  GREEN)
    tr("Leg 2  SELL put", f"${r['sell_put']:.2f}  ← short strike",  RED)
    tr("Leg 3  SELL call",f"${r['sell_call']:.2f}  ← short strike", RED)
    tr("Leg 4  BUY call", f"${r['buy_call']:.2f}", GREEN)
    tr("", "", divider=True)
    tr("Contracts",     str(r["n_contracts"]))
    tr("Order Type",    "LIMIT")

    lim = round(r["credit"] * 1.05, 2)
    flr = round(r["credit"] * 0.85, 2)
    tr("Limit Price",   f"{lim:.2f}")
    tr("Time-in-Force", "Day")

    # Fill guidance box
    ty -= 0.05 * inch
    fg_h = 0.52 * inch
    c.setFillColor(BLUE_LIGHT)
    c.roundRect(right_x + pad, ty - fg_h, iw, fg_h, 4, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(right_x + pad + 0.08 * inch, ty - 0.12 * inch, "FILL GUIDANCE")
    c.setFillColor(INK)
    c.setFont("Helvetica", 7)
    c.drawString(right_x + pad + 0.08 * inch, ty - 0.28 * inch,
                 f"Start at {lim:.2f}. No fill after 10 min → lower to {round(r['credit']*0.95,2):.2f}.")
    c.drawString(right_x + pad + 0.08 * inch, ty - 0.42 * inch,
                 f"Floor: {flr:.2f}. Don't go lower. Enter 9:30–10:30am only.")

    # Footer
    rule(c, 0.42 * inch, lw=0.5)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7)
    c.drawString(MAR, 0.26 * inch,
                 f"Close early if {r['ticker']} comes within $2.00 of ${r['sell_put']:.2f} "
                 f"or ${r['sell_call']:.2f} before Thursday.  "
                 f"Target 50% profit early exit.")
    page_number(c, page_n)


# ── Last page: Rules ───────────────────────────────────────────────────────────
def page_rules(c, recs, monday, exp_date, page_n):
    c.setFillColor(NAVY)
    c.rect(0, H - 0.72 * inch, W, 0.72 * inch, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MAR, H - 0.43 * inch, "TRADE MANAGEMENT  +  STRATEGY REFERENCE")
    c.setFillColor(HexColor("#94a3b8"))
    c.setFont("Helvetica", 8.5)
    c.drawRightString(W - MAR, H - 0.43 * inch,
                      f"Week of {monday.strftime('%b %d, %Y')}")

    y = H - 1.0 * inch

    # Section: Rules
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MAR, y, "DURING-THE-WEEK RULES  —  apply every trade, every week")
    y -= 0.08 * inch
    rule(c, y, lw=1.2, color=NAVY)
    y -= 0.18 * inch

    rules = [
        (GREEN,  "50% Profit Rule",
         "When the condor loses 50% of its value, BUY IT BACK. "
         "You've captured half your max gain — lock it in, free the capital."),
        (RED,    "50% Loss Rule",
         "If closing costs 2× your original credit, CLOSE NOW. "
         "Never hold hoping for a reversal. Cut and protect your account."),
        (AMBER,  "VIX Spikes > 25",
         "Close ALL open positions immediately, accept whatever P&L. "
         "High VIX means erratic moves that destroy condor positions."),
        (BLUE,   "Thursday Check",
         "Check each position vs its short strikes. Stock within $2? "
         "Close it before Friday. Never gamble on expiry-day pinning."),
        (NAVY,   "Friday — Let Expire",
         "Still open and stock is comfortably inside the band? "
         "Do nothing. Let it expire worthless and collect the full credit."),
        (MUTED,  "Fill Trouble",
         "Drop limit $0.05 every 10 min at open. "
         "Still unfilled after 30 min? Skip for the week. Never chase a fill."),
    ]

    for i, (color, title, body) in enumerate(rules):
        rh = 0.36 * inch
        c.setFillColor(SILVER)
        c.roundRect(MAR, y - rh + 0.04 * inch, CWIDTH, rh, 4, fill=1, stroke=0)
        c.setFillColor(color)
        c.roundRect(MAR, y - rh + 0.04 * inch, 0.05 * inch, rh, 4, fill=1, stroke=0)

        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(MAR + 0.16 * inch, y - 0.09 * inch, title + ":")
        offset = c.stringWidth(title + ":", "Helvetica-Bold", 8.5) + MAR + 0.16 * inch + 4
        c.setFillColor(INK)
        c.setFont("Helvetica", 8.5)
        c.drawString(offset, y - 0.09 * inch, body)
        y -= rh + 0.06 * inch

    # Section: Backtest reference
    y -= 0.15 * inch
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MAR, y, "HISTORICAL BACKTEST  —  realistic $4–$5 spreads  ·  7% risk/trade  ·  compound sizing")
    y -= 0.08 * inch
    rule(c, y, lw=1.2, color=NAVY)
    y -= 0.06 * inch

    # Table
    bt_headers = ["Scenario", "Trades", "Win Rate", "$37k → End", "Net Return", "Max Drawdown"]
    bt_rows    = [
        ["Calm VIX (<22)  —  10 calm weeks", "100", "84%", "$69,585", "+88%  (+75% net of fees)", "7.9%"],
        ["August 2024  —  VIX spiked to 65", "40",  "77.5%", "$44,410", "+20%  (+15% net of fees)", "16.4%"],
        ["Minimum win rate to break even",   "—",   "64%",  "—",       "—",                        "—"],
    ]
    bt_cw  = [i * inch for i in [2.7, 0.55, 0.70, 0.82, 1.62, 0.87]]
    total  = sum(bt_cw)
    bx0    = (W - total) / 2
    brh    = 0.26 * inch
    by     = y - brh

    c.setFillColor(NAVY)
    c.rect(bx0, by, total, brh, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 7.5)
    bx = bx0
    for h, w in zip(bt_headers, bt_cw):
        c.drawCentredString(bx + w / 2, by + 0.082 * inch, h)
        bx += w

    for ri, row in enumerate(bt_rows):
        by -= brh
        bg = GREEN_LIGHT if ri == 0 else (SILVER if ri % 2 == 0 else WHITE)
        c.setFillColor(bg)
        c.rect(bx0, by, total, brh, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica", 7.5)
        bx = bx0
        for cell, w in zip(row, bt_cw):
            c.drawCentredString(bx + w / 2, by + 0.082 * inch, cell)
            bx += w
        rule(c, by, bx0, bx0 + total, RULE, 0.3)

    rule(c, 0.42 * inch, lw=0.5)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 7)
    c.drawCentredString(W / 2, 0.26 * inch,
        "Kronos Weekly Report  ·  Generated by Kronos AI price model + Black-Scholes options pricing  "
        "·  Not financial advice")
    page_number(c, page_n)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    today    = date.today()
    monday   = next_monday()
    exp_date = next_friday()

    ticker_file = "../approved_tickers.txt"
    try:
        with open(ticker_file) as f:
            tickers = [ln.strip() for ln in f
                       if ln.strip() and not ln.strip().startswith("#")]
    except FileNotFoundError:
        print(f"ERROR: approved_tickers.txt not found at {ticker_file}")
        return

    print(f"\n{'='*55}")
    print(f"  KRONOS WEEKLY REPORT  —  {today}")
    print(f"  Week of {monday.strftime('%B %d')}  |  Expiry {exp_date.strftime('%B %d')}")
    print(f"  {len(tickers)} tickers loaded")
    print(f"{'='*55}")

    print("Fetching VIX and SPY …")
    vix_raw = yf.download("^VIX", period="30d", interval="1d",
                          auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    spy_raw = yf.download("SPY", period="40d", interval="1d",
                          auto_adjust=True, progress=False)["Close"].squeeze().dropna()
    vix     = float(vix_raw.iloc[-1])
    spy_mom = float(spy_raw.iloc[-1] / spy_raw.iloc[-21] - 1)
    regime  = "BULLISH" if spy_mom > 0.02 else "BEARISH" if spy_mom < -0.02 else "NEUTRAL"
    print(f"VIX={vix:.1f}  SPY={spy_mom:+.1%} ({regime})\n")

    recs, stock_data, skipped = [], {}, []

    for sym in tickers:
        print(f"  {sym:<6}", end=" ", flush=True)
        df = load_ticker(sym)
        if df is None:
            print("✗  no data"); skipped.append(sym); continue

        S = float(df["close"].iloc[-1])
        if S < 10:
            print(f"✗  ${S:.2f}"); skipped.append(sym); continue

        closes    = df["close"].values[-(VOL_WIN + 1):]
        rets      = np.diff(closes) / closes[:-1]
        daily_vol = np.std(rets)
        iv        = daily_vol * np.sqrt(252) * 1.15
        lo        = S * (1 - Z * daily_vol * np.sqrt(PRED_DAYS))
        hi        = S * (1 + Z * daily_vol * np.sqrt(PRED_DAYS))
        w         = spread_width(S)

        credit, max_loss = price_condor(S, lo, hi, T_DAYS / 252, RISK_FREE, iv, w)
        if credit < 0.05 or max_loss <= 0:
            print(f"✗  credit ${credit:.2f}"); skipped.append(sym); continue

        n_con = max(1, min(5, int((ACCOUNT * RISK_PCT) / (max_loss * 100))))
        a_cred = n_con * credit * 100
        if a_cred < MIN_CREDIT:
            print(f"✗  total credit ${a_cred:.0f} < ${MIN_CREDIT}")
            skipped.append(sym); continue

        a_risk      = n_con * max_loss * 100
        be_win_rate = max_loss / (max_loss + credit) * 100
        score       = (credit / S * 200) + ((hi - lo) / S / (daily_vol * np.sqrt(252)) * 100) - (be_win_rate / 100)

        stock_data[sym] = df
        recs.append(dict(
            ticker=sym, price=S, lo=lo, hi=hi, width=w,
            sell_put=round(lo/0.5)*0.5, buy_put=round(lo/0.5)*0.5 - w,
            sell_call=round(hi/0.5)*0.5, buy_call=round(hi/0.5)*0.5 + w,
            credit=credit, max_loss=max_loss,
            n_contracts=n_con, actual_risk=a_risk, actual_cred=a_cred,
            be_win_rate=be_win_rate, daily_vol=daily_vol * 100,
            iv=iv * 100, score=score,
        ))
        print(f"✓  ${S:.2f}  ${round(lo/0.5)*0.5:.2f}–${round(hi/0.5)*0.5:.2f}  "
              f"credit=${credit:.2f}  collect=${a_cred:.0f}  B/E={be_win_rate:.0f}%")

    if not recs:
        print("\nNo viable trades found."); return

    recs.sort(key=lambda x: x["score"], reverse=True)
    top5 = recs[:5]

    print(f"\n{'─'*55}")
    print(f"  {len(recs)} viable trades. Top 5: {[r['ticker'] for r in top5]}")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

    # Build PDF
    fname = f"Kronos_Weekly_Report_{today.strftime('%Y-%m-%d')}.pdf"
    print(f"\nBuilding {fname} …")
    c = rl_canvas.Canvas(fname, pagesize=letter)
    c.setTitle(f"Kronos Weekly Report — {monday.strftime('%B %d, %Y')}")
    c.setAuthor("Kronos AI")

    page_cover(c, today, monday, exp_date, vix, spy_mom, regime,
               len(tickers), len(recs), [r["ticker"] for r in top5])
    c.showPage()

    page_rankings(c, recs, exp_date, page_n=2)
    c.showPage()

    conv_labels    = ["Highest", "High", "High", "Solid", "Solid"]
    entry_schedule = [
        (monday,                     "Monday"),
        (monday + timedelta(days=1), "Tuesday"),
        (monday + timedelta(days=1), "Tuesday"),
        (monday + timedelta(days=2), "Wednesday"),
        (monday + timedelta(days=2), "Wednesday"),
    ]
    for i, r in enumerate(top5):
        entry_date, day_name = entry_schedule[i]
        page_trade(c, i + 1, r, stock_data[r["ticker"]],
                   entry_date, day_name, exp_date, conv_labels[i], page_n=i + 3)
        c.showPage()

    page_rules(c, recs, monday, exp_date, page_n=len(top5) + 3)
    c.showPage()

    c.save()
    print(f"✓  {len(top5) + 3} pages → examples/{fname}")
    print(f"   open {fname}\n")


if __name__ == "__main__":
    main()
