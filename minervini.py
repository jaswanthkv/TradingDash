"""
minervini.py — Mark Minervini SEPA trend-template screener.

SEPA Criteria (all 10 must pass to qualify):
  1. Price > 150-day SMA
  2. Price > 200-day SMA
  3. 150-day SMA > 200-day SMA
  4. 200-day SMA trending up (slope positive over last 25 days)
  5. 50-day SMA > 150-day SMA
  6. 50-day SMA > 200-day SMA
  7. Price > 50-day SMA
  8. Price within 25% of 52-week high
  9. Price at least 30% above 52-week low
  10. RS Rating >= 70

Ranking: RS Rating — cross-sectional percentile rank of 12-month return (0–99)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import date

import strategy as st


BENCHMARK = st.BENCHMARK

CRITERIA_LABELS = {
    "c1": "Price > 150d SMA",
    "c2": "Price > 200d SMA",
    "c3": "150d SMA > 200d SMA",
    "c4": "200d SMA uptrend (25d)",
    "c5": "50d SMA > 150d SMA",
    "c6": "50d SMA > 200d SMA",
    "c7": "Price > 50d SMA",
    "c8": "Within 25% of 52w High",
    "c9": "30%+ above 52w Low",
    "c10": "RS Rating ≥ 70",
}
CRITERIA_SHORT = {
    "c1": "P>150", "c2": "P>200", "c3": "150>200", "c4": "200↑",
    "c5": "50>150", "c6": "50>200", "c7": "P>50", "c8": "<25%H", "c9": ">30%L",
    "c10": "RS≥70",
}
RS_MIN = 70   # Minervini Trend Template: RS Rating must be ≥ 70


# ── SEPA signal computation ───────────────────────────────────────────────────

def compute_sepa(close: pd.DataFrame,
                 high: pd.DataFrame = None,
                 low:  pd.DataFrame = None,
                 benchmark: str = BENCHMARK) -> dict:
    """Vectorised SEPA criteria across all dates × stocks."""
    stocks = [c for c in close.columns if c != benchmark]
    sc = close[stocks]

    # yfinance occasionally drops a single daily bar. Because the SMAs require a
    # fully-populated window (min_periods == window), one missing close blanks
    # sma50/150/200 and wrongly fails every moving-average criterion (c1–c7) for
    # an otherwise-valid leader. Patch short isolated gaps before the rolling
    # means so a one-day feed hiccup can't disqualify a stock. The raw close (sc)
    # is still used for the price-vs-SMA comparisons.
    sc_ma = sc.ffill(limit=5)

    ema9   = sc.ewm(span=9,  adjust=False).mean()
    ema21  = sc.ewm(span=21, adjust=False).mean()
    sma50  = sc_ma.rolling(50,  min_periods=50).mean()
    sma150 = sc_ma.rolling(150, min_periods=150).mean()
    sma200 = sc_ma.rolling(200, min_periods=200).mean()

    # Use intraday High/Low for 52w range if provided (matches NSE display)
    # Fall back to close-based range when not available
    if high is not None and not high.empty:
        sh = high[stocks] if all(s in high.columns for s in stocks) else sc
    else:
        sh = sc
    if low is not None and not low.empty:
        sl = low[stocks] if all(s in low.columns for s in stocks) else sc
    else:
        sl = sc

    high52w = sh.rolling(252, min_periods=200).max()
    low52w  = sl.rolling(252, min_periods=200).min()

    # RS Rating: cross-sectional percentile of 12-month return (0–99)
    ret12m    = sc / sc.shift(252) - 1
    rs_rating = ret12m.rank(axis=1, pct=True, na_option="keep") * 99

    c1 = sc > sma150
    c2 = sc > sma200
    c3 = sma150 > sma200
    c4 = (sma200 - sma200.shift(25)) > 0
    c5 = sma50 > sma150
    c6 = sma50 > sma200
    c7 = sc > sma50
    c8 = sc >= high52w * 0.75
    c9 = sc >= low52w  * 1.30
    c10 = rs_rating >= RS_MIN          # relative-strength leadership gate

    sepa_pass = c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8 & c9 & c10

    return {
        "ema9": ema9, "ema21": ema21,
        "sma50": sma50, "sma150": sma150, "sma200": sma200,
        "high52w": high52w, "low52w": low52w,
        "rs_rating": rs_rating,
        "sepa_pass": sepa_pass,
        "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
        "c6": c6, "c7": c7, "c8": c8, "c9": c9, "c10": c10,
    }


# ── Live screener ─────────────────────────────────────────────────────────────

def screen_on_date(sepa: dict, close: pd.DataFrame, ref_date: pd.Timestamp,
                   stocks: list) -> list:
    """Return all stocks with criteria breakdown, sorted: full pass → RS Rating desc.

    Each stock is evaluated at ITS OWN latest valid close on/before ref_date, so a
    name with a one-day data gap (or an uneven yfinance response that lacks a bar on
    the single most-recent frame date) is not silently dropped from the screen."""
    idx = close.index[close.index <= ref_date]
    if idx.empty:
        return []

    rows = []
    for t in stocks:
        if t not in close.columns:
            continue
        # This stock's own last valid (>0) close on/before ref_date.
        col = close[t].reindex(idx)
        col = col[col.notna() & (col > 0)]
        if col.empty:
            continue
        dt    = col.index[-1]
        price = float(col.iloc[-1])

        def _get(key, dt=dt, t=t):
            df = sepa.get(key)
            if df is None or t not in df.columns or dt not in df.index:
                return None
            v = df.loc[dt, t]
            return None if pd.isna(v) else v

        rs    = _get("rs_rating")
        # high52w/low52w come from compute_sepa — already uses intraday H/L when available
        h52   = _get("high52w")
        l52   = _get("low52w")
        e9    = _get("ema9")
        e21   = _get("ema21")
        s50   = _get("sma50")
        s150  = _get("sma150")
        s200  = _get("sma200")

        criteria = {k: bool(_get(k)) for k in CRITERIA_LABELS}
        n_pass = sum(criteria.values())

        # pct_from_52h: positive = % below 52w high (e.g. 5.2 means 5.2% below high)
        # pct_above_52l: positive = % above 52w low (e.g. 45 means 45% above low)
        pct_h = round((1 - price / h52) * 100, 1) if h52 and h52 > 0 else None
        pct_l = round((price / l52 - 1) * 100,  1) if l52 and l52 > 0 else None

        rows.append({
            "ticker":        t,
            "symbol":        t.replace(".NS", ""),
            "price":         round(float(price), 2),
            "rs_rating":     round(float(rs), 1) if rs is not None else None,
            "ema9":          round(float(e9), 2) if e9 is not None else None,
            "ema21":         round(float(e21), 2) if e21 is not None else None,
            "sma50":         round(float(s50), 2) if s50 is not None else None,
            "sma150":        round(float(s150), 2) if s150 is not None else None,
            "sma200":        round(float(s200), 2) if s200 is not None else None,
            "high52w":       round(float(h52), 2) if h52 is not None else None,
            "low52w":        round(float(l52), 2) if l52 is not None else None,
            "pct_from_52h":  pct_h,
            "pct_above_52l": pct_l,
            "criteria":      criteria,
            "passing":       n_pass,
            "sepa_pass":     n_pass == len(CRITERIA_LABELS),
        })

    rows.sort(key=lambda r: (-r["passing"], -(r["rs_rating"] or 0)))
    return rows


def screen_market(market: str, use_cache: bool = True, get_prices=None) -> dict:
    """Screen `market`'s universe for SEPA trend-template passes as of today's
    close. Attaches company name and market cap to every row.

    `get_prices` overrides the price-fetch call (same signature as
    strategy.download_data) — pass a fake in tests instead of hitting yfinance.
    Defaults to strategy.download_data.

    Returns {'rows': [...], 'as_of': str, 'requested': int, 'missing': list[str]}.
    `missing` lists universe tickers that never came back from the price
    download, so a silently shrunk universe is visible, not hidden."""
    get_prices = get_prices or st.download_data
    benchmark = st.BENCHMARKS.get(market, st.BENCHMARK)
    # India uses the MCAP file as universe (covers all NSE stocks ≥ 500 Cr) so the
    # 1000–10000 Cr filter has the full population to work with. Standard index
    # CSVs (Nifty 500, Total Market) miss most small-cap stocks.
    tickers = st.load_universe_from_mcap(min_cr=500) if market == "india" \
              else st.load_universe(market)
    close, _, high, low = get_prices(
        tickers, years=5, include_hl=True, benchmark=benchmark, use_cache=use_cache)
    sepa   = compute_sepa(close, high, low, benchmark=benchmark)
    stocks = [c for c in close.columns if c != benchmark]
    rows   = screen_on_date(sepa, close, pd.Timestamp(date.today()), stocks)

    names = st.load_universe_names(market)
    caps  = st.load_market_caps() if market == "india" else {}
    for r in rows:
        r["name"]       = names.get(r["symbol"], "")
        r["market_cap"] = caps.get(r["symbol"])

    as_of   = str(close.index[-1].date()) if len(close.index) else ""
    missing = [t.replace(".NS", "") for t in tickers if t not in close.columns]
    return {"rows": rows, "as_of": as_of, "requested": len(tickers), "missing": missing}
