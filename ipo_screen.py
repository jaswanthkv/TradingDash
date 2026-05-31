"""
ipo_screen.py — IPO Breakout Screener for NSE-listed equities.

Scans stocks listed on NSE in the last MAX_AGE_DAYS for post-IPO base
formations and breakout setups.

Status definitions:
  Breaking Out  — close ≥ base_high  AND  recent vol ≥ VOL_MULT × 50d avg
  Above Pivot   — close ≥ base_high  but  volume not yet confirming
  Near Pivot    — within NEAR_PCT% below base_high
  Consolidating — in the base, more than NEAR_PCT% below base_high
  Broken Down   — close < base_low (failed IPO base)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import date, timedelta

from strategy import load_universe, download_data, BENCHMARK

MAX_AGE_DAYS = 548    # ~18 months
BASE_DAYS    = 30     # first 30 trading sessions form the IPO base
NEAR_PCT     = 3.0    # within 3% of pivot = "Near Pivot"
VOL_MULT     = 1.5    # 5d avg vol / 50d avg vol threshold for breakout confirmation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_recent_listings(close: pd.DataFrame) -> list[str]:
    """Return tickers whose first valid data date falls within MAX_AGE_DAYS."""
    cutoff = pd.Timestamp(date.today() - timedelta(days=MAX_AGE_DAYS))
    result = []
    for col in close.columns:
        if col == BENCHMARK:
            continue
        fv = close[col].first_valid_index()
        if fv is not None and fv >= cutoff:
            result.append(col)
    return result


def _stock_stats(ticker: str, close: pd.DataFrame, volume: pd.DataFrame) -> dict | None:
    """Compute IPO base + breakout metrics for one ticker."""
    s = close[ticker].dropna()
    if ticker in volume.columns:
        v = volume[ticker].reindex(s.index).fillna(0)
    else:
        v = pd.Series(0.0, index=s.index)

    if len(s) < 5:
        return None

    listing_date  = s.index[0]
    listing_price = float(s.iloc[0])
    current_price = float(s.iloc[-1])
    days_listed   = (s.index[-1] - listing_date).days

    # Base = first BASE_DAYS trading sessions after listing
    base_s    = s.iloc[:BASE_DAYS]
    base_high = float(base_s.max())
    base_low  = float(base_s.min())
    base_depth = round((base_high - base_low) / base_high * 100, 1) if base_high else None

    # Volume ratios: 5d avg / 50d avg
    vol_50 = float(v.rolling(50).mean().iloc[-1]) if len(v) >= 10 else 0.0
    vol_5  = float(v.iloc[-5:].mean())            if len(v) >= 5  else 0.0
    vol_ratio = round(vol_5 / vol_50, 2) if vol_50 > 0 else None

    pct_from_pivot = round((current_price - base_high) / base_high * 100, 1) if base_high else None

    # Status
    if current_price < base_low:
        status = "Broken Down"
    elif pct_from_pivot is not None and pct_from_pivot >= 0:
        status = "Breaking Out" if (vol_ratio or 0) >= VOL_MULT else "Above Pivot"
    elif pct_from_pivot is not None and pct_from_pivot >= -NEAR_PCT:
        status = "Near Pivot"
    else:
        status = "Consolidating"

    # 3-month RS vs benchmark
    rs_3m = None
    if BENCHMARK in close.columns:
        bench = close[BENCHMARK].reindex(s.index).dropna()
        w = min(63, len(s) - 1, len(bench) - 1)
        if w >= 10:
            sr = float(s.iloc[-1] / s.iloc[-w] - 1)
            br = float(bench.iloc[-1] / bench.iloc[-w] - 1)
            rs_3m = round((sr - br) * 100, 1)

    avg_vol_20d = int(v.iloc[-20:].mean()) if len(v) >= 5 else 0

    return {
        "ticker":            ticker,
        "symbol":            ticker.replace(".NS", ""),
        "listing_date":      listing_date.strftime("%Y-%m-%d"),
        "days_listed":       days_listed,
        "listing_price":     round(listing_price, 2),
        "base_high":         round(base_high, 2),
        "base_low":          round(base_low, 2),
        "base_depth_pct":    base_depth,
        "current_price":     round(current_price, 2),
        "pct_from_pivot":    pct_from_pivot,
        "vol_ratio":         vol_ratio,
        "return_vs_listing": round((current_price - listing_price) / listing_price * 100, 1) if listing_price else None,
        "rs_3m":             rs_3m,
        "avg_vol_20d":       avg_vol_20d,
        "status":            status,
    }


_STATUS_RANK = {
    "Breaking Out":  0,
    "Near Pivot":    1,
    "Above Pivot":   2,
    "Consolidating": 3,
    "Broken Down":   4,
}


# ── Public entry point ────────────────────────────────────────────────────────

def run_ipo_screen(years: int = 2, on_progress=None) -> dict:
    """
    Full IPO breakout screen pipeline.
    Downloads universe data, finds recent listings, computes base stats.
    Returns a dict with results sorted by status priority.
    """
    def _p(msg):
        if on_progress:
            on_progress(msg)

    _p("Downloading universe data…")
    tickers       = load_universe()
    close, volume = download_data(tickers, years=years)

    _p("Identifying recent listings…")
    recent = _find_recent_listings(close)

    _p(f"Analysing {len(recent)} recent listings…")
    results = []
    for ticker in recent:
        try:
            stats = _stock_stats(ticker, close, volume)
            if stats:
                results.append(stats)
        except Exception:
            pass

    results.sort(key=lambda r: (
        _STATUS_RANK.get(r["status"], 9),
        -(r["pct_from_pivot"] if r["pct_from_pivot"] is not None else -999),
    ))

    _p("Done.")
    return {
        "as_of":         str(date.today()),
        "universe_size": len([c for c in close.columns if c != BENCHMARK]),
        "total_scanned": len(recent),
        "results":       results,
    }
