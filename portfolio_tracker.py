"""
portfolio_tracker.py — Paper-trading journal for the SEPA Top 20 strategy.

Not connected to any broker — a systematic performance record only. Weekly
(ISO calendar week), liquidates and re-buys equal-weight into the current Top
20 (SEPA full-pass, ranked by RS Rating — the same rule the removed Buy Top 20
button used). Records one NAV snapshot per trading day it's asked, alongside
that day's close for three benchmarks (Nifty 50, Nifty 500, Nifty Midcap 150),
so portfolio vs. benchmarks can be plotted as indexed (base=100) lines.

No background thread: ensure_today() is called on demand (when the Portfolio
tab loads) and is idempotent — safe to call many times a day, rebalances and
snapshots at most once per trading day.

State lives in portfolio_state.json (gitignored) — a solo local journal, not
a shared or authoritative ledger.
"""
from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd

import strategy as st

STATE_FILE       = os.path.join(os.path.dirname(__file__), "portfolio_state.json")
STARTING_CAPITAL = 1_000_000.0
TOP_N            = 20

PORTFOLIO_BENCHMARKS = {
    "nifty50":   "^NSEI",
    "nifty500":  "^CRSLDX",
    "midcap150": "NIFTYMIDCAP150.NS",
}


# ── State I/O ────────────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "inception_date":     None,
        "starting_capital":   STARTING_CAPITAL,
        "cash":               STARTING_CAPITAL,
        "holdings":           {},   # symbol -> {"qty": int, "avg_cost": float}
        "last_rebalance_date": None,
        "nav_history":        [],   # [{date, nav, nifty50, nifty500, midcap150}]
        "rebalance_log":      [],   # [{date, added: [...], removed: [...]}]
    }


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return _default_state()
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Selection (mirrors the removed Buy Top 20 rule) ─────────────────────────

def _compute_top20(get_prices=None) -> list[str]:
    """SEPA full-pass NSE stocks (≥500 Cr), ranked by RS Rating, top 20."""
    import minervini as mv

    rows = mv.screen_market("india", use_cache=True, get_prices=get_prices)["rows"]
    candidates = [r for r in rows if r.get("sepa_pass") and r.get("rs_rating") is not None]
    candidates.sort(key=lambda r: r["rs_rating"], reverse=True)
    return [r["symbol"] for r in candidates[:TOP_N]]


# ── Prices ───────────────────────────────────────────────────────────────────

def _get_current_prices(symbols: list[str], get_prices=None):
    """Latest close for each bare NSE symbol + the 3 benchmarks, the trading-day
    date implied by that data (so weekend page-loads don't create a duplicate
    snapshot), and each ticker's day-over-day % change vs the previous close —
    already in the downloaded frame, so this costs nothing extra to fetch.

    `get_prices` overrides the price-fetch call — pass a fake in tests instead
    of hitting yfinance. Defaults to strategy.download_data."""
    get_prices = get_prices or st.download_data
    tickers = sorted({f"{s}.NS" for s in symbols} | set(PORTFOLIO_BENCHMARKS.values()))
    close, _ = get_prices(
        tickers, years=1, benchmark=PORTFOLIO_BENCHMARKS["nifty500"], use_cache=True)
    if close.empty:
        raise RuntimeError("No price data returned for portfolio tracking")
    # yfinance occasionally drops an isolated bar for a thin-liquidity stock (a
    # transient feed gap, not a real halt) — same ffill(limit=5) minervini.py
    # uses for SMA continuity, so one missing day doesn't blank out day_change.
    close = close.ffill(limit=5)

    last  = close.iloc[-1]
    prev  = close.iloc[-2] if len(close) >= 2 else None
    as_of = str(close.index[-1].date())

    def _day_change(ticker: str) -> float | None:
        if prev is None or ticker not in prev.index or ticker not in last.index:
            return None
        p0, p1 = prev[ticker], last[ticker]
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            return None
        return round((p1 / p0 - 1) * 100, 2)

    prices = {s: float(last[f"{s}.NS"]) for s in symbols
              if f"{s}.NS" in last.index and pd.notna(last[f"{s}.NS"])}
    bench  = {name: float(last[ticker]) for name, ticker in PORTFOLIO_BENCHMARKS.items()
              if ticker in last.index and pd.notna(last[ticker])}

    day_change: dict[str, float] = {}
    for s in symbols:
        chg = _day_change(f"{s}.NS")
        if chg is not None:
            day_change[s] = chg
    for name, ticker in PORTFOLIO_BENCHMARKS.items():
        chg = _day_change(ticker)
        if chg is not None:
            day_change[name] = chg

    return prices, bench, as_of, day_change


# ── Rebalance / snapshot ─────────────────────────────────────────────────────

def _week_key(d: str) -> tuple:
    iso = pd.Timestamp(d).isocalendar()
    return (iso.year, iso.week)


def needs_rebalance(state: dict, today: str) -> bool:
    if not state["last_rebalance_date"]:
        return True
    return _week_key(today) != _week_key(state["last_rebalance_date"])


def _portfolio_value(state: dict, prices: dict[str, float]) -> float:
    value = state["cash"]
    for sym, pos in state["holdings"].items():
        px = prices.get(sym)
        if px:
            value += pos["qty"] * px
    return value


def rebalance(state: dict, top20: list[str], prices: dict[str, float], today: str) -> dict:
    """Full liquidate + equal-weight re-buy into the current Top 20. Simplified
    on purpose — a paper journal doesn't need partial-position rebalancing math,
    and full reallocation is what "rebalance to the current Top 20" means here."""
    prev_symbols = set(state["holdings"].keys())
    total_value  = _portfolio_value(state, prices)

    priced_top20 = [s for s in top20 if s in prices]
    new_holdings = {}
    if priced_top20:
        per_stock = total_value / len(priced_top20)
        for sym in priced_top20:
            qty = int(per_stock // prices[sym])
            if qty > 0:
                new_holdings[sym] = {"qty": qty, "avg_cost": round(prices[sym], 2)}

    spent = sum(pos["qty"] * prices[sym] for sym, pos in new_holdings.items())
    state["holdings"]            = new_holdings
    state["cash"]                = round(total_value - spent, 2)
    state["last_rebalance_date"] = today
    if not state["inception_date"]:
        state["inception_date"] = today

    added   = sorted(set(new_holdings) - prev_symbols)
    removed = sorted(prev_symbols - set(new_holdings))
    state["rebalance_log"].append({"date": today, "added": added, "removed": removed})
    return state


def snapshot_nav(state: dict, prices: dict[str, float], bench: dict[str, float], today: str) -> dict:
    if state["nav_history"] and state["nav_history"][-1]["date"] == today:
        return state   # already recorded today
    entry = {"date": today, "nav": round(_portfolio_value(state, prices), 2), **bench}
    state["nav_history"].append(entry)
    return state


# ── Entry point ───────────────────────────────────────────────────────────────

def ensure_today() -> tuple[dict, dict, dict]:
    """Rebalance if a new ISO week has started, record today's NAV if not
    already done, persist, and return (state, current_prices, day_change) —
    current_prices/day_change let the caller show live P&L/weight/today's-move
    per holding without a second fetch."""
    state = load_state()
    top20 = _compute_top20()

    symbols_needed = sorted(set(top20) | set(state["holdings"].keys()))
    prices, bench, today, day_change = _get_current_prices(symbols_needed)

    if needs_rebalance(state, today) and top20:
        state = rebalance(state, top20, prices, today)

    state = snapshot_nav(state, prices, bench, today)
    save_state(state)
    return state, prices, day_change
