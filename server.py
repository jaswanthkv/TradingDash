"""
server.py — FastAPI backend for the QuantDesk dashboard.

Endpoints:
  GET  /                        — dashboard (index.html)

  GET  /api/screener            — live SEPA screener (market=india|us, fresh=bool)

  GET  /api/portfolio           — SEPA Top 20 paper-trading journal vs Nifty 50/500/Midcap 150
"""
import asyncio
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import pandas as pd
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config import PORT

app = FastAPI(title="QuantDesk")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_executor = ThreadPoolExecutor(max_workers=4)

_min_markets = ("india", "us")


def _norm_market(market: str) -> str:
    return market if market in _min_markets else "india"


# ── Live SEPA screener ─────────────────────────────────────────────────────────

_screener_running = {m: False for m in _min_markets}


@app.get("/api/screener")
async def screener(market: str = "india", fresh: bool = True):
    """Near-real-time SEPA screen for a market, computed on the latest available
    daily closes (not the stored backtest snapshot). Defaults to a live re-pull of
    real prices (fresh=true); pass fresh=false to reuse the same-day cache."""
    market = _norm_market(market)
    if _screener_running[market]:
        raise HTTPException(409, "Screener already refreshing for this market")
    import minervini as mv
    import strategy as st

    benchmark = st.BENCHMARKS.get(market, st.BENCHMARK)
    _, bench_label = _benchmark_for(market)

    def _run():
        _screener_running[market] = True
        try:
            # India screener uses the MCAP file as universe (covers all NSE stocks
            # ≥ 500 Cr) so the 1000–10000 Cr filter has the full population to work with.
            # Standard index CSVs (Nifty 500, Total Market) miss most small-cap stocks.
            tickers = st.load_universe_from_mcap(min_cr=500) if market == "india" \
                      else st.load_universe(market)
            # 5y window matches the backtest's cache key, so the screener reuses
            # the same-day price cache (instant) instead of re-downloading.
            close, _, high, low = st.download_data(
                tickers, years=5, include_hl=True, benchmark=benchmark,
                use_cache=not fresh)
            sepa   = mv.compute_sepa(close, high, low, benchmark=benchmark)
            stocks = [c for c in close.columns if c != benchmark]
            rows   = mv.screen_on_date(sepa, close, pd.Timestamp(date.today()), stocks)
            names = st.load_universe_names(market)
            caps  = st.load_market_caps() if market == "india" else {}
            for r in rows:
                r["name"]       = names.get(r["symbol"], "")
                r["market_cap"] = caps.get(r["symbol"])
            as_of   = str(close.index[-1].date()) if len(close.index) else ""
            # Universe tickers that never came back from the price download — so a
            # silently shrunk universe (yfinance batch failure) is visible, not hidden.
            missing = [t.replace(".NS", "") for t in tickers if t not in close.columns]
            return rows, as_of, len(tickers), missing
        finally:
            _screener_running[market] = False

    loop = asyncio.get_event_loop()
    rows, as_of, requested, missing = await loop.run_in_executor(_executor, _run)
    return {
        "market":          market,
        "benchmark_label": bench_label,
        "as_of":           as_of,
        "rows":            rows,
        "full_pass":       sum(1 for r in rows if r.get("sepa_pass")),
        "universe":        len(rows),
        "requested":       requested,
        "screened":        len(rows),
        "missing":         missing,
        "missing_count":   len(missing),
    }


def _benchmark_for(market: str):
    """(yfinance symbol, display label) for a market's benchmark index."""
    return ("^GSPC", "S&P 500") if market == "us" else ("^CRSLDX", "Nifty 500")


# ── SEPA Top 20 paper-trading journal ───────────────────────────────────────────

_portfolio_running = False


@app.get("/api/portfolio")
async def portfolio():
    """Rebalances (weekly, if due) and records today's NAV snapshot on demand —
    no background thread. Safe to call repeatedly; a no-op after the first
    successful call on a given trading day."""
    global _portfolio_running
    if _portfolio_running:
        raise HTTPException(409, "Portfolio already updating")
    import portfolio_tracker as pt

    def _run():
        global _portfolio_running
        _portfolio_running = True
        try:
            return pt.ensure_today()
        finally:
            _portfolio_running = False

    loop = asyncio.get_event_loop()
    state, prices, day_change = await loop.run_in_executor(_executor, _run)
    return {**state, "current_prices": prices, "day_change": day_change}


@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return open(path).read() if os.path.exists(path) else "<h1>index.html not found</h1>"
