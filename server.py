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

    bench_label = st.BENCHMARK_LABELS.get(market, "Nifty 500")

    def _run():
        _screener_running[market] = True
        try:
            return mv.screen_market(market, use_cache=not fresh)
        finally:
            _screener_running[market] = False

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    rows = result["rows"]
    return {
        "market":          market,
        "benchmark_label": bench_label,
        "as_of":           result["as_of"],
        "rows":            rows,
        "criteria_short":  mv.CRITERIA_SHORT,
        "criteria_labels": mv.CRITERIA_LABELS,
        "full_pass":       sum(1 for r in rows if r.get("sepa_pass")),
        "universe":        len(rows),
        "requested":       result["requested"],
        "screened":        len(rows),
        "missing":         result["missing"],
        "missing_count":   len(result["missing"]),
    }


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
