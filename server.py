"""
server.py — FastAPI backend for the QuantDesk dashboard.

Endpoints:
  GET  /                        — dashboard (index.html)

  GET  /api/screener            — live SEPA screener (market=india|us, fresh=bool)

  GET  /api/kite/status         — Kite connection status
  GET  /api/kite/login          — redirect to Kite OAuth login page
  GET  /api/kite/callback       — OAuth callback

  POST /api/trade/preview       — live LTP + qty preview for an equal-rupee buy across symbols
  POST /api/trade/buy           — place CNC market buy orders (equity)
  GET  /api/trade/holdings      — current NSE equity holdings
  POST /api/trade/sell          — sell (part or all of) an equity holding

  GET  /api/pulse/signal        — live NIFTY HA signal + option details
  POST /api/pulse/sell          — sell NIFTY ATM option
  POST /api/pulse/close         — close an open option position
  GET  /api/pulse/auto          — auto-execute status + trade log
  POST /api/pulse/auto          — enable / disable auto-execute
"""
import asyncio
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as _dt
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

import pandas as pd
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

import kite_auth
from config import PORT, KITE_API_KEY

app = FastAPI(title="QuantDesk")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_executor = ThreadPoolExecutor(max_workers=4)

_min_markets = ("india", "us")


# ── Pulse auto-execute state ──────────────────────────────────────────────────
_pulse_auto_enabled:  bool        = False
_pulse_auto_signal:   str         = "FLAT"   # last signal we ACTED on
_pulse_auto_log:      list        = []
_pulse_auto_lots:     int         = 1
_pulse_auto_thread:   threading.Thread | None = None
_IST                              = ZoneInfo("Asia/Kolkata")


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


# ── Kite auth ────────────────────────────────────────────────────────────────

@app.get("/api/kite/status")
def kite_status():
    return kite_auth.kite_status()

@app.get("/api/kite/login")
def kite_login():
    if not KITE_API_KEY:
        raise HTTPException(503, "KITE_API_KEY not set in .env")
    return RedirectResponse(kite_auth.get_login_url())

@app.get("/api/kite/callback")
def kite_callback(request_token: str = "", status: str = ""):
    if status != "success" or not request_token:
        return RedirectResponse("/?kite=failed")
    try:
        kite_auth.complete_login(request_token)
        return RedirectResponse("/?kite=connected")
    except Exception:
        return RedirectResponse("/?kite=error")


# ── Equity trading (Buy Top 20 / holdings) ─────────────────────────────────────

class TradePreviewParams(BaseModel):
    symbols: list[str]
    budget:  float


class TradeBuyOrder(BaseModel):
    symbol: str
    qty:    int


class TradeBuyParams(BaseModel):
    orders: list[TradeBuyOrder]


class TradeSellParams(BaseModel):
    symbol:   str
    quantity: int


def _require_kite():
    status = kite_auth.kite_status()
    if not status.get("connected"):
        raise HTTPException(401, status.get("reason", "Kite not connected"))
    return kite_auth.get_kite()


@app.post("/api/trade/preview")
async def trade_preview(params: TradePreviewParams):
    import trade_live as tl
    kite = _require_kite()
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(_executor, lambda: tl.preview_buy(kite, params.symbols, params.budget))
    return {"rows": rows, "budget": params.budget}


@app.post("/api/trade/buy")
async def trade_buy(params: TradeBuyParams):
    import trade_live as tl
    kite = _require_kite()
    orders = [o.dict() for o in params.orders]
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(_executor, lambda: tl.place_buy_orders(kite, orders))
    return {"results": results}


@app.get("/api/trade/holdings")
async def trade_holdings():
    import trade_live as tl
    kite = _require_kite()
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(_executor, lambda: tl.get_holdings(kite))
    return {"rows": rows}


@app.post("/api/trade/sell")
async def trade_sell(params: TradeSellParams):
    import trade_live as tl
    from kiteconnect.exceptions import KiteException
    kite = _require_kite()
    loop = asyncio.get_event_loop()
    try:
        oid = await loop.run_in_executor(
            _executor, lambda: tl.sell_holding(kite, params.symbol, params.quantity))
    except KiteException as e:
        raise HTTPException(403, str(e))
    return {"order_id": oid, "symbol": params.symbol, "quantity": params.quantity}


def _benchmark_for(market: str):
    """(yfinance symbol, display label) for a market's benchmark index."""
    return ("^GSPC", "S&P 500") if market == "us" else ("^CRSLDX", "Nifty 500")


# ── Pulse auto-execute background engine ─────────────────────────────────────

def _pulse_auto_loop():
    """
    Background thread: checks the HA signal 3 minutes after every 30-min
    candle close (09:18, 09:48, 10:18, … 15:18) during IST market hours.
    When the active_signal changes, closes any open NIFTY option position
    and sells the new ATM option in the correct direction.
    """
    global _pulse_auto_enabled, _pulse_auto_signal, _pulse_auto_log, _pulse_auto_lots

    def _log(entry: dict):
        _pulse_auto_log.append({**entry, "time": _dt.now(_IST).isoformat(timespec="seconds")})
        if len(_pulse_auto_log) > 100:
            _pulse_auto_log[:] = _pulse_auto_log[-100:]

    last_checked_slot = ""

    while True:
        time.sleep(20)

        if not _pulse_auto_enabled:
            last_checked_slot = ""
            continue

        now = _dt.now(_IST)

        # Only Mon–Fri, 09:15–15:30 IST
        if now.weekday() >= 5:
            continue
        h, m = now.hour, now.minute
        if not ((h == 9 and m >= 15) or (10 <= h <= 14) or (h == 15 and m <= 30)):
            continue

        # Fire 3 min after each 30-min boundary: :18 and :48
        slot = f"{now.date()}-{h}-{'A' if m < 30 else 'B'}"
        fire = (m in (18, 19, 20) or m in (48, 49, 50))
        if not fire or slot == last_checked_slot:
            continue
        last_checked_slot = slot

        try:
            status = kite_auth.kite_status()
            if not status.get("connected"):
                _log({"action": "SKIP", "reason": "Kite not connected"})
                continue

            import pulse_live as pl
            kite = kite_auth.get_kite()

            sig = pl.current_signal(kite)
            new_signal = sig.get("active_signal", "FLAT")

            _log({"action": "CHECK", "signal": new_signal,
                  "prev": _pulse_auto_signal, "nifty": sig.get("nifty_price")})

            if new_signal == "FLAT" or new_signal == _pulse_auto_signal:
                continue

            # ── Signal changed: close existing, open new ──────────────────
            positions = pl.nifty_positions(kite)
            for pos in positions:
                if pos["quantity"] == 0:
                    continue
                try:
                    oid = pl.close_option(kite, pos["tradingsymbol"], pos["quantity"])
                    _log({"action": "CLOSE", "symbol": pos["tradingsymbol"],
                          "qty": pos["quantity"], "order_id": oid})
                except Exception as exc:
                    _log({"action": "CLOSE_ERR", "symbol": pos["tradingsymbol"], "error": str(exc)})

            # Sell new ATM option
            expiry = pl.weekly_expiry(kite)
            if not expiry:
                _log({"action": "ERR", "reason": "No weekly expiry found"})
                continue

            opt = pl.option_details(kite, new_signal, sig["nifty_price"], expiry, strike_offset=0)
            if not opt or "error" in opt:
                _log({"action": "ERR", "reason": opt.get("error", "No instrument") if opt else "No instrument"})
                continue

            try:
                oid = pl.sell_option(kite, opt["tradingsymbol"], _pulse_auto_lots)
                _log({"action": "SELL", "signal": new_signal,
                      "symbol": opt["tradingsymbol"], "ltp": opt.get("ltp"),
                      "lots": _pulse_auto_lots, "order_id": oid})
                _pulse_auto_signal = new_signal
            except Exception as exc:
                _log({"action": "SELL_ERR", "symbol": opt["tradingsymbol"], "error": str(exc)})

        except Exception as exc:
            _log({"action": "ERR", "error": str(exc)})


def _ensure_auto_thread():
    global _pulse_auto_thread
    if _pulse_auto_thread is None or not _pulse_auto_thread.is_alive():
        _pulse_auto_thread = threading.Thread(target=_pulse_auto_loop, daemon=True, name="pulse-auto")
        _pulse_auto_thread.start()


# ── Pulse live signal + trade ─────────────────────────────────────────────────

@app.get("/api/pulse/signal")
async def pulse_signal():
    import pulse_live as pl

    # Check Kite connection first
    status = kite_auth.kite_status()
    if not status.get("connected"):
        raise HTTPException(401, status.get("reason", "Kite not connected — please login"))

    kite = kite_auth.get_kite()
    loop = asyncio.get_event_loop()

    sig    = await loop.run_in_executor(_executor, lambda: pl.current_signal(kite))
    expiry = await loop.run_in_executor(_executor, lambda: pl.weekly_expiry(kite))
    opt    = await loop.run_in_executor(_executor,
                lambda: pl.option_details(kite, sig["active_signal"], sig["nifty_price"],
                                          expiry, strike_offset=0)) \
             if expiry and sig.get("active_signal", "FLAT") != "FLAT" else None
    pos    = await loop.run_in_executor(_executor, lambda: pl.nifty_positions(kite))

    return {**sig, "option": opt, "expiry": str(expiry) if expiry else None,
            "positions": pos, "auto_enabled": _pulse_auto_enabled}


class PulseTradeParams(BaseModel):
    tradingsymbol: str
    lots:          int = 1


@app.post("/api/pulse/sell")
async def pulse_sell(params: PulseTradeParams):
    import pulse_live as pl
    from kiteconnect.exceptions import KiteException
    status = kite_auth.kite_status()
    if not status.get("connected"):
        raise HTTPException(401, status.get("reason", "Kite not connected"))
    kite = kite_auth.get_kite()
    loop = asyncio.get_event_loop()
    try:
        oid = await loop.run_in_executor(_executor,
                  lambda: pl.sell_option(kite, params.tradingsymbol, params.lots))
    except KiteException as e:
        raise HTTPException(403, str(e))
    return {"order_id": oid, "tradingsymbol": params.tradingsymbol, "lots": params.lots}


class PulseCloseParams(BaseModel):
    tradingsymbol: str
    quantity:      int


@app.post("/api/pulse/close")
async def pulse_close(params: PulseCloseParams):
    import pulse_live as pl
    from kiteconnect.exceptions import KiteException
    status = kite_auth.kite_status()
    if not status.get("connected"):
        raise HTTPException(401, status.get("reason", "Kite not connected"))
    kite = kite_auth.get_kite()
    loop = asyncio.get_event_loop()
    try:
        oid = await loop.run_in_executor(_executor,
                  lambda: pl.close_option(kite, params.tradingsymbol, params.quantity))
    except KiteException as e:
        raise HTTPException(403, str(e))
    return {"order_id": oid, "tradingsymbol": params.tradingsymbol}


class PulseAutoParams(BaseModel):
    enabled: bool
    lots:    int = 1


@app.post("/api/pulse/auto")
async def pulse_auto_toggle(params: PulseAutoParams):
    global _pulse_auto_enabled, _pulse_auto_lots
    _pulse_auto_enabled = params.enabled
    _pulse_auto_lots    = max(1, params.lots)
    if params.enabled:
        _ensure_auto_thread()
    return {"enabled": _pulse_auto_enabled, "lots": _pulse_auto_lots}


@app.get("/api/pulse/auto")
async def pulse_auto_status():
    return {
        "enabled":     _pulse_auto_enabled,
        "lots":        _pulse_auto_lots,
        "last_signal": _pulse_auto_signal,
        "log":         _pulse_auto_log[-30:],
    }


@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return open(path).read() if os.path.exists(path) else "<h1>index.html not found</h1>"
