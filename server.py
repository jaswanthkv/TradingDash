"""
server.py — FastAPI backend for the ML Strategy dashboard.

Endpoints:
  GET  /                              — dashboard (index.html)

  POST /api/ml/backtest/run           — launch walk-forward ML backtest (~60–90s)
  GET  /api/ml/backtest/result        — latest cached ML backtest result
  GET  /api/ml/backtest/status        — running / has_result
  GET  /api/ml/backtest/history       — all past ML runs (params + timestamp)
  GET  /api/ml/rank                   — current ML rankings (cached 5 min)

  POST /api/momentum/backtest/run     — launch momentum backtest (~30–60s)
  GET  /api/momentum/backtest/result  — latest cached momentum backtest result
  GET  /api/momentum/backtest/status  — running / has_result

  POST /api/minervini/backtest/run    — launch Minervini SEPA backtest
  GET  /api/minervini/backtest/result — latest cached Minervini backtest result
  GET  /api/minervini/backtest/status — running / has_result

  GET  /api/kite/status               — Kite connection status
  GET  /api/kite/login                — redirect to Kite OAuth login page
  GET  /api/kite/callback             — OAuth callback
  POST /api/ha/backtest/run           — launch HA 30-min backtest (needs Kite)
  GET  /api/ha/backtest/result        — latest HA backtest result
  GET  /api/ha/backtest/status        — running / has_result
"""
import asyncio
import calendar
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import pandas as pd
from datetime import date, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

import db
import kite_auth
from config import PORT, KITE_API_KEY

app = FastAPI(title="QuantDesk")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_executor = ThreadPoolExecutor(max_workers=4)

# In-memory caches — seeded from DB on startup
_ml_bt_cache: dict    = db.get_latest_backtest(strategy="ml")
_ml_bt_running: bool  = False
_ml_bt_progress: dict = {"step": 0, "total": 6, "msg": ""}

_mom_bt_cache: dict    = db.get_latest_backtest(strategy="momentum")
_mom_bt_running: bool  = False
_mom_bt_progress: dict = {"step": 0, "total": 5, "msg": ""}

_min_bt_cache: dict    = db.get_latest_backtest(strategy="minervini")
_min_bt_running: bool  = False
_min_bt_progress: dict = {"step": 0, "total": 5, "msg": ""}

_ml_rank_cache: dict = db.get_latest_rank_snapshot()
_ml_rank_ts:    float = 0.0   # force a fresh fetch on first request
_ML_RANK_TTL    = 300         # seconds


# ── ML backtest ───────────────────────────────────────────────────────────────

class MLBacktestParams(BaseModel):
    years:    int   = 5
    top_n:    int   = 20
    cost_bps: float = 20


@app.post("/api/ml/backtest/run")
async def ml_backtest_run(params: MLBacktestParams):
    global _ml_bt_running
    if _ml_bt_running:
        raise HTTPException(409, "ML backtest already running")
    import backtest_ml as btml

    def _run():
        global _ml_bt_running, _ml_bt_cache, _ml_bt_progress
        _ml_bt_running = True
        _ml_bt_progress = {"step": 0, "total": 6, "msg": "Starting…"}
        def _on_progress(step, total, msg):
            _ml_bt_progress.update({"step": step, "total": total, "msg": msg})
        try:
            result = btml.run_backtest(
                years=params.years, top_n=params.top_n, cost_bps=params.cost_bps,
                progress_cb=_on_progress,
            )
            _ml_bt_cache = result
            db.save_backtest(params.years, params.top_n, params.cost_bps, result, strategy="ml")
            return result
        finally:
            _ml_bt_running = False

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    return result


@app.get("/api/ml/backtest/result")
def ml_backtest_result():
    if not _ml_bt_cache:
        raise HTTPException(404, "No backtest result yet — run one first")
    return _ml_bt_cache


@app.get("/api/ml/backtest/status")
def ml_backtest_status():
    return {"running": _ml_bt_running, "has_result": bool(_ml_bt_cache),
            "progress": _ml_bt_progress}


@app.get("/api/ml/backtest/history")
def ml_backtest_history():
    return db.list_backtest_runs()


# ── ML rankings ───────────────────────────────────────────────────────────────

@app.get("/api/ml/rank")
async def ml_rank(years: int = 3):
    global _ml_rank_cache, _ml_rank_ts
    if _ml_rank_cache and (time.time() - _ml_rank_ts) < _ML_RANK_TTL:
        return _ml_rank_cache
    import strategy as strat
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, lambda: strat.get_current_rankings(years))
    _ml_rank_cache = result
    _ml_rank_ts    = time.time()
    db.save_rank_snapshot(result)
    return result


# ── Momentum backtest ─────────────────────────────────────────────────────────

class MomentumParams(BaseModel):
    m:     int = 8
    x:     int = 3
    years: int = 10


@app.post("/api/momentum/backtest/run")
async def momentum_backtest_run(params: MomentumParams):
    global _mom_bt_running
    if _mom_bt_running:
        raise HTTPException(409, "Momentum backtest already running")
    import backtest as bt

    def _run():
        global _mom_bt_running, _mom_bt_cache, _mom_bt_progress
        _mom_bt_running = True
        _mom_bt_progress = {"step": 0, "total": 5, "msg": "Starting…"}
        def _on_progress(step, total, msg):
            _mom_bt_progress.update({"step": step, "total": total, "msg": msg})
        try:
            result = bt.run_backtest(m=params.m, x=params.x, years=params.years,
                                     progress_cb=_on_progress)
            _mom_bt_cache = result
            db.save_backtest(params.years, params.m, params.x, result, strategy="momentum")
            return result
        finally:
            _mom_bt_running = False

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    return result


@app.get("/api/momentum/backtest/result")
def momentum_backtest_result():
    if not _mom_bt_cache:
        raise HTTPException(404, "No momentum backtest result — run one first")
    return _mom_bt_cache


@app.get("/api/momentum/backtest/status")
def momentum_backtest_status():
    return {"running": _mom_bt_running, "has_result": bool(_mom_bt_cache),
            "progress": _mom_bt_progress}


# ── Minervini SEPA backtest ───────────────────────────────────────────────────

class MinerviniParams(BaseModel):
    years:    int   = 5
    top_n:    int   = 20
    cost_bps: float = 20


@app.post("/api/minervini/backtest/run")
async def minervini_backtest_run(params: MinerviniParams):
    global _min_bt_running
    if _min_bt_running:
        raise HTTPException(409, "Minervini backtest already running")
    import minervini as mv

    def _run():
        global _min_bt_running, _min_bt_cache, _min_bt_progress
        _min_bt_running = True
        _min_bt_progress = {"step": 0, "total": 5, "msg": "Starting…"}
        def _on_progress(step, total, msg):
            _min_bt_progress.update({"step": step, "total": total, "msg": msg})
        try:
            result = mv.run_backtest(
                years=params.years, top_n=params.top_n, cost_bps=params.cost_bps,
                progress_cb=_on_progress,
            )
            _min_bt_cache = result
            db.save_backtest(params.years, params.top_n, params.cost_bps, result,
                             strategy="minervini")
            return result
        finally:
            _min_bt_running = False

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    return result


@app.get("/api/minervini/backtest/result")
def minervini_backtest_result():
    if not _min_bt_cache:
        raise HTTPException(404, "No Minervini backtest result — run one first")
    return _min_bt_cache


@app.get("/api/minervini/backtest/status")
def minervini_backtest_status():
    return {"running": _min_bt_running, "has_result": bool(_min_bt_cache),
            "progress": _min_bt_progress}


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


# ── Portfolio live ────────────────────────────────────────────────────────────

def _next_rebalance() -> dict:
    today = date.today()
    def _last_biz(y, m):
        last = date(y, m, calendar.monthrange(y, m)[1])
        while last.weekday() >= 5:
            last -= timedelta(days=1)
        return last
    r = _last_biz(today.year, today.month)
    if today >= r:
        nm = today.month % 12 + 1
        ny = today.year + (1 if today.month == 12 else 0)
        r = _last_biz(ny, nm)
    return {"date": r.isoformat(), "days_left": (r - today).days, "label": r.strftime("%b %d")}


@app.get("/api/portfolio/daily-change")
async def portfolio_daily_change():
    """Daily % change for ML + Momentum strategy portfolios vs Nifty 500."""
    import yfinance as yf
    import numpy as np

    # Current display portfolio: today's rankings (for Today % column and table display)
    ml_tickers_now, mom_tickers_now = [], []
    if _ml_rank_cache:
        ml_tickers_now = [r["ticker"] for r in _ml_rank_cache.get("rankings", []) if r.get("in_top20")]
    if _mom_bt_cache:
        rh = _mom_bt_cache.get("rebalance_history", [])
        if rh:
            mom_tickers_now = rh[-1].get("holdings", [])

    # MTD portfolio: last rebalance holdings (what was actually entered this month)
    # Falls back to current rankings if backtest hasn't been run.
    ml_tickers_mtd = ml_tickers_now
    if _ml_bt_cache:
        rh_ml = _ml_bt_cache.get("rebalance_history", [])
        if rh_ml:
            ml_tickers_mtd = rh_ml[-1].get("holdings", [])
    mom_tickers_mtd = mom_tickers_now  # momentum already uses rebalance history

    # Union for a single download pass
    ml_tickers  = ml_tickers_now
    mom_tickers = mom_tickers_now
    all_tickers = list(set(ml_tickers_now + mom_tickers_now +
                           ml_tickers_mtd + mom_tickers_mtd + ["^CRSLDX"]))
    from datetime import date as _date
    month_start = _date.today().replace(day=1).strftime("%Y-%m-%d")

    def _fetch():
        # Last-session change (5d window, last bar vs prev bar)
        raw5 = yf.download(all_tickers, period="5d", interval="1d",
                           auto_adjust=True, progress=False, threads=True)
        c5 = raw5["Close"] if isinstance(raw5.columns, pd.MultiIndex) else raw5
        c5 = c5.ffill().dropna(how="all")
        day_chg = {}
        if len(c5) >= 2:
            pct = c5.pct_change().iloc[-1] * 100
            day_chg = {col: round(float(pct[col]), 2)
                       for col in pct.index if not pd.isna(pct[col])}

        # MTD change: from first trading day of current month to latest close
        raw_m = yf.download(all_tickers, start=month_start, interval="1d",
                            auto_adjust=True, progress=False, threads=True)
        cm = raw_m["Close"] if isinstance(raw_m.columns, pd.MultiIndex) else raw_m
        cm = cm.dropna(how="all")
        mtd_chg = {}
        if len(cm) >= 1:
            # Use first non-NaN per column so tickers starting mid-month don't get NaN
            first_vals = cm.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else float("nan"))
            last_vals  = cm.apply(lambda col: col.dropna().iloc[-1] if col.notna().any() else float("nan"))
            pct_m = (last_vals / first_vals - 1) * 100
            mtd_chg = {col: round(float(pct_m[col]), 2)
                       for col in pct_m.index if not pd.isna(pct_m[col])}

        return day_chg, mtd_chg, c5, cm

    loop = asyncio.get_event_loop()
    changes, mtd, c5, cm = await loop.run_in_executor(_executor, _fetch)

    def _port_avg(tickers, src):
        vals = [src[t] for t in tickers if t in src]
        return round(float(np.mean(vals)), 2) if vals else None

    # Date labels
    def _closes(df):
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"].squeeze() if "Close" in df.columns.get_level_values(0) else pd.Series(dtype=float)
        return df.squeeze() if not df.empty else pd.Series(dtype=float)

    b5 = c5["^CRSLDX"].dropna() if "^CRSLDX" in c5.columns else pd.Series(dtype=float)
    as_of = str(b5.index[-1])[:10] if len(b5) >= 1 else ""
    prev  = str(b5.index[-2])[:10] if len(b5) >= 2 else ""
    bm    = cm["^CRSLDX"].dropna() if "^CRSLDX" in cm.columns else pd.Series(dtype=float)
    month_start_actual = str(bm.index[0])[:10] if len(bm) >= 1 else month_start

    return {
        # Last-session — cover all displayed tickers (backtest holdings ∪ current rankings)
        "nifty500":        changes.get("^CRSLDX"),
        "ml_portfolio":    _port_avg(ml_tickers_now, changes),
        "mom_portfolio":   _port_avg(mom_tickers_now, changes),
        "ml_stocks":       {t.replace(".NS",""): changes.get(t)
                            for t in set(ml_tickers_now) | set(ml_tickers_mtd) if changes.get(t) is not None},
        "mom_stocks":      {t.replace(".NS",""): changes.get(t)
                            for t in set(mom_tickers_now) | set(mom_tickers_mtd) if changes.get(t) is not None},
        "as_of":           as_of,
        "prev_close_date": prev,
        # MTD (last-rebalance holdings = what was entered at start of month)
        "mtd_nifty500":      mtd.get("^CRSLDX"),
        "mtd_ml_portfolio":  _port_avg(ml_tickers_mtd, mtd),
        "mtd_mom_portfolio": _port_avg(mom_tickers_mtd, mtd),
        # Per-stock MTD covers both rebalance holdings AND current top-20 so every
        # row in the table gets a value regardless of which portfolio it came from.
        "mtd_ml_stocks":     {t.replace(".NS",""): mtd.get(t)
                              for t in set(ml_tickers_mtd) | set(ml_tickers_now) if mtd.get(t) is not None},
        "mtd_mom_stocks":    {t.replace(".NS",""): mtd.get(t)
                              for t in set(mom_tickers_mtd) | set(mom_tickers_now) if mtd.get(t) is not None},
        "month_start_date":  month_start_actual,
    }


@app.get("/api/portfolio/strategies")
def portfolio_strategies():
    """Current ML + Momentum strategy picks — no Kite required."""
    result = {"ml": None, "momentum": None, "next_rebalance": _next_rebalance()}

    if _ml_rank_cache:
        # Build rank/score/price lookup from today's rankings
        rank_map = {r["ticker"].replace(".NS",""): r
                    for r in _ml_rank_cache.get("rankings", [])}

        # Use last-rebalance holdings when backtest has been run so the table
        # matches the MTD aggregate (both reflect what was entered at month-start).
        # Fall back to today's top-20 if no backtest cache exists.
        if _ml_bt_cache:
            rh_ml = _ml_bt_cache.get("rebalance_history", [])
            held_tickers = [t.replace(".NS","") for t in rh_ml[-1].get("holdings",[])] if rh_ml else []
        else:
            held_tickers = [r["ticker"].replace(".NS","")
                            for r in _ml_rank_cache.get("rankings",[]) if r.get("in_top20")]

        holdings = []
        for i, sym in enumerate(held_tickers, 1):
            r = rank_map.get(sym, {})
            holdings.append({
                "symbol": sym,
                "ticker": sym + ".NS",
                "rank":   r.get("rank",  i),
                "score":  r.get("composite_score"),
                "price":  r.get("price"),
            })

        result["ml"] = {
            "as_of":    _ml_bt_cache["rebalance_history"][-1].get("date") if _ml_bt_cache and _ml_bt_cache.get("rebalance_history") else _ml_rank_cache.get("as_of"),
            "holdings": holdings,
        }

    if _mom_bt_cache:
        rh = _mom_bt_cache.get("rebalance_history", [])
        if rh:
            last = rh[-1]
            result["momentum"] = {
                "as_of":    last.get("date"),
                "holdings": [t.replace(".NS", "") for t in last.get("holdings", [])],
                "added":    [t.replace(".NS", "") for t in last.get("added", [])],
                "removed":  [t.replace(".NS", "") for t in last.get("removed", [])],
            }

    return result


@app.get("/api/portfolio/live")
def portfolio_live():
    status = kite_auth.kite_status()
    if not status.get("connected"):
        return {"connected": False, "reason": status.get("reason", "Kite not connected")}
    try:
        import kite_orders
        holdings_map = kite_orders.get_holdings()
        if not holdings_map:
            return {"connected": True, "holdings": [], "summary": {
                "total_invested": 0, "total_value": 0, "total_pnl": 0,
                "total_pnl_pct": 0, "day_change": 0, "count": 0,
            }, "next_rebalance": _next_rebalance(), "exit_candidates": []}

        # Build rank map from cached ML rankings
        rank_map, portfolio_set = {}, set()
        if _ml_rank_cache:
            for r in _ml_rank_cache.get("rankings", []):
                rank_map[r["ticker"].replace(".NS", "")] = r.get("rank", 999)
            portfolio_set = {t.replace(".NS", "") for t in _ml_rank_cache.get("portfolio", [])}

        holdings, total_invested, total_value, total_day_pnl = [], 0, 0, 0
        for sym, h in holdings_map.items():
            qty       = h.get("quantity", 0)
            avg_price = h.get("average_price", 0)
            ltp       = h.get("last_price", 0)
            day_chg   = h.get("day_change", 0)
            day_chg_p = h.get("day_change_percentage", 0)
            invested  = round(qty * avg_price, 2)
            value     = round(qty * ltp, 2)
            pnl_pct   = round((ltp - avg_price) / avg_price * 100, 2) if avg_price else 0
            total_invested  += invested
            total_value     += value
            total_day_pnl   += day_chg * qty
            holdings.append({
                "symbol": sym, "qty": qty,
                "avg_price": round(avg_price, 2), "ltp": round(ltp, 2),
                "invested": invested, "value": value,
                "pnl": round(value - invested, 2), "pnl_pct": pnl_pct,
                "day_change": round(day_chg, 2), "day_change_pct": round(day_chg_p, 2),
                "ml_rank": rank_map.get(sym),
                "in_portfolio": sym in portfolio_set if portfolio_set else None,
            })

        holdings.sort(key=lambda x: x["value"], reverse=True)
        total_pnl     = round(total_value - total_invested, 2)
        total_pnl_pct = round(total_pnl / total_invested * 100, 2) if total_invested else 0
        exit_candidates = [h["symbol"] for h in holdings
                           if portfolio_set and not h["in_portfolio"]]
        return {
            "connected": True,
            "holdings": holdings,
            "summary": {
                "total_invested": round(total_invested, 2),
                "total_value":    round(total_value, 2),
                "total_pnl":      total_pnl,
                "total_pnl_pct":  total_pnl_pct,
                "day_change":     round(total_day_pnl, 2),
                "count":          len(holdings),
            },
            "next_rebalance":  _next_rebalance(),
            "exit_candidates": exit_candidates,
        }
    except Exception as e:
        return {"connected": True, "error": str(e), "holdings": []}


# ── HA backtest ───────────────────────────────────────────────────────────────

_ha_bt_cache:   dict = {}
_ha_bt_running: bool = False


class HABacktestParams(BaseModel):
    from_date: str  = ""          # ISO date, defaults to 1 year ago
    to_date:   str  = ""          # ISO date, defaults to today
    mode:      str  = "futures"   # futures | options_sell | options_buy
    sl_pts:    float = 0
    sl_pct:    float = 0.5
    lots:      int   = 1


@app.post("/api/ha/backtest/run")
async def ha_backtest_run(params: HABacktestParams):
    global _ha_bt_running
    if _ha_bt_running:
        raise HTTPException(409, "HA backtest already running")
    import ha_backtest as ha

    from_date = date.fromisoformat(params.from_date) if params.from_date \
                else date.today().replace(year=date.today().year - 1)
    to_date   = date.fromisoformat(params.to_date) if params.to_date \
                else date.today()

    def _run():
        global _ha_bt_running, _ha_bt_cache
        _ha_bt_running = True
        try:
            result = ha.run_backtest(
                from_date=from_date, to_date=to_date,
                mode=params.mode, sl_pts=params.sl_pts,
                sl_pct=params.sl_pct, lots=params.lots,
            )
            _ha_bt_cache = result
            return result
        finally:
            _ha_bt_running = False

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    return result


@app.get("/api/ha/backtest/result")
def ha_backtest_result():
    if not _ha_bt_cache:
        raise HTTPException(404, "No HA backtest result — run one first")
    return _ha_bt_cache


@app.get("/api/ha/backtest/status")
def ha_backtest_status():
    return {"running": _ha_bt_running, "has_result": bool(_ha_bt_cache)}


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
    expiry = await loop.run_in_executor(_executor, lambda: pl.monthly_expiry(kite))
    opt    = await loop.run_in_executor(_executor,
                lambda: pl.option_details(kite, sig["active_signal"], sig["nifty_price"], expiry)) \
             if expiry and sig.get("active_signal", "FLAT") != "FLAT" else None
    pos    = await loop.run_in_executor(_executor, lambda: pl.nifty_positions(kite))

    return {**sig, "option": opt, "expiry": str(expiry) if expiry else None, "positions": pos}


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


# ── Order execution ───────────────────────────────────────────────────────────

class OrderParams(BaseModel):
    to_buy:             list  = []
    to_sell:            list  = []
    capital_per_stock:  float = 0


@app.post("/api/orders/preview")
async def orders_preview(params: OrderParams):
    import kite_orders as ko
    from kiteconnect.exceptions import KiteException
    status = kite_auth.kite_status()
    if not status.get("connected"):
        raise HTTPException(401, status.get("reason", "Kite not connected"))
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            _executor,
            lambda: ko.preview(params.to_buy, params.to_sell, params.capital_per_stock),
        )
    except KiteException as e:
        raise HTTPException(403, str(e))


@app.post("/api/orders/execute")
async def orders_execute(params: OrderParams):
    import kite_orders as ko
    from kiteconnect.exceptions import KiteException
    status = kite_auth.kite_status()
    if not status.get("connected"):
        raise HTTPException(401, status.get("reason", "Kite not connected"))
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            _executor,
            lambda: ko.execute(params.to_buy, params.to_sell, params.capital_per_stock),
        )
    except KiteException as e:
        raise HTTPException(403, str(e))


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return open(path).read() if os.path.exists(path) else "<h1>index.html not found</h1>"
