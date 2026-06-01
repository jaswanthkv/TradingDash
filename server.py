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

  POST /api/ipo-screen/run            — launch IPO breakout screen
  GET  /api/ipo-screen/result         — latest IPO screen results
  GET  /api/ipo-screen/status         — running / has_result / progress message

  GET  /api/pulse/auto                — auto-execute status + trade log
  POST /api/pulse/auto                — enable / disable auto-execute
"""
import asyncio
import calendar
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as _dt
from zoneinfo import ZoneInfo

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

_ipo_cache:    dict = {}
_ipo_running:  bool = False
_ipo_progress: str  = ""

# ── Pulse auto-execute state ──────────────────────────────────────────────────
_pulse_auto_enabled:  bool        = False
_pulse_auto_signal:   str         = "FLAT"   # last signal we ACTED on
_pulse_auto_log:      list        = []
_pulse_auto_lots:     int         = 1
_pulse_auto_thread:   threading.Thread | None = None
_IST                              = ZoneInfo("Asia/Kolkata")


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
        raise HTTPException(409, "Trend Breakout backtest already running")
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
        raise HTTPException(404, "No Trend Breakout backtest result — run one first")
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
    """Daily % change for Minervini + Momentum strategy portfolios vs Nifty 500."""
    import yfinance as yf
    import numpy as np

    # Last rebalance holdings for each strategy
    min_tickers_now, mom_tickers_now = [], []
    if _min_bt_cache:
        rh = _min_bt_cache.get("rebalance_history", [])
        if rh:
            min_tickers_now = rh[-1].get("holdings", [])
    if _mom_bt_cache:
        rh = _mom_bt_cache.get("rebalance_history", [])
        if rh:
            mom_tickers_now = rh[-1].get("holdings", [])

    min_tickers_mtd = min_tickers_now
    mom_tickers_mtd = mom_tickers_now

    all_tickers = list(set(min_tickers_now + mom_tickers_now +
                           min_tickers_mtd + mom_tickers_mtd + ["^CRSLDX"]))
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
        "nifty500":        changes.get("^CRSLDX"),
        "min_portfolio":   _port_avg(min_tickers_now, changes),
        "mom_portfolio":   _port_avg(mom_tickers_now, changes),
        "min_stocks":      {t.replace(".NS",""): changes.get(t)
                            for t in set(min_tickers_now) | set(min_tickers_mtd) if changes.get(t) is not None},
        "mom_stocks":      {t.replace(".NS",""): changes.get(t)
                            for t in set(mom_tickers_now) | set(mom_tickers_mtd) if changes.get(t) is not None},
        "as_of":           as_of,
        "prev_close_date": prev,
        "mtd_nifty500":       mtd.get("^CRSLDX"),
        "mtd_min_portfolio":  _port_avg(min_tickers_mtd, mtd),
        "mtd_mom_portfolio":  _port_avg(mom_tickers_mtd, mtd),
        "mtd_min_stocks":     {t.replace(".NS",""): mtd.get(t)
                               for t in set(min_tickers_mtd) | set(min_tickers_now) if mtd.get(t) is not None},
        "mtd_mom_stocks":     {t.replace(".NS",""): mtd.get(t)
                               for t in set(mom_tickers_mtd) | set(mom_tickers_now) if mtd.get(t) is not None},
        "month_start_date":   month_start_actual,
    }


@app.get("/api/portfolio/strategies")
def portfolio_strategies():
    """Current Minervini + Momentum strategy picks — no Kite required."""
    result = {"minervini": None, "momentum": None, "next_rebalance": _next_rebalance()}

    if _min_bt_cache:
        rh_min = _min_bt_cache.get("rebalance_history", [])
        if rh_min:
            last_min = rh_min[-1]
            # Build RS rating and price lookup from current_screen
            screen_map = {s["symbol"]: s for s in _min_bt_cache.get("current_screen", [])}
            holdings = []
            for sym_ns in last_min.get("holdings", []):
                sym = sym_ns.replace(".NS", "")
                sc = screen_map.get(sym, {})
                holdings.append({
                    "symbol":    sym,
                    "ticker":    sym_ns,
                    "rs_rating": sc.get("rs_rating"),
                    "price":     sc.get("price"),
                })
            result["minervini"] = {
                "as_of":    last_min.get("date"),
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


# ── IPO Breakout Screen ───────────────────────────────────────────────────────

class IPOScreenParams(BaseModel):
    years: int = 2


@app.post("/api/ipo-screen/run")
async def ipo_screen_run(params: IPOScreenParams):
    global _ipo_running, _ipo_cache, _ipo_progress
    if _ipo_running:
        raise HTTPException(409, "IPO screen already running")
    _ipo_running  = True
    _ipo_progress = "Starting…"

    def _run():
        global _ipo_running, _ipo_cache, _ipo_progress
        try:
            from ipo_screen import run_ipo_screen
            def _prog(msg):
                global _ipo_progress
                _ipo_progress = msg
            _ipo_cache = run_ipo_screen(years=params.years, on_progress=_prog)
        except Exception as exc:
            _ipo_progress = f"Error: {exc}"
        finally:
            _ipo_running = False

    _executor.submit(_run)
    return {"started": True}


@app.get("/api/ipo-screen/result")
def ipo_screen_result():
    return _ipo_cache or {}


@app.get("/api/ipo-screen/status")
def ipo_screen_status():
    return {
        "running":    _ipo_running,
        "has_result": bool(_ipo_cache),
        "progress":   _ipo_progress,
    }


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/report", response_class=HTMLResponse)
def monthly_report():
    """Self-contained monthly performance report card — screenshot and share."""
    from datetime import date as _date

    today = _date.today()
    month_name = today.strftime("%B %Y")
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    def _latest_month(rows: list[dict]) -> tuple[float | None, float | None]:
        """Return (current_month_pct, ytd_pct) from a monthly_returns grid."""
        if not rows:
            return None, None
        yr_row = next((r for r in reversed(rows) if r.get("year") == today.year), None)
        if not yr_row:
            yr_row = rows[-1]
        # find latest non-None month
        cur = None
        for m in reversed(MONTHS):
            if yr_row.get(m) is not None:
                cur = yr_row[m]
                break
        ytd = yr_row.get("Annual")
        return cur, ytd

    def _card(label: str, cache: dict | None, color: str) -> dict:
        if not cache:
            return {"label": label, "color": color, "available": False}
        kpi = cache.get("strategy_kpi", {})
        bkpi = cache.get("benchmark_kpi", {})
        cur, ytd = _latest_month(cache.get("monthly_returns", []))
        bcur, bytd = _latest_month(cache.get("monthly_bench", []))
        run_at = cache.get("run_at", "")[:10]
        return {
            "label":     label,
            "color":     color,
            "available": True,
            "cagr":      kpi.get("cagr_pct"),
            "sharpe":    kpi.get("sharpe"),
            "max_dd":    kpi.get("max_dd_pct"),
            "win_rate":  kpi.get("win_rate_pct"),
            "months":    kpi.get("total_months"),
            "cur":       cur,
            "ytd":       ytd,
            "bcur":      bcur,
            "bytd":      bytd,
            "b_cagr":    bkpi.get("cagr_pct"),
            "run_at":    run_at,
        }

    cards = [
        _card("Multi-Factor Momentum", _mom_bt_cache, "#6366f1"),
        _card("Trend Breakout",        _min_bt_cache, "#f59e0b"),
    ]

    def _pct(v, decimals=1):
        if v is None: return "—"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.{decimals}f}%"

    def _clr(v):
        if v is None: return "#94a3b8"
        return "#16a34a" if v >= 0 else "#dc2626"

    def _num(v, decimals=2):
        if v is None: return "—"
        return f"{v:.{decimals}f}"

    def _strategy_block(c: dict) -> str:
        if not c["available"]:
            return f'<div class="strategy-card" style="border-left:4px solid {c["color"]};opacity:.5"><div class="s-label" style="color:{c["color"]}">{c["label"]}</div><div style="color:#94a3b8;font-size:13px;padding:20px 0">No backtest data — run backtest first.</div></div>'

        alpha_cur  = None if (c["cur"]  is None or c["bcur"]  is None) else round(c["cur"]  - c["bcur"],  1)
        alpha_ytd  = None if (c["ytd"]  is None or c["bytd"]  is None) else round(c["ytd"]  - c["bytd"],  1)
        alpha_cagr = None if (c["cagr"] is None or c["b_cagr"] is None) else round(c["cagr"] - c["b_cagr"], 1)

        return f"""
        <div class="strategy-card" style="border-left:4px solid {c['color']}">
          <div class="s-label" style="color:{c['color']}">{c['label']}</div>
          <div class="period-row">
            <div class="period-block">
              <div class="period-name">This Month</div>
              <div class="period-val" style="color:{_clr(c['cur'])}">{_pct(c['cur'])}</div>
              <div class="period-bench">Nifty 500 {_pct(c['bcur'])} &nbsp;·&nbsp; <span style="color:{_clr(alpha_cur)}">α {_pct(alpha_cur)}</span></div>
            </div>
            <div class="period-divider"></div>
            <div class="period-block">
              <div class="period-name">Year to Date</div>
              <div class="period-val" style="color:{_clr(c['ytd'])}">{_pct(c['ytd'])}</div>
              <div class="period-bench">Nifty 500 {_pct(c['bytd'])} &nbsp;·&nbsp; <span style="color:{_clr(alpha_ytd)}">α {_pct(alpha_ytd)}</span></div>
            </div>
            <div class="period-divider"></div>
            <div class="period-block">
              <div class="period-name">CAGR (inception)</div>
              <div class="period-val" style="color:{_clr(c['cagr'])}">{_pct(c['cagr'])}</div>
              <div class="period-bench">Nifty 500 {_pct(c['b_cagr'])} &nbsp;·&nbsp; <span style="color:{_clr(alpha_cagr)}">α {_pct(alpha_cagr)}</span></div>
            </div>
          </div>
          <div class="kpi-row">
            <div class="kpi-block"><div class="kpi-val">{_num(c['sharpe'])}</div><div class="kpi-name">Sharpe</div></div>
            <div class="kpi-block"><div class="kpi-val" style="color:#dc2626">{_pct(c['max_dd'])}</div><div class="kpi-name">Max Drawdown</div></div>
            <div class="kpi-block"><div class="kpi-val">{_num(c['win_rate'], 0)}%</div><div class="kpi-name">Win Rate</div></div>
            <div class="kpi-block"><div class="kpi-val">{c['months'] or '—'}</div><div class="kpi-name">Months Tested</div></div>
          </div>
        </div>"""

    blocks = "\n".join(_strategy_block(c) for c in cards)
    run_dates = [c["run_at"] for c in cards if c.get("run_at")]
    footer_date = max(run_dates) if run_dates else str(today)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>QuantDesk — {month_name} Report</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#f8fafc;color:#1e293b;
  min-height:100vh;display:flex;align-items:center;justify-content:center;padding:32px 16px}}
.card{{background:#fff;border-radius:20px;box-shadow:0 4px 40px rgba(0,0,0,.10);
  width:100%;max-width:680px;overflow:hidden}}
.card-header{{background:linear-gradient(135deg,#1e293b 0%,#334155 100%);
  padding:32px 36px 28px;color:#fff}}
.badge{{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;
  background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
  border-radius:20px;font-size:11px;font-weight:600;color:rgba(255,255,255,.8);margin-bottom:14px}}
.report-title{{font-size:26px;font-weight:800;letter-spacing:-.5px;margin-bottom:4px}}
.report-title span{{color:#818cf8}}
.report-sub{{font-size:13px;color:rgba(255,255,255,.6);margin-top:6px}}
.card-body{{padding:28px 36px 32px;display:flex;flex-direction:column;gap:20px}}
.strategy-card{{background:#f8fafc;border-radius:12px;padding:20px 22px;border:1px solid #e2e8f0}}
.s-label{{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px}}
.period-row{{display:grid;grid-template-columns:1fr 1px 1fr 1px 1fr;gap:0;margin-bottom:16px}}
.period-divider{{background:#e2e8f0;margin:0 12px}}
.period-block{{padding:0 4px}}
.period-name{{font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px}}
.period-val{{font-size:22px;font-weight:800;letter-spacing:-.5px;margin-bottom:3px}}
.period-bench{{font-size:10px;color:#94a3b8}}
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #e2e8f0;padding-top:14px;gap:8px}}
.kpi-block{{text-align:center}}
.kpi-val{{font-size:16px;font-weight:700;color:#1e293b;margin-bottom:2px}}
.kpi-name{{font-size:10px;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:.06em}}
.card-footer{{border-top:1px solid #e2e8f0;padding:14px 36px;display:flex;
  align-items:center;justify-content:space-between;background:#f8fafc}}
.footer-brand{{font-size:13px;font-weight:700;color:#1e293b}}
.footer-brand span{{color:#6366f1}}
.footer-meta{{font-size:11px;color:#94a3b8}}
.disclaimer{{font-size:10px;color:#94a3b8;padding:0 36px 20px;line-height:1.6}}
@media print{{body{{background:#fff;padding:0}}}}
</style>
</head>
<body>
<div class="card">
  <div class="card-header">
    <div class="badge">NSE India · Systematic Equity · Rules-Based</div>
    <div class="report-title">Quant<span>Desk</span> — {month_name}</div>
    <div class="report-sub">Monthly performance update across systematic strategies</div>
  </div>
  <div class="card-body">
    {blocks}
  </div>
  <div class="card-footer">
    <div class="footer-brand">Quant<span>Desk</span></div>
    <div class="footer-meta">As of {footer_date} · Backtested returns · Not investment advice</div>
  </div>
  <div class="disclaimer">
    All returns are from walk-forward backtests on NSE-listed equities. Past performance does not guarantee future results.
    Transaction costs included. Returns shown in INR.
  </div>
</div>
</body>
</html>"""

    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return open(path).read() if os.path.exists(path) else "<h1>index.html not found</h1>"
