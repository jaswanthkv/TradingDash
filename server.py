"""
server.py — FastAPI backend. Single process serves UI + all API endpoints.

Endpoints:
  GET  /                      — TradeBoard dashboard (index.html)
  GET  /api/positions          — live P&L + 5MA status for all open positions
  GET  /api/portfolio          — header stats (invested %, P&L, risk, locked)
  GET  /api/chart/{ticker}     — OHLCV + EMA5 data for candlestick chart
  POST /api/scan               — run universe scan + Claude picks (~30-60s)
  POST /api/apply              — commit rebalance to open_positions.json
  GET  /api/stream             — SSE live price updates every 60s
  GET  /api/tradelog           — historical closed trades

  GET  /api/kite/status        — Kite connection status
  GET  /api/kite/login         — redirect to Kite OAuth login
  GET  /api/kite/callback      — OAuth callback (exchanges request_token)
  GET  /api/kite/margins       — available equity margin
  GET  /api/kite/quote/{ticker}— live quote from Kite
  GET  /api/kite/market        — market open/closed status
  POST /api/kite/order/buy     — place CNC buy (+ optional SL-M)
  POST /api/kite/order/sell    — place CNC sell
  GET  /api/kite/orders        — today's orders
  GET  /api/kite/order/{id}    — single order status
"""
import json, os, datetime, warnings, asyncio
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel

import engine
from config import CAPITAL, PRICE_CACHE_TTL, SSE_INTERVAL, PORT, KITE_API_KEY

# Kite is optional — app works without kiteconnect installed
try:
    import kite_auth, kite_orders
    _KITE = True
except ImportError:
    _KITE = False

app = FastAPI(title="TradeBoard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_executor = ThreadPoolExecutor(max_workers=4)

# ── Price cache ───────────────────────────────────────────────────────────────

_cache: dict[str, tuple[pd.DataFrame, datetime.datetime]] = {}


def get_df(ticker: str, days: int = 120) -> pd.DataFrame | None:
    entry = _cache.get(ticker)
    if entry and (datetime.datetime.now() - entry[1]).seconds < PRICE_CACHE_TTL:
        return entry[0]
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    try:
        df = yf.download(ticker, start=str(start), end=str(end),
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how="all")
        if not df.empty:
            _cache[ticker] = (df, datetime.datetime.now())
            return df
    except Exception:
        pass
    return None


def enrich(pos: dict, df: pd.DataFrame) -> dict:
    close        = df["Close"].squeeze()
    ema5         = close.ewm(span=5,  adjust=False).mean()
    ema10        = close.ewm(span=10, adjust=False).mean()
    sma20        = close.rolling(20).mean()
    cmp          = float(close.iloc[-1])
    last_ema5    = float(ema5.iloc[-1])
    last_ema10   = float(ema10.iloc[-1])

    below_ema10_now  = cmp < last_ema10
    below_ema10_prev = float(close.iloc[-2]) < float(ema10.iloc[-2])
    below_ema5_now   = cmp < last_ema5

    if below_ema10_now and below_ema10_prev:
        urgency, status = "exit",    "10MA Break"
    elif below_ema5_now:
        urgency, status = "warning", "Below 5MA"
    else:
        urgency, status = "safe",    "10MA Safe"

    days_above = 0
    for i in range(len(close) - 1, -1, -1):
        if float(close.iloc[i]) > float(ema10.iloc[i]):
            days_above += 1
        else:
            break

    entry      = pos["entry_price"]
    qty        = pos["qty"]
    atr        = engine.compute_atr(df)
    init_stop  = pos.get("initial_stop") or (entry - 2 * atr)
    risk       = max(entry - init_stop, 0.01)
    trail_stop = max(init_stop, last_ema10)
    today_open = float(df["Open"].squeeze().iloc[-1])
    holding    = (datetime.date.today() -
                  datetime.date.fromisoformat(pos["entry_date"])).days

    return {
        **pos,
        "cmp":           round(cmp, 2),
        "today_open":    round(today_open, 2),
        "ema5":          round(last_ema5, 2),
        "ema10":         round(float(ema10.iloc[-1]), 2),
        "sma20":         round(float(sma20.iloc[-1]), 2),
        "urgency":       urgency,
        "ma_status":     status,
        "days_above_5ma": days_above,
        "init_stop":     round(init_stop, 2),
        "trail_stop":    round(trail_stop, 2),
        "r_multiple":    round((cmp - entry) / risk, 2),
        "pnl_pct":       round((cmp - entry) / entry * 100, 2),
        "day_pnl":       round((cmp - today_open) * qty, 2),
        "day_pnl_pct":   round((cmp - today_open) / today_open * 100, 2),
        "total_pnl":     round((cmp - entry) * qty, 2),
        "current_value": round(cmp * qty, 2),
        "holding_days":  holding,
    }

# ── GET /api/positions ────────────────────────────────────────────────────────

def _placeholder(pos: dict) -> dict:
    """Fallback when price data is unavailable — show position with entry price."""
    entry   = pos["entry_price"]
    stop    = pos.get("initial_stop") or round(entry * 0.95, 2)
    holding = (datetime.date.today() -
               datetime.date.fromisoformat(pos["entry_date"])).days
    return {
        **pos,
        "cmp": entry, "today_open": entry,
        "ema5": 0, "ema10": 0, "sma20": 0,
        "urgency": "safe", "ma_status": "Loading…",
        "days_above_5ma": 0,
        "init_stop": stop, "trail_stop": stop,
        "r_multiple": 0, "pnl_pct": 0,
        "day_pnl": 0, "day_pnl_pct": 0, "total_pnl": 0,
        "current_value": round(entry * pos["qty"], 2),
        "holding_days": holding,
    }

@app.get("/api/positions")
def api_positions():
    out = []
    for pos in engine.load_positions():
        df = get_df(pos["ticker"])
        out.append(enrich(pos, df) if (df is not None and len(df) >= 10)
                   else _placeholder(pos))
    return out

# ── GET /api/portfolio ────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def api_portfolio():
    enriched = []
    for pos in engine.load_positions():
        df = get_df(pos["ticker"])
        enriched.append(enrich(pos, df) if (df is not None and len(df) >= 10)
                        else _placeholder(pos))
    if not enriched:
        return {
            "pct_invested": 0, "invested_inr": 0, "capital_inr": CAPITAL,
            "day_pnl": 0, "day_pnl_pct": 0, "total_pnl": 0, "total_pnl_pct": 0,
            "open_risk_inr": 0, "open_risk_pct": 0, "locked_profit": 0,
            "n_positions": 0, "exits_pending": 0,
        }

    invested      = sum(p["alloc_inr"]  for p in enriched)
    day_pnl       = sum(p["day_pnl"]    for p in enriched)
    total_pnl     = sum(p["total_pnl"]  for p in enriched)
    open_risk     = sum(max(p["cmp"] - p["init_stop"], 0) * p["qty"] for p in enriched)
    locked_profit = sum(
        max(p["trail_stop"] - p["entry_price"], 0) * p["qty"]
        for p in enriched if p["trail_stop"] > p["entry_price"]
    )
    exits_count = sum(1 for p in enriched if p["urgency"] == "exit")

    return {
        "pct_invested":   round(invested / CAPITAL * 100, 1),
        "invested_inr":   round(invested),
        "capital_inr":    CAPITAL,
        "day_pnl":        round(day_pnl),
        "day_pnl_pct":    round(day_pnl / invested * 100, 2) if invested else 0,
        "total_pnl":      round(total_pnl),
        "total_pnl_pct":  round(total_pnl / invested * 100, 2) if invested else 0,
        "open_risk_inr":  round(open_risk),
        "open_risk_pct":  round(open_risk / CAPITAL * 100, 1),
        "locked_profit":  round(locked_profit),
        "n_positions":    len(enriched),
        "exits_pending":  exits_count,
    }

# ── GET /api/chart/{ticker} ───────────────────────────────────────────────────

@app.get("/api/chart/{ticker}")
def api_chart(ticker: str, days: int = 60):
    df = get_df(ticker, days=days + 30)
    if df is None:
        return {"candles": [], "ema5": []}
    df    = df.tail(days)
    close = df["Close"].squeeze()
    ema5  = close.ewm(span=5,  adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()

    candles = [{"time": ts.strftime("%Y-%m-%d"),
                "open":  round(float(r["Open"]),  2),
                "high":  round(float(r["High"]),  2),
                "low":   round(float(r["Low"]),   2),
                "close": round(float(r["Close"]), 2)}
               for ts, r in df.iterrows()]

    ema5_data  = [{"time": ts.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
                  for ts, v in ema5.items()]
    ema10_data = [{"time": ts.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
                  for ts, v in ema10.items()]

    return {"candles": candles, "ema5": ema5_data, "ema10": ema10_data}

# ── POST /api/scan ────────────────────────────────────────────────────────────

@app.post("/api/scan")
async def api_scan():
    """Run full universe scan + Claude. Takes 30–60 seconds."""
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, engine.run_scan)
    _cache.clear()   # fresh prices after scan
    return result

# ── POST /api/apply ───────────────────────────────────────────────────────────

class ApplyRequest(BaseModel):
    exit_tickers:    list[str]
    approved_picks:  list[dict]
    exit_reasons:    dict[str, str] = {}

@app.post("/api/apply")
def api_apply(req: ApplyRequest):
    """Commit rebalance: remove exits, add approved picks."""
    updated = engine.apply_rebalance(req.exit_tickers, req.approved_picks, req.exit_reasons)
    _cache.clear()
    return updated

@app.get("/api/tradelog")
def api_tradelog():
    return engine.load_trade_log()

# ── GET /api/stream (SSE live prices) ────────────────────────────────────────

def _fetch_price_updates() -> dict:
    positions = engine.load_positions()
    updates   = {}
    for pos in positions:
        df = get_df(pos["ticker"], days=5)
        if df is None or len(df) < 2:
            continue
        cmp   = float(df["Close"].squeeze().iloc[-1])
        opn   = float(df["Open"].squeeze().iloc[-1])
        entry = pos["entry_price"]
        qty   = pos["qty"]
        updates[pos["ticker"]] = {
            "cmp":         round(cmp, 2),
            "pnl_pct":     round((cmp - entry) / entry * 100, 2),
            "day_pnl_pct": round((cmp - opn) / opn * 100, 2),
            "total_pnl":   round((cmp - entry) * qty, 2),
        }
    return updates


async def _sse_generator():
    loop = asyncio.get_event_loop()
    while True:
        try:
            data = await loop.run_in_executor(_executor, _fetch_price_updates)
            yield f"data: {json.dumps(data)}\n\n"
        except Exception:
            yield "data: {}\n\n"
        await asyncio.sleep(SSE_INTERVAL)


@app.get("/api/stream")
async def api_stream():
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Kite Connect ─────────────────────────────────────────────────────────────

def _kite_guard():
    if not _KITE:
        raise HTTPException(503, "kiteconnect not installed. Run: pip install kiteconnect")
    if not KITE_API_KEY:
        raise HTTPException(503, "KITE_API_KEY not set in config.py")

@app.get("/api/kite/status")
def api_kite_status():
    if not _KITE or not KITE_API_KEY:
        return {"connected": False, "reason": "Kite not configured"}
    return kite_auth.kite_status()

@app.get("/api/kite/login")
def api_kite_login():
    _kite_guard()
    return RedirectResponse(kite_auth.get_login_url())

@app.get("/api/kite/callback")
def api_kite_callback(request_token: str = "", status: str = ""):
    _kite_guard()
    if status != "success" or not request_token:
        return RedirectResponse("/?kite=failed")
    try:
        kite_auth.complete_login(request_token)
        return RedirectResponse("/?kite=connected")
    except Exception as exc:
        return RedirectResponse(f"/?kite=error")

@app.get("/api/kite/market")
def api_kite_market():
    if not _KITE:
        return {"open": False, "message": "Kite not configured"}
    return kite_orders.market_status()

@app.get("/api/kite/margins")
def api_kite_margins():
    _kite_guard()
    try:
        return kite_orders.get_margins()
    except RuntimeError as exc:
        raise HTTPException(401, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@app.get("/api/kite/quote/{ticker}")
def api_kite_quote(ticker: str):
    _kite_guard()
    try:
        return kite_orders.get_quote(ticker)
    except RuntimeError as exc:
        raise HTTPException(401, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

class BuyRequest(BaseModel):
    ticker:      str
    qty:         int
    order_type:  str   = "MARKET"
    limit_price: float = 0
    stop_loss:   float = 0
    place_sl:    bool  = True
    dry_run:     bool  = False

@app.post("/api/kite/order/buy")
def api_kite_buy(req: BuyRequest):
    _kite_guard()
    try:
        return kite_orders.place_buy(
            req.ticker, req.qty, req.order_type,
            req.limit_price, req.stop_loss, req.place_sl, req.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(401, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

class SellRequest(BaseModel):
    ticker:      str
    qty:         int
    order_type:  str   = "MARKET"
    limit_price: float = 0
    dry_run:     bool  = False

@app.post("/api/kite/order/sell")
def api_kite_sell(req: SellRequest):
    _kite_guard()
    try:
        return kite_orders.place_sell(
            req.ticker, req.qty, req.order_type,
            req.limit_price, req.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(401, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

@app.get("/api/kite/holdings")
def api_kite_holdings():
    if not _KITE:
        return []
    try:
        return kite_orders.get_holdings()
    except Exception:
        return []

@app.get("/api/kite/orders")
def api_kite_orders():
    if not _KITE:
        return []
    try:
        return kite_orders.get_today_orders()
    except Exception:
        return []

@app.get("/api/kite/order/{order_id}")
def api_kite_order_status(order_id: str):
    _kite_guard()
    try:
        return kite_orders.get_order_status(order_id)
    except RuntimeError as exc:
        raise HTTPException(401, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

# ── Options endpoints ─────────────────────────────────────────────────────────

@app.get("/api/options/expiries/{symbol}")
def api_opt_expiries(symbol: str):
    _kite_guard()
    try:
        import options as opt
        return opt.get_expiries(symbol.upper())
    except Exception as exc:
        raise HTTPException(500, str(exc))

@app.get("/api/options/chain/{symbol}/{expiry}")
def api_opt_chain(symbol: str, expiry: str):
    _kite_guard()
    try:
        import options as opt
        return opt.get_chain(symbol.upper(), expiry)
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── GET /api/health ───────────────────────────────────────────────────────────

@app.get("/api/health")
def api_health():
    import importlib, config
    importlib.reload(config)
    return {
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "kite_key":  bool(config.KITE_API_KEY),
        "kite_secret": bool(config.KITE_API_SECRET),
        "ready":     bool(config.ANTHROPIC_API_KEY),
    }

# ── POST /api/setup ───────────────────────────────────────────────────────────

class SetupRequest(BaseModel):
    anthropic_key: str
    kite_key:      str = ""
    kite_secret:   str = ""
    capital:       int = 0

@app.post("/api/setup")
def api_setup(req: SetupRequest):
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    lines = []
    if req.anthropic_key.strip():
        lines.append(f"ANTHROPIC_API_KEY={req.anthropic_key.strip()}")
    if req.kite_key.strip():
        lines.append(f"KITE_API_KEY={req.kite_key.strip()}")
    if req.kite_secret.strip():
        lines.append(f"KITE_API_SECRET={req.kite_secret.strip()}")
    if req.capital > 0:
        lines.append(f"CAPITAL={req.capital}")
    with open(env_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    # apply to running process immediately
    for line in lines:
        k, v = line.split("=", 1)
        os.environ[k] = v
    import importlib, config, engine as eng
    importlib.reload(config)
    importlib.reload(eng)
    return {"status": "saved"}

# ── Backtest ──────────────────────────────────────────────────────────────────

_BT_CACHE_FILE    = os.path.join(os.path.dirname(__file__), ".bt_cache_india.json")
_BT_US_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".bt_cache_us.json")

def _load_cache(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(path: str, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

_backtest_cache: dict = _load_cache(_BT_CACHE_FILE)
_backtest_running = False


class BacktestParams(BaseModel):
    m: int     = 8
    x: int     = 3
    years: int = 10


@app.post("/api/backtest/run")
async def run_backtest_endpoint(params: BacktestParams):
    """
    Launch a backtest in the thread pool.
    Results cached until next run.
    """
    global _backtest_running
    if _backtest_running:
        raise HTTPException(status_code=409, detail="Backtest already running")

    import backtest as bt

    def _run():
        global _backtest_running, _backtest_cache
        _backtest_running = True
        try:
            result = bt.run_backtest(m=params.m, x=params.x, years=params.years)
            _backtest_cache = result
            _save_cache(_BT_CACHE_FILE, result)
            return result
        finally:
            _backtest_running = False

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    return result


@app.get("/api/backtest/result")
def get_backtest_result():
    if not _backtest_cache:
        raise HTTPException(status_code=404, detail="No backtest result yet — run one first")
    return _backtest_cache


@app.get("/api/backtest/status")
def get_backtest_status():
    return {
        "running": _backtest_running,
        "has_result": bool(_backtest_cache),
    }


@app.get("/api/backtest/current-pf")
async def get_current_pf(m: int = 8, x: int = 3, years: int = 10):
    """Current month's portfolio with MTD and last-day performance."""
    import backtest as bt
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, lambda: bt.get_current_pf(m=m, x=x, years=years)
    )
    return result


@app.get("/api/backtest/rebalance-plan")
async def rebalance_plan(
    m: int = 8, x: int = 3, years: int = 10, capital: float = 0,
    exclude: str = "",   # comma-separated symbols, e.g. "ANTHEM,NIFTYBEES"
):
    """Compare pflio target vs Kite demat. Returns SELL/BUY/HOLD actions."""
    if not _KITE:
        raise HTTPException(status_code=503, detail="Kite not available")
    import backtest as bt
    exclude_list = [s.strip() for s in exclude.split(",") if s.strip()]
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: bt.get_rebalance_plan(m=m, x=x, years=years, capital=capital,
                                      exclude_tickers=exclude_list)
    )
    return result


class RebalanceExecuteBody(BaseModel):
    sells:   list[dict]
    buys:    list[dict]
    dry_run: bool = True


@app.post("/api/backtest/rebalance-execute")
async def rebalance_execute(body: RebalanceExecuteBody):
    """Place CNC sell then buy orders for the rebalance."""
    if not _KITE:
        raise HTTPException(status_code=503, detail="Kite not available")
    import backtest as bt
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: bt.execute_rebalance(body.sells, body.buys, dry_run=body.dry_run)
    )
    return result




# ── US Backtest (NASDAQ 100) ──────────────────────────────────────────────────

_us_backtest_cache: dict = _load_cache(_BT_US_CACHE_FILE)
_us_backtest_running = False


@app.post("/api/backtest-us/run")
async def run_us_backtest(params: BacktestParams):
    global _us_backtest_running
    if _us_backtest_running:
        raise HTTPException(status_code=409, detail="US backtest already running")
    import backtest_us as bt_us

    def _run():
        global _us_backtest_running, _us_backtest_cache
        _us_backtest_running = True
        try:
            result = bt_us.run_backtest(m=params.m, x=params.x, years=params.years)
            _us_backtest_cache = result
            _save_cache(_BT_US_CACHE_FILE, result)
            return result
        finally:
            _us_backtest_running = False

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run)
    return result


@app.get("/api/backtest-us/result")
def get_us_backtest_result():
    if not _us_backtest_cache:
        raise HTTPException(status_code=404, detail="No US backtest result — run one first")
    return _us_backtest_cache


@app.get("/api/backtest-us/status")
def get_us_backtest_status():
    return {"running": _us_backtest_running, "has_result": bool(_us_backtest_cache)}


@app.get("/api/backtest-us/current-pf")
async def get_us_current_pf(m: int = 10, x: int = 3, years: int = 10):
    import backtest_us as bt_us
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, lambda: bt_us.get_current_pf(m=m, x=x, years=years)
    )
    return result


# ── Options Scorecard ─────────────────────────────────────────────────────────

@app.get("/api/options/scorecard")
def api_options_scorecard(symbol: str = "NIFTY", expiry1: str = "", expiry2: str = ""):
    _kite_guard()
    try:
        from options_scorecard import run_scorecard
        result = run_scorecard(
            symbol=symbol,
            expiry1=expiry1 or None,
            expiry2=expiry2 or None,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Weekly Report Card ────────────────────────────────────────────────────────

@app.get("/api/report/data")
async def api_report_data(market: str = "india"):
    """Compile all data for the weekly social media report card."""
    import report as rpt
    if market == "us":
        import backtest_us as bt
        bt_result = _us_backtest_cache
        loop   = asyncio.get_event_loop()
        pf     = await loop.run_in_executor(_executor, lambda: bt.get_current_pf())
    else:
        import backtest as bt
        bt_result = _backtest_cache
        loop   = asyncio.get_event_loop()
        pf     = await loop.run_in_executor(_executor, lambda: bt.get_current_pf())

    data = await loop.run_in_executor(
        _executor, lambda: rpt.generate_report_data(market, bt_result, pf)
    )
    return data


@app.get("/report", response_class=HTMLResponse)
def report_page(market: str = "india"):
    path = os.path.join(os.path.dirname(__file__), "report.html")
    return open(path).read() if os.path.exists(path) else "<h1>report.html not found</h1>"


# ── GET / — serve dashboard ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return open(path).read() if os.path.exists(path) else "<h1>index.html not found</h1>"
