"""
backtest.py — Monthly rebalancing backtest for Nifty Microcap 250 universe.

Strategy (pflio):
  - Each month, hold top `m` stocks ranked by prior-month return.
  - Remove `x` worst performers every month and replace with next best.
  - Benchmark: Nifty 50 (^NSEI) buy-and-hold.

KPIs: CAGR, Sharpe ratio (rf=6% annualised for India), Max Drawdown.
"""
import csv, logging, os
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import UNIVERSE_CSV

logger = logging.getLogger(__name__)

_BENCHMARK  = "^NSEI"
_RISK_FREE  = 0.06          # 6% annualised (India short-term)
_DEFAULT_M  = 8             # portfolio size (matches current swing book)
_DEFAULT_X  = 3             # stocks removed / replaced each month
_YEARS      = 10            # lookback years


# ── universe ──────────────────────────────────────────────────────────────────

def load_universe(csv_path: str = UNIVERSE_CSV) -> list[str]:
    """Return list of yfinance tickers (SYMBOL.NS) from the NSE watchlist CSV."""
    tickers = []
    try:
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)  # header row
            next(reader)  # index aggregate row (NIFTY MICROCAP 250)
            for row in reader:
                sym = row[0].strip() if row else ""
                if sym:
                    tickers.append(sym + ".NS")
    except FileNotFoundError:
        logger.error("Universe CSV not found: %s", csv_path)
    return tickers


# ── data download ─────────────────────────────────────────────────────────────

def download_monthly(tickers: list[str], years: int = _YEARS) -> pd.DataFrame:
    """
    Download adjusted monthly close prices for tickers + benchmark.
    Returns a DataFrame indexed by month-end dates, columns = tickers + benchmark.
    Drops columns with >40% missing data.
    Always excludes the current incomplete month so portfolio selection
    is stable throughout the month (based only on prior completed months).
    """
    period = f"{years}y"
    all_tickers = tickers + [_BENCHMARK]
    logger.info("Downloading %d tickers (%s period) …", len(all_tickers), period)

    raw = yf.download(
        all_tickers,
        period=period,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes.index = pd.to_datetime(closes.index).tz_localize(None)

    # Drop the current incomplete month — only use rows where the bar is a full month
    today = date.today()
    month_start = pd.Timestamp(today.replace(day=1))
    closes = closes[closes.index < month_start]

    # Drop tickers with too many NaNs
    threshold = 0.6 * len(closes)
    closes = closes.dropna(axis=1, thresh=int(threshold))
    logger.info("Data shape after quality filter: %s", closes.shape)
    return closes


# ── KPIs ──────────────────────────────────────────────────────────────────────

def cagr(returns: pd.Series) -> float:
    """Annualised compounded return from a monthly return series."""
    cumulative = (1 + returns).prod()
    n_years    = len(returns) / 12
    return float(cumulative ** (1 / n_years) - 1) if n_years > 0 else 0.0


def sharpe(returns: pd.Series, rf: float = _RISK_FREE) -> float:
    """Annualised Sharpe ratio from monthly returns."""
    monthly_rf = (1 + rf) ** (1 / 12) - 1
    excess     = returns - monthly_rf
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(12))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    wealth = (1 + returns).cumprod()
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    return float(dd.min())


def compute_kpis(returns: pd.Series, label: str) -> dict:
    return {
        "label":        label,
        "cagr_pct":     round(cagr(returns) * 100, 2),
        "sharpe":       round(sharpe(returns), 2),
        "max_dd_pct":   round(max_drawdown(returns) * 100, 2),
        "total_months": int(len(returns)),
    }


# ── pflio core (shared by backtest + current-pf) ──────────────────────────────

def _run_pflio(monthly_ret: pd.DataFrame, m: int, x: int):
    """
    Run the momentum rebalancing loop.
    Returns (port_returns: dict[date, float], final_portfolio: list[str]).
    """
    portfolio    = []
    port_returns = {}

    for i, dt in enumerate(monthly_ret.index):
        if i == 0:
            avail     = monthly_ret.loc[dt].dropna().sort_values(ascending=False)
            portfolio = list(dict.fromkeys(avail.head(m).index))[:m]
            continue

        held_rets = monthly_ret.loc[dt, portfolio].dropna()
        if len(held_rets) == 0:
            port_returns[dt] = 0.0
            continue

        port_returns[dt] = float(held_rets.mean())

        all_avail   = monthly_ret.loc[dt].dropna().sort_values(ascending=False)
        to_remove   = set(held_rets.sort_values(ascending=True).head(x).index)
        kept        = [p for p in portfolio if p not in to_remove]
        kept_set    = set(kept)
        candidates  = [t for t in all_avail.index if t not in kept_set]
        portfolio   = list(dict.fromkeys(kept + candidates[:x]))[:m]

    return port_returns, portfolio


def pflio(price_df: pd.DataFrame, m: int = _DEFAULT_M, x: int = _DEFAULT_X) -> pd.Series:
    """Equal-weight monthly momentum portfolio. Returns monthly return Series."""
    stock_cols  = [c for c in price_df.columns if c != _BENCHMARK]
    monthly_ret = price_df[stock_cols].pct_change().dropna(how="all")
    port_returns, _ = _run_pflio(monthly_ret, m, x)
    return pd.Series(port_returns, name="Strategy")


# ── current portfolio snapshot ────────────────────────────────────────────────

def get_current_pf(
    m: int = _DEFAULT_M,
    x: int = _DEFAULT_X,
    years: int = _YEARS,
    csv_path: str = UNIVERSE_CSV,
) -> dict:
    """
    Return the current holdings after the last monthly rebalance, with:
      - MTD return per stock vs Nifty 50 MTD
      - Last trading day return per stock vs Nifty 50 last-day
    """
    tickers  = load_universe(csv_path)
    price_df = download_monthly(tickers, years)

    stock_cols  = [c for c in price_df.columns if c != _BENCHMARK]
    monthly_ret = price_df[stock_cols].pct_change().dropna(how="all")
    _, holdings = _run_pflio(monthly_ret, m, x)

    # Daily data: from Dec 20 of prior year to capture YTD base + week base
    today       = date.today()
    month_start = today.replace(day=1)
    year_start  = today.replace(month=1, day=1)
    fetch_from  = (year_start - timedelta(days=15)).strftime("%Y-%m-%d")
    # Week base = Monday of current ISO week
    week_start  = today - timedelta(days=today.weekday())

    _BENCHMARK2 = "MOSMALL250.NS"   # Nifty Smallcap 250 ETF
    daily_tickers = holdings + [_BENCHMARK, _BENCHMARK2]
    daily_raw = yf.download(
        daily_tickers,
        start=fetch_from,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    daily = (daily_raw["Close"] if isinstance(daily_raw.columns, pd.MultiIndex) else daily_raw)
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    daily = daily.dropna(how="all")

    if daily.empty:
        return {"holdings": holdings, "stocks": [], "portfolio_mtd": None,
                "benchmark_mtd": None, "portfolio_last_day": None,
                "benchmark_last_day": None, "as_of": str(today)}

    month_start_ts = pd.Timestamp(month_start)
    year_start_ts  = pd.Timestamp(year_start)
    week_start_ts  = pd.Timestamp(week_start)

    # MTD base = last trading close BEFORE the 1st of this month
    prior_month = daily[daily.index < month_start_ts]
    mtd_base_row = prior_month.iloc[-1] if not prior_month.empty else daily.iloc[0]

    # YTD base = last trading close BEFORE Jan 1 of this year (i.e. Dec 31 prev year)
    prior_year   = daily[daily.index < year_start_ts]
    ytd_base_row = prior_year.iloc[-1] if not prior_year.empty else daily.iloc[0]

    # Week base = last trading close BEFORE Monday of this week
    prior_week   = daily[daily.index < week_start_ts]
    wk_base_row  = prior_week.iloc[-1] if not prior_week.empty else daily.iloc[0]

    last_row = daily.iloc[-1]
    prev_row = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]

    def _pct(now, base):
        try:
            if pd.isna(base) or float(base) == 0:
                return None
            return round((float(now) / float(base) - 1) * 100, 2)
        except Exception:
            return None

    stocks = []
    pf_mtd_vals, pf_ld_vals, pf_ytd_vals, pf_wk_vals = [], [], [], []
    for t in holdings:
        if t not in daily.columns:
            continue
        mtd = _pct(last_row[t], mtd_base_row[t])
        ld  = _pct(last_row[t], prev_row[t])
        ytd = _pct(last_row[t], ytd_base_row[t])
        wk  = _pct(last_row[t], wk_base_row[t])
        stocks.append({
            "ticker":       t,
            "symbol":       t.replace(".NS", ""),
            "mtd_pct":      mtd,
            "last_day_pct": ld,
            "last_price":   round(float(last_row[t]), 2) if not pd.isna(last_row[t]) else None,
        })
        if mtd is not None: pf_mtd_vals.append(mtd)
        if ld  is not None: pf_ld_vals.append(ld)
        if ytd is not None: pf_ytd_vals.append(ytd)
        if wk  is not None: pf_wk_vals.append(wk)

    def _bench_base(col, before_ts):
        """Last non-NaN close for col before a given timestamp."""
        if col not in daily.columns:
            return None
        s = daily[col].dropna()
        s = s[s.index < before_ts]
        return float(s.iloc[-1]) if not s.empty else None

    bench_last  = last_row.get(_BENCHMARK)
    bench2_last = last_row.get(_BENCHMARK2)
    bmt = _pct(bench_last,  _bench_base(_BENCHMARK,  month_start_ts))
    bld = _pct(bench_last,  _bench_base(_BENCHMARK,  daily.index[-1]))
    byt = _pct(bench_last,  _bench_base(_BENCHMARK,  year_start_ts))
    bwk = _pct(bench_last,  _bench_base(_BENCHMARK,  week_start_ts))
    bwk2 = _pct(bench2_last, _bench_base(_BENCHMARK2, week_start_ts))
    bmt2 = _pct(bench2_last, _bench_base(_BENCHMARK2, month_start_ts))
    byt2 = _pct(bench2_last, _bench_base(_BENCHMARK2, year_start_ts))

    return {
        "m": m, "x": x,
        "as_of":              str(today),
        "last_trading_day":   daily.index[-1].strftime("%d %b %Y"),
        "holdings":           holdings,
        "stocks":             stocks,
        "portfolio_mtd":      round(sum(pf_mtd_vals) / len(pf_mtd_vals), 2) if pf_mtd_vals else None,
        "benchmark_mtd":      bmt,
        "portfolio_last_day": round(sum(pf_ld_vals)  / len(pf_ld_vals),  2) if pf_ld_vals  else None,
        "benchmark_last_day": bld,
        "portfolio_ytd":      round(sum(pf_ytd_vals) / len(pf_ytd_vals), 2) if pf_ytd_vals else None,
        "benchmark_ytd":      byt,
        "portfolio_week":     round(sum(pf_wk_vals)  / len(pf_wk_vals),  2) if pf_wk_vals  else None,
        "benchmark_week":     bwk,
        "smallcap_week":      bwk2,
        "smallcap_mtd":       bmt2,
        "smallcap_ytd":       byt2,
    }


# ── rebalance plan ───────────────────────────────────────────────────────────

def get_rebalance_plan(
    m: int = _DEFAULT_M,
    x: int = _DEFAULT_X,
    years: int = _YEARS,
    capital: float = 0,          # 0 = use CAPITAL from config
    exclude_tickers: list[str] = None,   # tickers to ignore (existing non-pflio positions)
    csv_path: str = UNIVERSE_CSV,
) -> dict:
    """
    Compare pflio target holdings vs actual Kite demat.
    Returns lists of SELL / BUY / HOLD actions with quantities and values.
    Does NOT place any orders.
    exclude_tickers: symbols like ["ANTHEM", "NIFTYBEES"] or full tickers ["ANTHEM.NS"]
    """
    from kite_orders import get_holdings, get_quote, get_margins
    from config import CAPITAL as _CONFIG_CAPITAL

    target_capital = capital if capital > 0 else _CONFIG_CAPITAL

    # Normalise exclusion list to full ticker format
    excluded = set()
    for t in (exclude_tickers or []):
        t = t.strip().upper()
        if t:
            excluded.add(t if t.endswith(".NS") else t + ".NS")

    # Always exclude active swing positions from open_positions.json
    try:
        import json as _json
        from config import POSITIONS_FILE
        with open(POSITIONS_FILE) as _f:
            for pos in _json.load(_f):
                tk = pos.get("ticker", "").strip().upper()
                if tk:
                    excluded.add(tk if tk.endswith(".NS") else tk + ".NS")
        logger.info("Auto-excluded %d swing positions from rebalance", len(excluded))
    except Exception as exc:
        logger.warning("Could not load open_positions.json for exclusion: %s", exc)

    # 1. Determine pflio target
    tickers  = load_universe(csv_path)
    price_df = download_monthly(tickers, years)
    stock_cols  = [c for c in price_df.columns if c != _BENCHMARK]
    monthly_ret = price_df[stock_cols].pct_change().dropna(how="all")
    _, target_holdings = _run_pflio(monthly_ret, m, x)
    target_set = set(target_holdings)

    # 2. Actual Kite demat — strip out excluded (pre-existing non-pflio) positions
    kite_rows = get_holdings()
    kite_map  = {h["ticker"]: h for h in kite_rows if h["ticker"] not in excluded}

    # 3. Margins
    margins = get_margins()
    available = margins["available"]

    # 4. Live quotes for target holdings not yet held
    target_per_stock = round(target_capital / m, 2)
    sells, buys, holds = [], [], []

    # SELL: in demat but NOT in target
    for ticker, h in kite_map.items():
        if ticker not in target_set:
            sells.append({
                "ticker":    ticker,
                "symbol":    ticker.replace(".NS", ""),
                "action":    "SELL",
                "qty":       h["qty"],
                "avg_price": h["avg_price"],
                "last_price": h["last_price"],
                "pnl_pct":   h["pnl_pct"],
                "est_value": round(h["last_price"] * h["qty"], 2),
            })

    # BUY: in target but NOT in demat
    for ticker in target_holdings:
        if ticker not in kite_map:
            try:
                import math as _math
                q     = get_quote(ticker)
                price = float(q["last_price"] or 0)
                if not price or _math.isnan(price):
                    raise ValueError(f"No valid price for {ticker}")
                qty   = int(target_per_stock // price)
                if qty < 1:
                    qty = 1
                buys.append({
                    "ticker":    ticker,
                    "symbol":    ticker.replace(".NS", ""),
                    "action":    "BUY",
                    "qty":       qty,
                    "last_price": round(price, 2),
                    "est_value": round(price * qty, 2),
                    "target_alloc": target_per_stock,
                })
            except Exception as exc:
                logger.warning("Quote failed for %s: %s", ticker, exc)
                buys.append({
                    "ticker": ticker, "symbol": ticker.replace(".NS", ""),
                    "action": "BUY", "qty": 0, "last_price": None,
                    "est_value": 0, "target_alloc": target_per_stock,
                    "error": str(exc),
                })
        else:
            h = kite_map[ticker]
            holds.append({
                "ticker":    ticker,
                "symbol":    ticker.replace(".NS", ""),
                "action":    "HOLD",
                "qty":       h["qty"],
                "avg_price": h["avg_price"],
                "last_price": h["last_price"],
                "pnl_pct":   h["pnl_pct"],
            })

    total_buy_value  = sum(b["est_value"] for b in buys)
    total_sell_value = sum(s["est_value"] for s in sells)

    return {
        "m": m, "x": x,
        "target_capital":   target_capital,
        "target_per_stock": target_per_stock,
        "available_margin": available,
        "proceeds_from_sells": round(total_sell_value, 2),
        "total_buy_value":  round(total_buy_value, 2),
        "net_cash_needed":  round(total_buy_value - total_sell_value - available, 2),
        "sells":  sells,
        "buys":   buys,
        "holds":  holds,
        "as_of":  str(date.today()),
    }


def execute_rebalance(
    sells: list[dict],
    buys: list[dict],
    dry_run: bool = True,
) -> dict:
    """
    Place CNC sell orders first, then CNC buy orders.
    sell/buy items must have: ticker, qty.
    Returns results per order.
    """
    from kite_orders import place_sell, place_buy

    results = []
    for s in sells:
        try:
            qty = int(s["qty"])
            r = place_sell(s["ticker"], qty, order_type="MARKET", dry_run=dry_run)
            results.append({"action": "SELL", **r})
        except Exception as exc:
            results.append({"action": "SELL", "ticker": s["ticker"],
                            "qty": s.get("qty"), "status": "error", "error": str(exc)})

    for b in buys:
        if b.get("qty", 0) < 1:
            results.append({"action": "BUY", "ticker": b["ticker"],
                            "qty": 0, "status": "skipped", "error": "qty=0"})
            continue
        try:
            import math as _m
            bqty = b["qty"]
            if bqty is None or (isinstance(bqty, float) and _m.isnan(bqty)):
                raise ValueError(f"Invalid qty for {b['ticker']}")
            r = place_buy(b["ticker"], int(bqty), order_type="MARKET", dry_run=dry_run)
            results.append({"action": "BUY", **r})
        except Exception as exc:
            results.append({"action": "BUY", "ticker": b["ticker"],
                            "qty": b["qty"], "status": "error", "error": str(exc)})

    return {
        "dry_run": dry_run,
        "total":   len(results),
        "placed":  sum(1 for r in results if r.get("status") in ("placed", "dry_run")),
        "errors":  sum(1 for r in results if r.get("status") == "error"),
        "results": results,
    }


# ── main entry ────────────────────────────────────────────────────────────────

def run_backtest(
    m: int  = _DEFAULT_M,
    x: int  = _DEFAULT_X,
    years: int = _YEARS,
    csv_path: str = UNIVERSE_CSV,
    progress_cb=None,
) -> dict:
    """
    Full backtest run. Returns dict with KPIs + monthly equity curves.
    `progress_cb(step, total, message)` for SSE progress updates.
    """
    def _progress(step, total, msg):
        if progress_cb:
            progress_cb(step, total, msg)
        logger.info("[%d/%d] %s", step, total, msg)

    _progress(1, 5, "Loading universe …")
    tickers = load_universe(csv_path)
    if not tickers:
        raise ValueError(f"No tickers loaded from {csv_path}")

    _progress(2, 5, f"Downloading {len(tickers)} tickers ({years}y monthly) …")
    price_df = download_monthly(tickers, years)

    _progress(3, 5, "Running pflio strategy …")
    strat_returns = pflio(price_df, m=m, x=x)

    _progress(4, 5, "Computing benchmark returns …")
    if _BENCHMARK not in price_df.columns:
        raise ValueError("Benchmark data unavailable — check internet connection.")
    bench_price   = price_df[_BENCHMARK].dropna()
    bench_returns = bench_price.pct_change().dropna()

    # Align to common dates
    common = strat_returns.index.intersection(bench_returns.index)
    strat_r = strat_returns.loc[common]
    bench_r = bench_returns.loc[common]

    _progress(5, 5, "Computing KPIs …")
    strat_kpi = compute_kpis(strat_r, f"Microcap250 Momentum (m={m}, x={x})")
    bench_kpi = compute_kpis(bench_r, "Nifty 50 Buy & Hold")

    # Equity curves (normalised to ₹1)
    strat_curve = (1 + strat_r).cumprod()
    bench_curve = (1 + bench_r).cumprod()

    # Monthly returns grid: year → {month: pct, ..., "Annual": pct}
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    def _monthly_grid(ret: pd.Series) -> list[dict]:
        df = ret.copy()
        df.index = pd.to_datetime(df.index)
        rows = []
        for yr in sorted(df.index.year.unique()):
            yr_data = df[df.index.year == yr]
            row = {"year": int(yr)}
            annual = 1.0
            for mi, mon in enumerate(months, 1):
                mo_data = yr_data[yr_data.index.month == mi]
                if len(mo_data):
                    v = round(float(mo_data.iloc[-1]) * 100, 2)
                    row[mon] = v
                    annual *= (1 + mo_data.iloc[-1])
                else:
                    row[mon] = None
            row["Annual"] = round((annual - 1) * 100, 2)
            rows.append(row)
        return rows

    return {
        "params":          {"m": m, "x": x, "years": years, "universe_size": len(tickers)},
        "strategy_kpi":    strat_kpi,
        "benchmark_kpi":   bench_kpi,
        "dates":           [d.strftime("%Y-%m") for d in common],
        "strategy_curve":  [round(v, 4) for v in strat_curve.tolist()],
        "benchmark_curve": [round(v, 4) for v in bench_curve.tolist()],
        "monthly_returns": _monthly_grid(strat_r),
        "monthly_bench":   _monthly_grid(bench_r),
        "run_at":          datetime.now().isoformat(timespec="seconds"),
    }
