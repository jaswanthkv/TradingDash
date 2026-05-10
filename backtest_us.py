"""
backtest_us.py — Monthly momentum backtest for NASDAQ 100 universe.

Same strategy as backtest.py (India):
  - Hold top `m` stocks ranked by prior-month return
  - Replace `x` worst performers each month
  - Benchmark: QQQ buy-and-hold
  - No order execution (US stocks, observation only)
"""
import logging
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from backtest import cagr, sharpe, max_drawdown, compute_kpis, _run_pflio

logger = logging.getLogger(__name__)

_BENCHMARK  = "QQQ"
_RISK_FREE  = 0.045      # ~4.5% US short-term
_DEFAULT_M  = 8
_DEFAULT_X  = 3
_YEARS      = 10

# Current NASDAQ 100 constituents (approximate — yfinance tickers)
NASDAQ100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "AMD", "ADBE", "QCOM", "ASML", "CSCO", "INTU", "AMAT", "TXN", "BKNG",
    "ISRG", "AMGN", "MU", "HON", "VRTX", "LRCX", "KLAC", "ADI", "MELI", "PANW",
    "REGN", "CDNS", "SNPS", "ABNB", "CRWD", "MDLZ", "FTNT", "MRVL", "KDP", "CEG",
    "CTAS", "ORLY", "AEP", "IDXX", "PCAR", "CPRT", "PAYX", "ROST", "DXCM", "MNST",
    "FAST", "ODFL", "BIIB", "EA", "BKR", "VRSK", "TEAM", "WBD", "ZS", "XEL",
    "CTSH", "DLTR", "ANSS", "TTD", "ON", "ENPH", "ILMN", "MDB", "DDOG", "PLTR",
    "COIN", "APP", "AXON", "EBAY", "TTWO", "NTAP", "MCHP", "SBUX", "PYPL", "PEP",
    "GILD", "CSX", "VRSN", "GEHC", "GFS", "FANG", "CCEP", "MSTR", "AZN", "CHTR",
    "CDW", "CINF", "LPLA", "NTRS", "SIRI", "SMCI", "ROP", "ODFL", "BMRN", "ALGN",
]


def load_universe() -> list[str]:
    return sorted(set(NASDAQ100))


def download_monthly(tickers: list[str], years: int = _YEARS) -> pd.DataFrame:
    all_tickers = tickers + [_BENCHMARK]
    logger.info("US: Downloading %d tickers (%dy monthly) …", len(all_tickers), years)
    raw = yf.download(
        all_tickers, period=f"{years}y", interval="1mo",
        auto_adjust=True, progress=False, threads=True,
    )
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes.index = pd.to_datetime(closes.index).tz_localize(None)

    today = date.today()
    closes = closes[closes.index < pd.Timestamp(today.replace(day=1))]
    closes = closes.dropna(axis=1, thresh=int(0.6 * len(closes)))
    logger.info("US: Data shape after filter: %s", closes.shape)
    return closes


def pflio(price_df: pd.DataFrame, m: int = _DEFAULT_M, x: int = _DEFAULT_X) -> pd.Series:
    stock_cols  = [c for c in price_df.columns if c != _BENCHMARK]
    monthly_ret = price_df[stock_cols].pct_change().dropna(how="all")
    port_returns, _ = _run_pflio(monthly_ret, m, x)
    return pd.Series(port_returns, name="Strategy")


def get_current_pf(m: int = _DEFAULT_M, x: int = _DEFAULT_X, years: int = _YEARS) -> dict:
    tickers  = load_universe()
    price_df = download_monthly(tickers, years)
    stock_cols  = [c for c in price_df.columns if c != _BENCHMARK]
    monthly_ret = price_df[stock_cols].pct_change().dropna(how="all")
    _, holdings = _run_pflio(monthly_ret, m, x)

    today       = date.today()
    month_start = today.replace(day=1)
    year_start  = today.replace(month=1, day=1)
    fetch_from  = (year_start - timedelta(days=15)).strftime("%Y-%m-%d")
    week_start  = today - timedelta(days=today.weekday())

    daily_raw = yf.download(
        holdings + [_BENCHMARK], start=fetch_from,
        auto_adjust=True, progress=False, threads=True,
    )
    daily = (daily_raw["Close"] if isinstance(daily_raw.columns, pd.MultiIndex) else daily_raw)
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    daily = daily.dropna(how="all")

    if daily.empty:
        return {"holdings": holdings, "stocks": [], "portfolio_mtd": None,
                "benchmark_mtd": None, "as_of": str(today)}

    month_start_ts = pd.Timestamp(month_start)
    year_start_ts  = pd.Timestamp(year_start)
    week_start_ts  = pd.Timestamp(week_start)

    prior_month  = daily[daily.index < month_start_ts]
    mtd_base_row = prior_month.iloc[-1] if not prior_month.empty else daily.iloc[0]

    prior_year   = daily[daily.index < year_start_ts]
    ytd_base_row = prior_year.iloc[-1] if not prior_year.empty else daily.iloc[0]

    prior_week   = daily[daily.index < week_start_ts]
    wk_base_row  = prior_week.iloc[-1] if not prior_week.empty else daily.iloc[0]

    last_row     = daily.iloc[-1]
    prev_row     = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]

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
            "symbol":       t,
            "mtd_pct":      mtd,
            "last_day_pct": ld,
            "last_price":   round(float(last_row[t]), 2) if not pd.isna(last_row[t]) else None,
        })
        if mtd is not None: pf_mtd_vals.append(mtd)
        if ld  is not None: pf_ld_vals.append(ld)
        if ytd is not None: pf_ytd_vals.append(ytd)
        if wk  is not None: pf_wk_vals.append(wk)

    def _bench_base(before_ts):
        s = daily[_BENCHMARK].dropna()
        s = s[s.index < before_ts]
        return float(s.iloc[-1]) if not s.empty else None

    bench_last = last_row.get(_BENCHMARK)
    bmt = _pct(bench_last, _bench_base(month_start_ts))
    bld = _pct(bench_last, _bench_base(daily.index[-1]))
    byt = _pct(bench_last, _bench_base(year_start_ts))
    bwk = _pct(bench_last, _bench_base(week_start_ts))

    return {
        "m": m, "x": x, "market": "US / NASDAQ 100",
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
        "benchmark_label":    "QQQ",
    }


def run_backtest(m: int = _DEFAULT_M, x: int = _DEFAULT_X, years: int = _YEARS,
                 progress_cb=None) -> dict:
    def _p(step, total, msg):
        if progress_cb:
            progress_cb(step, total, msg)
        logger.info("[%d/%d] %s", step, total, msg)

    _p(1, 5, "Loading NASDAQ 100 universe …")
    tickers = load_universe()

    _p(2, 5, f"Downloading {len(tickers)} tickers ({years}y monthly) …")
    price_df = download_monthly(tickers, years)

    _p(3, 5, "Running momentum strategy …")
    strat_returns = pflio(price_df, m=m, x=x)

    _p(4, 5, "Computing QQQ benchmark …")
    if _BENCHMARK not in price_df.columns:
        raise ValueError("QQQ benchmark data unavailable.")
    bench_returns = price_df[_BENCHMARK].dropna().pct_change().dropna()

    common  = strat_returns.index.intersection(bench_returns.index)
    strat_r = strat_returns.loc[common]
    bench_r = bench_returns.loc[common]

    _p(5, 5, "Computing KPIs …")
    strat_kpi = compute_kpis(strat_r, f"NASDAQ100 Momentum (m={m}, x={x})")
    bench_kpi = compute_kpis(bench_r, "QQQ Buy & Hold")

    strat_curve = (1 + strat_r).cumprod()
    bench_curve = (1 + bench_r).cumprod()

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    def _grid(ret: pd.Series) -> list[dict]:
        df = ret.copy(); df.index = pd.to_datetime(df.index)
        rows = []
        for yr in sorted(df.index.year.unique()):
            yr_data = df[df.index.year == yr]
            row = {"year": int(yr)}; annual = 1.0
            for mi, mon in enumerate(months, 1):
                mo = yr_data[yr_data.index.month == mi]
                if len(mo):
                    v = round(float(mo.iloc[-1]) * 100, 2)
                    row[mon] = v; annual *= (1 + mo.iloc[-1])
                else:
                    row[mon] = None
            row["Annual"] = round((annual - 1) * 100, 2)
            rows.append(row)
        return rows

    return {
        "market":          "US / NASDAQ 100",
        "params":          {"m": m, "x": x, "years": years, "universe_size": len(tickers)},
        "strategy_kpi":    strat_kpi,
        "benchmark_kpi":   bench_kpi,
        "benchmark_label": "QQQ",
        "dates":           [d.strftime("%Y-%m") for d in common],
        "strategy_curve":  [round(v, 4) for v in strat_curve.tolist()],
        "benchmark_curve": [round(v, 4) for v in bench_curve.tolist()],
        "monthly_returns": _grid(strat_r),
        "monthly_bench":   _grid(bench_r),
        "run_at":          datetime.now().isoformat(timespec="seconds"),
    }
