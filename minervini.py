"""
minervini.py — Mark Minervini SEPA strategy backtest + screener.

SEPA Criteria (all 9 must pass to qualify):
  1. Price > 150-day SMA
  2. Price > 200-day SMA
  3. 150-day SMA > 200-day SMA
  4. 200-day SMA trending up (slope positive over last 25 days)
  5. 50-day SMA > 150-day SMA
  6. 50-day SMA > 200-day SMA
  7. Price > 50-day SMA
  8. Price within 25% of 52-week high
  9. Price at least 30% above 52-week low

Ranking: RS Rating — cross-sectional percentile rank of 12-month return (0–99)
Portfolio: top-N qualifying stocks ranked by RS Rating, equal-weight, monthly rebalance
"""
import warnings; warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd
from datetime import date

import strategy as st

BENCHMARK = st.BENCHMARK
RISK_FREE  = st.RISK_FREE
MIN_BARS   = 280   # 252 (52w) + 28 buffer for SMA200 slope

CRITERIA_LABELS = {
    "c1": "Price > 150d SMA",
    "c2": "Price > 200d SMA",
    "c3": "150d SMA > 200d SMA",
    "c4": "200d SMA uptrend (25d)",
    "c5": "50d SMA > 150d SMA",
    "c6": "50d SMA > 200d SMA",
    "c7": "Price > 50d SMA",
    "c8": "Within 25% of 52w High",
    "c9": "30%+ above 52w Low",
}
CRITERIA_SHORT = {
    "c1": "P>150", "c2": "P>200", "c3": "150>200", "c4": "200↑",
    "c5": "50>150", "c6": "50>200", "c7": "P>50", "c8": "<25%H", "c9": ">30%L",
}


# ── SEPA signal computation ───────────────────────────────────────────────────

def compute_sepa(close: pd.DataFrame) -> dict:
    """Vectorised SEPA criteria across all dates × stocks."""
    stocks = [c for c in close.columns if c != BENCHMARK]
    sc = close[stocks]

    sma50  = sc.rolling(50,  min_periods=50).mean()
    sma150 = sc.rolling(150, min_periods=150).mean()
    sma200 = sc.rolling(200, min_periods=200).mean()

    high52w = sc.rolling(252, min_periods=200).max()
    low52w  = sc.rolling(252, min_periods=200).min()

    # RS Rating: cross-sectional percentile of 12-month return (0–99)
    ret12m    = sc / sc.shift(252) - 1
    rs_rating = ret12m.rank(axis=1, pct=True, na_option="keep") * 99

    c1 = sc > sma150
    c2 = sc > sma200
    c3 = sma150 > sma200
    c4 = (sma200 - sma200.shift(25)) > 0
    c5 = sma50 > sma150
    c6 = sma50 > sma200
    c7 = sc > sma50
    c8 = sc >= high52w * 0.75
    c9 = sc >= low52w  * 1.30

    sepa_pass = c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8 & c9

    return {
        "sma50": sma50, "sma150": sma150, "sma200": sma200,
        "high52w": high52w, "low52w": low52w,
        "rs_rating": rs_rating,
        "sepa_pass": sepa_pass,
        "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
        "c6": c6, "c7": c7, "c8": c8, "c9": c9,
    }


# ── Performance metrics ───────────────────────────────────────────────────────

def _cagr(ret: pd.Series) -> float:
    if len(ret) == 0: return 0.0
    n_y = len(ret) / 12
    return float((1 + ret).prod() ** (1 / n_y) - 1) if n_y > 0 else 0.0

def _sharpe(ret: pd.Series) -> float:
    mrf = (1 + RISK_FREE) ** (1/12) - 1
    exc = ret - mrf
    return float(exc.mean() / exc.std() * math.sqrt(12)) if exc.std() > 0 else 0.0

def _sortino(ret: pd.Series) -> float:
    mrf  = (1 + RISK_FREE) ** (1/12) - 1
    exc  = ret - mrf
    dstd = exc[exc < 0].std()
    return float(exc.mean() / dstd * math.sqrt(12)) if dstd and dstd > 0 else 0.0

def _max_dd(ret: pd.Series) -> float:
    wealth = (1 + ret).cumprod()
    dd = (wealth - wealth.cummax()) / wealth.cummax()
    return float(dd.min())

def _kpis(ret: pd.Series, label: str) -> dict:
    c = _cagr(ret) * 100
    d = _max_dd(ret) * 100
    return {
        "label":            label,
        "cagr_pct":         round(c, 2),
        "sharpe":           round(_sharpe(ret), 2),
        "sortino":          round(_sortino(ret), 2),
        "max_dd_pct":       round(d, 2),
        "calmar":           round(abs(c / d) if d else 0, 2),
        "win_rate_pct":     round(float((ret > 0).mean() * 100) if len(ret) else 0.0, 1),
        "total_months":     int(len(ret)),
        "total_return_pct": round(float((1 + ret).prod() - 1) * 100, 2),
    }

def _monthly_grid(ret: pd.Series) -> list:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df = ret.copy(); df.index = pd.to_datetime(df.index)
    rows = []
    for yr in sorted(df.index.year.unique()):
        yr_d = df[df.index.year == yr]
        row = {"year": int(yr)}
        ann = 1.0
        for mi, mon in enumerate(months, 1):
            mo_d = yr_d[yr_d.index.month == mi]
            if len(mo_d):
                v = round(float(mo_d.iloc[-1]) * 100, 2)
                row[mon] = v; ann *= (1 + mo_d.iloc[-1])
            else:
                row[mon] = None
        row["Annual"] = round((ann - 1) * 100, 2)
        rows.append(row)
    return rows


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    years:    int   = 5,
    top_n:    int   = 20,
    capital:  float = 500_000,
    cost_bps: float = 20,
    progress_cb=None,
) -> dict:
    def _prog(step, total, msg):
        if progress_cb: progress_cb(step, total, msg)

    _prog(1, 5, "Loading universe …")
    tickers = st.load_universe()

    _prog(2, 5, f"Downloading {len(tickers)} tickers ({years}y daily) …")
    close, _ = st.download_data(tickers, years=years)

    _prog(3, 5, "Computing SEPA criteria …")
    sepa   = compute_sepa(close)
    stocks = [c for c in close.columns if c != BENCHMARK]

    # Rebalance dates: first trading day each month after warmup
    warmup_end = close.index[0] + pd.DateOffset(days=int(MIN_BARS * 365 / 252) + 30)
    td_after   = close.index[close.index > warmup_end]
    if td_after.empty:
        raise ValueError("Not enough history for SEPA warmup.")

    td_df = pd.DataFrame({"date": td_after})
    td_df["ym"] = td_df["date"].dt.to_period("M")
    rebalance_dates = list(td_df.groupby("ym")["date"].first().values)

    _prog(4, 5, f"Walk-forward simulation: {len(rebalance_dates)} months …")

    port_returns: dict  = {}
    bench_returns: dict = {}
    rebalance_log: list = []
    prev_portfolio: list = []

    for i, rd in enumerate(rebalance_dates):
        rd_ts   = pd.Timestamp(rd)
        next_rd = (pd.Timestamp(rebalance_dates[i + 1])
                   if i < len(rebalance_dates) - 1
                   else pd.Timestamp(date.today()))

        avail = close.index[close.index <= rd_ts]
        if avail.empty:
            continue
        dt = avail[-1]

        if dt not in sepa["sepa_pass"].index:
            continue

        pass_row = sepa["sepa_pass"].loc[dt]
        rs_row   = sepa["rs_rating"].loc[dt]

        # Stocks passing all 9 SEPA criteria with valid RS
        passing = [
            t for t in stocks
            if t in pass_row.index and pass_row.get(t) is True
            and t in rs_row.index  and pd.notna(rs_row.get(t))
        ]

        if not passing:
            prev_portfolio = []
            rebalance_log.append({
                "date": rd_ts.strftime("%Y-%m"),
                "holdings": [], "added": [], "removed": prev_portfolio[:],
                "qualifying": 0, "turnover_pct": 0,
                "period_ret_pct": 0.0, "bench_ret_pct": 0.0, "top5": [],
            })
            prev_portfolio = []
            continue

        # Rank by RS Rating descending
        portfolio = list(rs_row[passing].sort_values(ascending=False).head(top_n).index)

        added    = [t for t in portfolio if t not in prev_portfolio]
        removed  = [t for t in prev_portfolio if t not in portfolio]
        turnover = len(added) / max(len(portfolio), 1)
        cost     = turnover * (cost_bps / 10_000) * 2

        idx_gte_rd   = close.index[close.index >= rd_ts]
        idx_gte_next = close.index[close.index >= next_rd]
        if idx_gte_rd.empty:
            prev_portfolio = portfolio[:]
            continue
        entry_row = close.loc[idx_gte_rd[0]]
        exit_row  = close.loc[idx_gte_next[0]] if not idx_gte_next.empty else close.iloc[-1]

        valid = [
            t for t in portfolio
            if t in close.columns
            and pd.notna(entry_row.get(t)) and pd.notna(exit_row.get(t))
            and entry_row[t] > 0
        ]
        if not valid:
            prev_portfolio = portfolio[:]
            continue

        ind_rets     = (exit_row[valid] / entry_row[valid]) - 1
        port_monthly = float(ind_rets.mean()) - cost

        try:
            b_entry = entry_row.get(BENCHMARK) if BENCHMARK in close.columns else None
            b_exit  = exit_row.get(BENCHMARK)  if BENCHMARK in close.columns else None
            bench_monthly = float(b_exit / b_entry - 1) \
                if (b_entry and b_exit and b_entry > 0 and math.isfinite(b_exit / b_entry)) \
                else 0.0
        except Exception:
            bench_monthly = 0.0

        label = rd_ts.strftime("%Y-%m")
        port_returns[label]  = port_monthly
        bench_returns[label] = bench_monthly

        top5 = [{"ticker": t, "rs_rating": round(float(rs_row[t]), 1)}
                for t in portfolio[:5] if t in rs_row.index]

        rebalance_log.append({
            "date":           label,
            "holdings":       portfolio[:],
            "added":          added,
            "removed":        removed,
            "qualifying":     len(passing),
            "turnover_pct":   round(turnover * 100, 1),
            "period_ret_pct": round(port_monthly * 100, 2),
            "bench_ret_pct":  round(bench_monthly * 100, 2),
            "top5":           top5,
        })
        prev_portfolio = portfolio[:]

    _prog(5, 5, "Computing performance metrics …")

    strat_r = pd.Series(port_returns)
    bench_r = pd.Series(bench_returns).reindex(strat_r.index).fillna(0)

    strat_kpi = _kpis(strat_r, f"Minervini SEPA Top-{top_n}")
    bench_kpi = _kpis(bench_r, "Nifty 500 Buy & Hold")
    strat_kpi["alpha_pct"] = round(strat_kpi["cagr_pct"] - bench_kpi["cagr_pct"], 2)

    strat_curve = (1 + strat_r).cumprod()
    bench_curve = (1 + bench_r).cumprod()

    current_screen = screen_on_date(sepa, close, pd.Timestamp(date.today()), stocks)

    def _safe(v):
        if isinstance(v, (float, np.floating)):
            if not math.isfinite(v): return None
            return round(float(v), 4)
        if isinstance(v, (bool, np.bool_)): return bool(v)
        if isinstance(v, (int, np.integer)): return int(v)
        return v

    def _san(obj):
        if isinstance(obj, dict): return {k: _san(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_san(i) for i in obj]
        return _safe(obj) if isinstance(obj, (float, np.floating)) else obj

    result = {
        "params": {
            "years": years, "top_n": top_n,
            "cost_bps": cost_bps, "universe_size": len(stocks),
        },
        "strategy_kpi":      strat_kpi,
        "benchmark_kpi":     bench_kpi,
        "dates":             list(strat_r.index),
        "strategy_curve":    [round(float(v), 4) for v in strat_curve],
        "benchmark_curve":   [round(float(v), 4) for v in bench_curve],
        "monthly_returns":   _monthly_grid(strat_r),
        "monthly_bench":     _monthly_grid(bench_r),
        "rebalance_history": rebalance_log,
        "current_screen":    current_screen,
        "criteria_labels":   CRITERIA_LABELS,
        "criteria_short":    CRITERIA_SHORT,
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    return _san(result)


# ── Live screener ─────────────────────────────────────────────────────────────

def screen_on_date(sepa: dict, close: pd.DataFrame, ref_date: pd.Timestamp,
                   stocks: list) -> list:
    """Return all stocks with criteria breakdown, sorted: full pass → RS Rating desc."""
    avail = close.index[close.index <= ref_date]
    if avail.empty:
        return []
    dt = avail[-1]

    if dt not in sepa["sepa_pass"].index:
        return []

    rows = []
    for t in stocks:
        if t not in close.columns:
            continue
        price = close.loc[dt, t] if t in close.columns else None
        if price is None or pd.isna(price) or price <= 0:
            continue

        def _get(key):
            df = sepa.get(key)
            if df is None or t not in df.columns or dt not in df.index:
                return None
            v = df.loc[dt, t]
            return None if pd.isna(v) else v

        rs    = _get("rs_rating")
        h52   = _get("high52w")
        l52   = _get("low52w")
        s50   = _get("sma50")
        s150  = _get("sma150")
        s200  = _get("sma200")

        criteria = {k: bool(_get(k)) for k in CRITERIA_LABELS}
        n_pass = sum(criteria.values())

        rows.append({
            "ticker":        t,
            "symbol":        t.replace(".NS", ""),
            "price":         round(float(price), 2),
            "rs_rating":     round(float(rs), 1) if rs is not None else None,
            "sma50":         round(float(s50), 2) if s50 is not None else None,
            "sma150":        round(float(s150), 2) if s150 is not None else None,
            "sma200":        round(float(s200), 2) if s200 is not None else None,
            "high52w":       round(float(h52), 2) if h52 is not None else None,
            "low52w":        round(float(l52), 2) if l52 is not None else None,
            "pct_from_52h":  round((price / h52 - 1) * 100, 1) if h52 and h52 > 0 else None,
            "pct_above_52l": round((price / l52 - 1) * 100, 1) if l52 and l52 > 0 else None,
            "criteria":      criteria,
            "passing":       n_pass,
            "sepa_pass":     n_pass == 9,
        })

    rows.sort(key=lambda r: (-r["passing"], -(r["rs_rating"] or 0)))
    return rows
