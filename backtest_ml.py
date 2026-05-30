"""
backtest_ml.py — Walk-forward ML strategy backtest.

Every month-start:
  1. Re-rank universe with ML factor model (no lookahead bias)
  2. Hold top `top_n` stocks equal-weight
  3. Charge `cost_bps` per leg on stocks added/removed

Benchmarks: Nifty 50 (^NSEI)
KPIs: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Alpha
"""
import warnings; warnings.filterwarnings("ignore")
import math
import numpy as np
import pandas as pd
from datetime import date

import strategy as st

RISK_FREE = st.RISK_FREE
BENCHMARK = st.BENCHMARK


# ── Performance metrics ───────────────────────────────────────────────────────

def _cagr(ret: pd.Series) -> float:
    if len(ret) == 0: return 0.0
    n_y = len(ret) / 12
    return float((1 + ret).prod() ** (1 / n_y) - 1) if n_y > 0 else 0.0

def _sharpe(ret: pd.Series) -> float:
    mrf = (1 + RISK_FREE) ** (1 / 12) - 1
    exc = ret - mrf
    return float(exc.mean() / exc.std() * np.sqrt(12)) if exc.std() > 0 else 0.0

def _sortino(ret: pd.Series) -> float:
    mrf  = (1 + RISK_FREE) ** (1 / 12) - 1
    exc  = ret - mrf
    dstd = exc[exc < 0].std()
    return float(exc.mean() / dstd * np.sqrt(12)) if dstd and dstd > 0 else 0.0

def _max_dd(ret: pd.Series) -> float:
    wealth = (1 + ret).cumprod()
    dd     = (wealth - wealth.cummax()) / wealth.cummax()
    return float(dd.min())

def _win_rate(ret: pd.Series) -> float:
    return float((ret > 0).mean() * 100) if len(ret) else 0.0

def _kpis(ret: pd.Series, label: str) -> dict:
    c = _cagr(ret) * 100
    d = _max_dd(ret) * 100
    return {
        "label":         label,
        "cagr_pct":      round(c, 2),
        "sharpe":        round(_sharpe(ret), 2),
        "sortino":       round(_sortino(ret), 2),
        "max_dd_pct":    round(d, 2),
        "calmar":        round(abs(c / d) if d else 0, 2),
        "win_rate_pct":  round(_win_rate(ret), 1),
        "total_months":  int(len(ret)),
        "total_return_pct": round(float((1 + ret).prod() - 1) * 100, 2),
    }

def _monthly_grid(ret: pd.Series) -> list[dict]:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df = ret.copy(); df.index = pd.to_datetime(df.index)
    rows = []
    for yr in sorted(df.index.year.unique()):
        yr_d = df[df.index.year == yr]
        row  = {"year": int(yr)}
        ann  = 1.0
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
    """
    Full walk-forward ML backtest.
    cost_bps: one-way transaction cost in basis points (applied to turnover fraction).
    Returns dict with KPIs, equity curves, monthly grids, rebalance history, current portfolio.
    """
    def _prog(step, total, msg):
        if progress_cb: progress_cb(step, total, msg)

    _prog(1, 6, "Loading universe …")
    tickers = st.load_universe()

    _prog(2, 6, f"Downloading {len(tickers)} tickers ({years}y daily) …")
    close, volume = st.download_data(tickers, years=years)

    _prog(3, 6, "Computing ML factors (vectorised) …")
    factor_dfs = st.compute_factors(close, volume)

    stocks = [c for c in close.columns if c != BENCHMARK]

    # ── Rebalance dates: first trading day of each month ──────────────────────
    # Exclude warm-up period (need MIN_BARS for factors)
    warmup_end = close.index[0] + pd.DateOffset(months=14)
    td_after_warmup = close.index[close.index > warmup_end]
    if td_after_warmup.empty:
        raise ValueError("Not enough history for warm-up period.")

    td_df = pd.DataFrame({"date": td_after_warmup})
    td_df["ym"] = td_df["date"].dt.to_period("M")
    rebalance_dates = list(td_df.groupby("ym")["date"].first().values)

    _prog(4, 6, f"Walk-forward simulation: {len(rebalance_dates)} months …")

    port_returns  = {}
    bench_returns = {}
    rebalance_log = []
    prev_portfolio: list[str] = []

    for i, rd in enumerate(rebalance_dates):
        rd_ts   = pd.Timestamp(rd)
        next_rd = (pd.Timestamp(rebalance_dates[i + 1])
                   if i < len(rebalance_dates) - 1
                   else pd.Timestamp(date.today()))

        # Rank universe up to this date (no lookahead)
        ranked = st.rank_on_date(factor_dfs, rd_ts, close)
        if ranked.empty:
            continue
        portfolio = list(ranked.head(top_n)["ticker"])

        # Transaction cost on turnover
        added    = [t for t in portfolio if t not in prev_portfolio]
        removed  = [t for t in prev_portfolio if t not in portfolio]
        turnover = len(added) / max(len(portfolio), 1)
        cost     = turnover * (cost_bps / 10_000) * 2   # in + out legs

        # Use prices AT the rebalance dates so that monthly returns telescope
        # into the exact buy-and-hold return (no overnight gaps lost).
        idx_gte_rd   = close.index[close.index >= rd_ts]
        idx_gte_next = close.index[close.index >= next_rd]
        if idx_gte_rd.empty:
            prev_portfolio = portfolio[:]
            continue
        entry_row = close.loc[idx_gte_rd[0]]
        exit_row  = close.loc[idx_gte_next[0]] if not idx_gte_next.empty else close.iloc[-1]

        port_tickers  = [t for t in portfolio if t in close.columns]
        valid_tickers = [t for t in port_tickers
                         if pd.notna(entry_row.get(t)) and pd.notna(exit_row.get(t))
                         and entry_row[t] > 0]
        if not valid_tickers:
            prev_portfolio = portfolio[:]
            continue

        individual_rets = (exit_row[valid_tickers] / entry_row[valid_tickers]) - 1
        port_monthly    = float(individual_rets.mean()) - cost

        # Benchmark: same rebalance-date methodology
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

        # Top-5 for history log
        top5 = ranked.head(5)[["ticker", "composite_score"]].to_dict("records")

        rebalance_log.append({
            "date":           label,
            "holdings":       portfolio[:],
            "added":          added,
            "removed":        removed,
            "turnover_pct":   round(turnover * 100, 1),
            "period_ret_pct": round(port_monthly * 100, 2),
            "bench_ret_pct":  round(bench_monthly * 100, 2),
            "top5": [{**r, "composite_score": round(float(r["composite_score"]), 1)} for r in top5],
        })
        prev_portfolio = portfolio[:]

    _prog(5, 6, "Computing performance metrics …")

    strat_r = pd.Series(port_returns)
    bench_r = pd.Series(bench_returns).reindex(strat_r.index).fillna(0)

    strat_kpi = _kpis(strat_r, f"ML Top-{top_n} ({len(stocks)} stocks)")
    bench_kpi = _kpis(bench_r, "Nifty 500 Buy & Hold")
    strat_kpi["alpha_pct"] = round(strat_kpi["cagr_pct"] - bench_kpi["cagr_pct"], 2)

    strat_curve = (1 + strat_r).cumprod()
    bench_curve = (1 + bench_r).cumprod()

    # ── Current portfolio (ranked today) ─────────────────────────────────────
    _prog(6, 6, "Building current portfolio …")
    today_ranked = st.rank_on_date(factor_dfs, pd.Timestamp(date.today()), close)

    def _safe(v):
        if isinstance(v, (float, np.floating)):
            if not math.isfinite(v): return None
            return round(float(v), 4)
        if isinstance(v, (bool, np.bool_)): return bool(v)
        if isinstance(v, (int, np.integer)): return int(v)
        return v

    def _sanitize(obj):
        """Recursively replace NaN/inf in nested dicts and lists."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(i) for i in obj]
        return _safe(obj) if isinstance(obj, (float, np.floating)) else obj

    current_top20 = [
        {k: _safe(v) for k, v in row.items()}
        for _, row in today_ranked.head(top_n).iterrows()
    ]

    # ── Rolling 12-month alpha ────────────────────────────────────────────────
    rolling_alpha = []
    if len(strat_r) >= 12:
        for i in range(12, len(strat_r) + 1):
            w_strat = strat_r.iloc[i - 12 : i]
            w_bench = bench_r.iloc[i - 12 : i]
            rolling_alpha.append({
                "date":  strat_r.index[i - 1],
                "alpha": round((_cagr(w_strat) - _cagr(w_bench)) * 100, 2),
            })

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
        "current_top20":     current_top20,
        "rolling_alpha":     rolling_alpha,
        "factor_weights":    st.WEIGHTS,
        "factor_labels":     st.FACTOR_LABELS,
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    return _sanitize(result)
