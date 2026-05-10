"""
report.py — Weekly portfolio update card data.
"""
import logging
import datetime
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Known sector map for NASDAQ 100 to avoid slow API calls
_US_SECTORS = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMZN":"Consumer Disc.",
    "META":"Technology","GOOGL":"Technology","GOOG":"Technology","TSLA":"Consumer Disc.",
    "AVGO":"Technology","COST":"Consumer Staples","NFLX":"Communication","AMD":"Technology",
    "ADBE":"Technology","QCOM":"Technology","ASML":"Technology","CSCO":"Technology",
    "INTU":"Technology","AMAT":"Technology","TXN":"Technology","BKNG":"Consumer Disc.",
    "ISRG":"Healthcare","AMGN":"Healthcare","MU":"Technology","HON":"Industrials",
    "VRTX":"Healthcare","LRCX":"Technology","KLAC":"Technology","ADI":"Technology",
    "MELI":"Consumer Disc.","PANW":"Technology","REGN":"Healthcare","CDNS":"Technology",
    "SNPS":"Technology","ABNB":"Consumer Disc.","CRWD":"Technology","MDLZ":"Consumer Staples",
    "FTNT":"Technology","MRVL":"Technology","KDP":"Consumer Staples","CEG":"Utilities",
    "CTAS":"Industrials","ORLY":"Consumer Disc.","AEP":"Utilities","IDXX":"Healthcare",
    "PCAR":"Industrials","CPRT":"Industrials","PAYX":"Technology","ROST":"Consumer Disc.",
    "DXCM":"Healthcare","MNST":"Consumer Staples","FAST":"Industrials","ODFL":"Industrials",
    "BIIB":"Healthcare","EA":"Communication","BKR":"Energy","VRSK":"Industrials",
    "TEAM":"Technology","WBD":"Communication","ZS":"Technology","XEL":"Utilities",
    "CTSH":"Technology","DLTR":"Consumer Disc.","ANSS":"Technology","TTD":"Technology",
    "ON":"Technology","ENPH":"Technology","ILMN":"Healthcare","MDB":"Technology",
    "DDOG":"Technology","PLTR":"Technology","COIN":"Financials","APP":"Technology",
    "AXON":"Technology","EBAY":"Consumer Disc.","TTWO":"Communication","NTAP":"Technology",
    "MCHP":"Technology","SBUX":"Consumer Staples","PYPL":"Financials","PEP":"Consumer Staples",
    "GILD":"Healthcare","CSX":"Industrials","VRSN":"Technology","GEHC":"Healthcare",
    "GFS":"Technology","FANG":"Energy","CCEP":"Consumer Staples","CHTR":"Communication",
}

_SECTOR_CACHE: dict[str, str] = {}


def _get_sector_india(ticker: str) -> str:
    """Fetch sector for an NSE ticker via yfinance."""
    if ticker in _SECTOR_CACHE:
        return _SECTOR_CACHE[ticker]
    try:
        info = yf.Ticker(ticker).info
        s = info.get("sector") or info.get("sectorDisp") or "Other"
        _SECTOR_CACHE[ticker] = s
        return s
    except Exception:
        return "Other"


def _get_mcap_label_india(ticker: str) -> str:
    try:
        mc = yf.Ticker(ticker).info.get("marketCap") or 0
        mc_cr = mc / 1e7   # convert to crores
        if mc_cr > 20000: return "Large"
        if mc_cr > 5000:  return "Mid"
        return "Small"
    except Exception:
        return "Small"


def sector_distribution(holdings: list[str], market: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for t in holdings:
        if market == "us":
            s = _US_SECTORS.get(t, "Technology")
        else:
            s = _get_sector_india(t)
        counts[s] = counts.get(s, 0) + 1
    total = len(holdings) or 1
    return {k: round(v / total * 100, 1) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def mcap_distribution(holdings: list[str], market: str) -> dict[str, float]:
    if market == "us":
        # NASDAQ 100 holdings are all large/mega cap
        counts = {"Mega": 0, "Large": 0, "Mid": 0, "Small": 0}
        mc_thresholds = {
            "AAPL": "Mega", "MSFT": "Mega", "NVDA": "Mega", "AMZN": "Mega",
            "META": "Mega", "GOOGL": "Mega", "TSLA": "Large", "AVGO": "Large",
        }
        for t in holdings:
            label = mc_thresholds.get(t, "Large")
            counts[label] = counts.get(label, 0) + 1
        return {k: v for k, v in counts.items() if v > 0}
    else:
        counts: dict[str, int] = {}
        for t in holdings:
            label = _get_mcap_label_india(t)
            counts[label] = counts.get(label, 0) + 1
        return counts


def _fetch_week_curve(holdings: list[str], benchmark: str, benchmark2: str = "MOSMALL250.NS") -> dict:
    """Fetch daily curve for the current week for holdings vs Nifty 50 and Nifty SC250."""
    try:
        today      = datetime.date.today()
        week_start = today - datetime.timedelta(days=today.weekday())
        fetch_from = (week_start - datetime.timedelta(days=10)).strftime("%Y-%m-%d")

        tickers = list(dict.fromkeys(holdings + [benchmark, benchmark2]))
        raw = yf.download(tickers, start=fetch_from, auto_adjust=True, progress=False, threads=True)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        closes = closes.dropna(how="all")

        if closes.empty or len(closes) < 2:
            return {"dates": [], "strategy_curve": [], "benchmark_curve": [], "benchmark2_curve": []}

        week_start_ts = pd.Timestamp(week_start)
        this_week = closes[closes.index >= week_start_ts]
        if this_week.empty:
            this_week = closes

        def _base(col):
            s = closes[col].dropna() if col in closes.columns else pd.Series(dtype=float)
            s = s[s.index < week_start_ts]
            return float(s.iloc[-1]) if not s.empty else None

        stock_cols = [c for c in closes.columns if c not in (benchmark, benchmark2)]
        base_row   = closes[closes.index < week_start_ts]
        base_row   = base_row.iloc[-1] if not base_row.empty else closes.iloc[0]
        bench_base  = _base(benchmark)
        bench2_base = _base(benchmark2)

        pf_curve, bn_curve, bn2_curve, dates = [], [], [], []
        for ts, row in this_week.iterrows():
            pf_vals = []
            for t in stock_cols:
                try:
                    bv = float(base_row[t]); cv = float(row[t])
                    if bv and not (bv != bv) and not (cv != cv):
                        pf_vals.append(cv / bv)
                except Exception:
                    continue
            if not pf_vals:
                continue

            pf_curve.append(round(sum(pf_vals) / len(pf_vals), 4))

            def _ratio(col, base):
                try:
                    v = float(row[col])
                    return round(v / base, 4) if base and not (v != v) else None
                except Exception:
                    return None

            bn_curve.append(_ratio(benchmark,  bench_base))
            bn2_curve.append(_ratio(benchmark2, bench2_base))
            dates.append(ts.strftime("%a"))

        return {"dates": dates, "strategy_curve": pf_curve,
                "benchmark_curve": bn_curve, "benchmark2_curve": bn2_curve}
    except Exception as e:
        logger.warning("week curve fetch failed: %s", e)
        return {"dates": [], "strategy_curve": [], "benchmark_curve": [], "benchmark2_curve": []}


def generate_report_data(market: str, bt_result: dict, pf: dict) -> dict:
    holdings = pf.get("holdings", [])

    dates = bt_result.get("dates", [])
    strat = bt_result.get("strategy_curve", [])
    bench = bt_result.get("benchmark_curve", [])

    # Always show current week's daily curve (overrides backtest curve for the card)
    if holdings:
        benchmark_ticker = "QQQ" if market == "us" else "^NSEI"
        week = _fetch_week_curve(holdings, benchmark_ticker)
        dates  = week["dates"]
        strat  = week["strategy_curve"]
        bench  = week["benchmark_curve"]
        bench2 = week.get("benchmark2_curve", [])

    sk = bt_result.get("strategy_kpi", {})
    bk = bt_result.get("benchmark_kpi", {})

    # Annual return: backtest monthly grid (compounded) → fall back to live YTD
    annual_return = None
    year = datetime.date.today().year
    for row in bt_result.get("monthly_returns", []):
        if row.get("year") == year:
            annual_return = row.get("Annual")
            break

    # Sector & market cap (limit to avoid slowness)
    sample = holdings[:12]
    sectors = sector_distribution(sample, market)
    mcaps   = mcap_distribution(sample, market)

    return {
        "market":           market,
        "date":             datetime.date.today().strftime("%B %Y"),
        "positions":        len(holdings),
        "holdings":         holdings,
        "portfolio_mtd":    pf.get("portfolio_mtd"),
        "benchmark_mtd":    pf.get("benchmark_mtd"),
        "portfolio_last_day": pf.get("portfolio_last_day"),
        "portfolio_ytd":    annual_return,
        "benchmark_ytd":    pf.get("benchmark_ytd"),
        "portfolio_week":   pf.get("portfolio_week"),
        "benchmark_week":   pf.get("benchmark_week"),
        "smallcap_week":    pf.get("smallcap_week"),
        "smallcap_mtd":     pf.get("smallcap_mtd"),
        "smallcap_ytd":     pf.get("smallcap_ytd"),
        "strategy_label":   sk.get("label", "Portfolio (equal-weight)"),
        "benchmark_label":  bk.get("label", pf.get("benchmark_label", "Benchmark")),
        "sectors":          sectors,
        "mcaps":            mcaps,
        "dates":             dates,
        "strategy_curve":    strat,
        "benchmark_curve":   bench,
        "benchmark2_curve":  bench2,
        "monthly_returns":   bt_result.get("monthly_returns", []),
        "as_of":            pf.get("last_trading_day") or pf.get("as_of", ""),
    }
