"""
strategy.py — Vectorized ML multi-factor stock ranking engine.

Factors (cross-sectional percentile ranks, 0–100):
  mom_12_1  : 12-month return skipping last month  [25%]
  mom_6_1   : 6-month return skipping last month   [20%]
  rs_nifty  : 3-month excess return vs Nifty 50    [20%]
  trend     : EMA/SMA alignment score 0–4          [15%]
  sharpe_3m : 63-day rolling Sharpe ratio          [10%]
  vol_exp   : volume expansion 5d / 60d avg        [ 5%]
  inv_vol   : negative 20-day realized volatility  [ 5%]
"""
import warnings; warnings.filterwarnings("ignore")
import csv
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

from config import UNIVERSE_CSV, UNIVERSE_CSV_2

BENCHMARK = "^CRSLDX"
RISK_FREE  = 0.06      # 6% annualised (India)
TOP_N      = 20
MIN_BARS   = 274       # 252 + 22 needed for mom_12_1

WEIGHTS = {
    "mom_12_1": 0.25,
    "mom_6_1":  0.20,
    "rs_nifty": 0.20,
    "trend":    0.15,
    "sharpe_3m":0.10,
    "vol_exp":  0.05,
    "inv_vol":  0.05,
}

FACTOR_LABELS = {
    "mom_12_1":  "12M Mom",
    "mom_6_1":   "6M Mom",
    "rs_nifty":  "RS vs Nifty",
    "trend":     "Trend",
    "sharpe_3m": "Sharpe 3M",
    "vol_exp":   "Vol Exp",
    "inv_vol":   "Low Vol",
}

# Nifty 100 fallback when universe CSV is absent
_NIFTY100 = [
    "RELIANCE","TCS","HDFCBANK","BHARTIARTL","ICICIBANK","SBIN","INFY",
    "HINDUNILVR","ITC","LT","BAJFINANCE","KOTAKBANK","HCLTECH","MARUTI",
    "AXISBANK","TITAN","SUNPHARMA","NTPC","POWERGRID","WIPRO","ADANIENT",
    "ONGC","NESTLEIND","ASIANPAINT","JSWSTEEL","TATAMOTORS","TECHM","M&M",
    "BPCL","CIPLA","DRREDDY","EICHERMOT","HINDALCO","DIVISLAB","APOLLOHOSP",
    "TATACONSUM","COALINDIA","BAJAJFINSV","INDUSINDBK","SHRIRAMFIN","ADANIPORTS",
    "BEL","BAJAJ-AUTO","BRITANNIA","HEROMOTOCO","TRENT","VEDL","MOTHERSON",
    "PFC","RECLTD","TATAPOWER","CANBK","SAIL","NMDC","BHEL","SIEMENS",
    "HAVELLS","MUTHOOTFIN","BANKBARODA","LICHSGFIN","IRFC","PNB","IOC",
    "GAIL","COLPAL","MARICO","PIDILITIND","BERGEPAINT","GODREJCP","DABUR",
    "MCDOWELL-N","TATACOMM","OFSS","MPHASIS","PERSISTENT","COFORGE","LTIM",
    "IPCALAB","TORNTPHARM","AUROPHARMA","LUPIN","BIOCON","ALKEM","GLENMARK",
    "ZYDUSLIFE","MANKIND","METROPOLIS","LAURUSLABS","GRANULES","NATCOPHARM",
    "ABCAPITAL","CHOLAMANDLAM","M&MFIN","IDFCFIRSTB","FEDERALBNK","RBLBANK",
]


# ── Universe ──────────────────────────────────────────────────────────────────

def _read_csv(path: str) -> list[str]:
    """
    Parse one NSE universe CSV and return a list of '.NS' tickers.

    Supports two formats:
      - NSE index list  (ind_nifty500list.csv): header has 'Symbol' col; EQ series only
      - NSE market-watch (MW-NIFTY-*.csv):      symbol at col 0, skip first two rows
    """
    tickers = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_clean = [h.strip().lower() for h in header]
        if "symbol" in header_clean:
            sym_col = header_clean.index("symbol")
            ser_col = header_clean.index("series") if "series" in header_clean else None
            for row in reader:
                if not row:
                    continue
                if ser_col is not None and row[ser_col].strip() != "EQ":
                    continue
                sym = row[sym_col].strip()
                if sym:
                    tickers.append(sym + ".NS")
        else:
            next(reader)   # skip aggregate row in MW format
            for row in reader:
                sym = row[0].strip() if row else ""
                if sym and " " not in sym:
                    tickers.append(sym + ".NS")
    return tickers


def load_universe() -> list[str]:
    """Load and merge tickers from configured universe CSVs. Falls back to Nifty 100."""
    seen, tickers = set(), []
    for path in [UNIVERSE_CSV, UNIVERSE_CSV_2]:
        if not path:
            continue
        try:
            for t in _read_csv(path):
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
        except Exception:
            pass
    return tickers if tickers else [s + ".NS" for s in _NIFTY100]


# ── Data download ─────────────────────────────────────────────────────────────

_BATCH_SIZE = 100
_BATCH_PAUSE = 5


def download_data(tickers: list[str], years: int = 5,
                  include_hl: bool = False):
    """
    Download daily OHLCV for all tickers + benchmark.
    Returns (close, volume) by default; (close, volume, high, low) when include_hl=True.
    Batched to avoid yfinance rate limiting; benchmark fetched first.
    """
    warmup = int(years * 365) + 430
    start  = (date.today() - timedelta(days=warmup)).strftime("%Y-%m-%d")

    def _fetch(syms):
        raw = yf.download(syms, start=start, auto_adjust=True, progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            c = raw["Close"].copy()
            v = raw["Volume"].copy()
            h = raw["High"].copy()  if "High"  in raw.columns.get_level_values(0) else pd.DataFrame()
            l = raw["Low"].copy()   if "Low"   in raw.columns.get_level_values(0) else pd.DataFrame()
        else:
            c = raw[["Close"]].copy()
            v = raw[["Volume"]].copy()
            h = raw[["High"]].copy()  if "High"  in raw.columns else pd.DataFrame()
            l = raw[["Low"]].copy()   if "Low"   in raw.columns else pd.DataFrame()
        return c, v, h, l

    # Benchmark first, retry once if rate-limited
    close_bench, vol_bench, high_bench, low_bench = None, None, pd.DataFrame(), pd.DataFrame()
    for attempt in range(2):
        try:
            close_bench, vol_bench, high_bench, low_bench = _fetch([BENCHMARK])
            if not close_bench.empty:
                break
        except Exception:
            pass
        time.sleep(10)
    if close_bench is None or close_bench.empty:
        raise RuntimeError("Cannot download Nifty 500 benchmark — check internet.")

    batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
    close_frames  = [close_bench]
    volume_frames = [vol_bench]
    high_frames   = [high_bench]
    low_frames    = [low_bench]
    for idx, batch in enumerate(batches):
        if idx > 0:
            time.sleep(_BATCH_PAUSE)
        try:
            c, v, h, l = _fetch(batch)
            close_frames.append(c)
            volume_frames.append(v)
            high_frames.append(h)
            low_frames.append(l)
        except Exception:
            pass  # skip failed batches

    close  = pd.concat(close_frames,  axis=1)
    volume = pd.concat(volume_frames, axis=1)
    close  = close.loc[:,  ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    close.index  = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)

    # Filter on recent activity (last ~1 year), not total download length.
    recent_win = min(252, len(close))
    recent_ok  = close.iloc[-recent_win:].notna().sum() >= (recent_win // 2)
    keep_bench = close[[BENCHMARK]].copy() if BENCHMARK in close.columns else None
    close  = close.loc[:, recent_ok]
    if keep_bench is not None and BENCHMARK not in close.columns:
        close[BENCHMARK] = keep_bench[BENCHMARK]
    volume = volume.reindex(columns=close.columns).fillna(0)

    if not include_hl:
        return close, volume

    high = pd.concat(high_frames, axis=1)
    low  = pd.concat(low_frames,  axis=1)
    high = high.loc[:, ~high.columns.duplicated()]
    low  = low.loc[:,  ~low.columns.duplicated()]
    high.index = pd.to_datetime(high.index).tz_localize(None)
    low.index  = pd.to_datetime(low.index).tz_localize(None)
    high = high.reindex(columns=close.columns)
    low  = low.reindex(columns=close.columns)
    return close, volume, high, low


# ── Factor computation (fully vectorised) ────────────────────────────────────

def compute_factors(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute all 7 factor DataFrames (dates × stocks) in one vectorised pass.
    Benchmark column is included in `close`; factors are computed on stocks only.
    """
    stocks = [c for c in close.columns if c != BENCHMARK]
    sc = close[stocks]
    sv = volume[stocks] if stocks else pd.DataFrame()
    bench = close[BENCHMARK] if BENCHMARK in close.columns else None

    daily_ret = sc.pct_change()

    # Momentum (skip last ~1 month to avoid short-term reversal)
    mom_12_1 = sc.shift(21).div(sc.shift(252 + 21)) - 1
    mom_6_1  = sc.shift(21).div(sc.shift(126 + 21)) - 1

    # Relative strength vs Nifty 50 (3-month)
    stock_3m = sc.div(sc.shift(63)) - 1
    if bench is not None:
        bench_3m = bench.div(bench.shift(63)) - 1
        rs_nifty = stock_3m.subtract(bench_3m, axis=0)
    else:
        rs_nifty = stock_3m

    # Trend: count of satisfied EMA/SMA stack conditions (0–4)
    ema5  = sc.ewm(span=5,  adjust=False).mean()
    ema10 = sc.ewm(span=10, adjust=False).mean()
    sma20 = sc.rolling(20).mean()
    sma50 = sc.rolling(50).mean()
    trend = (
        (sc    > ema5 ).astype(float) +
        (ema5  > ema10).astype(float) +
        (ema10 > sma20).astype(float) +
        (sma20 > sma50).astype(float)
    )

    # Volume expansion: recent 5-day avg vs 60-day avg
    vol_exp = sv.rolling(5).mean().div(sv.rolling(60).mean().replace(0, np.nan))

    # 3-month rolling Sharpe (63 trading days)
    rf_daily  = (1 + RISK_FREE) ** (1 / 252) - 1
    excess    = daily_ret - rf_daily
    sharpe_3m = (
        excess.rolling(63).mean()
        .div(daily_ret.rolling(63).std(ddof=1))
        .mul(np.sqrt(252))
    )

    # Inverse realised volatility (20-day); lower vol → higher rank
    inv_vol = -(daily_ret.rolling(20).std(ddof=1) * np.sqrt(252))

    return {
        "mom_12_1": mom_12_1,
        "mom_6_1":  mom_6_1,
        "rs_nifty": rs_nifty,
        "trend":    trend,
        "sharpe_3m":sharpe_3m,
        "vol_exp":  vol_exp,
        "inv_vol":  inv_vol,
    }


# ── Composite scoring ─────────────────────────────────────────────────────────

def _composite_score(snapshot: pd.DataFrame) -> pd.Series:
    """Weighted sum of cross-sectional percentile ranks (0–100)."""
    ranked  = snapshot.rank(pct=True, na_option="keep") * 100
    comp    = pd.Series(0.0, index=snapshot.index)
    total_w = 0.0
    for col, w in WEIGHTS.items():
        if col in ranked.columns:
            valid = ranked[col].notna()
            comp[valid] += ranked.loc[valid, col] * w
            total_w += w
    return (comp / total_w).round(2) if total_w else comp


# ── Date-point ranking ────────────────────────────────────────────────────────

def rank_on_date(
    factor_dfs: dict[str, pd.DataFrame],
    ref_date:   pd.Timestamp,
    close:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank all eligible stocks using factor values on or just before ref_date.
    Returns DataFrame sorted by composite_score descending.
    """
    stocks = [c for c in close.columns if c != BENCHMARK]

    avail = close.index[close.index <= ref_date]
    if avail.empty:
        return pd.DataFrame()
    dt = avail[-1]

    # Require MIN_BARS of history
    bars_avail = close[stocks].loc[:dt].notna().sum()
    eligible   = bars_avail[bars_avail >= MIN_BARS].index.tolist()
    if not eligible:
        return pd.DataFrame()

    snap_data = {}
    for fname, fdf in factor_dfs.items():
        if dt in fdf.index:
            snap_data[fname] = fdf.loc[dt, [t for t in eligible if t in fdf.columns]]

    if not snap_data:
        return pd.DataFrame()

    snapshot = pd.DataFrame(snap_data, index=eligible)
    score    = _composite_score(snapshot)
    prices   = close.loc[dt, eligible]

    out = snapshot.copy()
    out["composite_score"] = score
    out["price"] = prices
    out = out.sort_values("composite_score", ascending=False).reset_index()
    out.rename(columns={"index": "ticker"}, inplace=True)
    out["rank"] = range(1, len(out) + 1)
    out["in_top20"] = out["rank"] <= TOP_N
    return out


# ── Public entry point ────────────────────────────────────────────────────────

def get_current_rankings(years: int = 3) -> dict:
    """Full pipeline → current rankings + portfolio."""
    tickers    = load_universe()
    close, vol = download_data(tickers, years=years)
    factors    = compute_factors(close, vol)
    ranked     = rank_on_date(factors, pd.Timestamp(date.today()), close)

    def _fmt(v):
        if isinstance(v, float) and np.isnan(v): return None
        if isinstance(v, (float, np.floating)):  return round(float(v), 4)
        if isinstance(v, (bool, np.bool_)):      return bool(v)
        return v

    records = [{k: _fmt(v) for k, v in row.items()}
               for _, row in ranked.iterrows()]

    return {
        "as_of":          str(date.today()),
        "universe_size":  len([c for c in close.columns if c != BENCHMARK]),
        "eligible_count": len(ranked),
        "portfolio":      list(ranked.head(TOP_N)["ticker"]),
        "rankings":       records[:60],
        "factor_weights": WEIGHTS,
        "factor_labels":  FACTOR_LABELS,
    }
