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
import hashlib
import os
import pickle
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

from config import UNIVERSE_CSV, UNIVERSE_CSV_2

BENCHMARK = "^CRSLDX"
BENCHMARKS = {"india": "^CRSLDX", "us": "^GSPC"}   # per-market index
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


def load_universe(market: str = "india") -> list[str]:
    """Tickers for the requested market. US → S&P 500, India → NSE universe CSVs."""
    if market == "us":
        return load_universe_us()
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


# Minimal S&P 500 fallback (large caps) if the constituents fetch/cache both fail.
_SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","TSLA",
    "JPM","V","UNH","XOM","MA","COST","HD","PG","JNJ","ABBV","WMT","NFLX",
    "BAC","CRM","ORCL","KO","AMD","CVX","PEP","WFC","ADBE","LIN","ACN","MRK",
    "TMO","CSCO","MCD","ABT","INTC","QCOM","TXN","DHR","INTU","AMAT","MU",
]
_SP500_CACHE = os.path.join(os.path.dirname(__file__), "sp500_constituents.csv")
_SP500_URL = ("https://raw.githubusercontent.com/datasets/"
              "s-and-p-500-companies/main/data/constituents.csv")


def load_universe_us() -> list[str]:
    """S&P 500 constituents. Refreshes the local cache from GitHub when possible,
    else uses the cached CSV, else a hard-coded large-cap fallback."""
    import urllib.request, io
    try:
        req = urllib.request.Request(_SP500_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=20).read().decode()
        rows = list(csv.DictReader(io.StringIO(data)))
        syms = [r["Symbol"].replace(".", "-").strip() for r in rows if r.get("Symbol")]
        if len(syms) > 400:
            with open(_SP500_CACHE, "w", encoding="utf-8") as f:
                f.write(data)
            return syms
    except Exception:
        pass
    try:
        with open(_SP500_CACHE, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        syms = [r["Symbol"].replace(".", "-").strip() for r in rows if r.get("Symbol")]
        if syms:
            return syms
    except Exception:
        pass
    return _SP500_FALLBACK


# ── Data download ─────────────────────────────────────────────────────────────

_BATCH_SIZE = 100
_BATCH_PAUSE = 5
_PRICE_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".price_cache")


def _price_cache_key(tickers, benchmark, years) -> str:
    raw = "|".join(sorted(tickers)) + f"|{benchmark}|{years}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _load_price_cache(key: str):
    """Return cached (close, volume, high, low) if cached TODAY, else None.
    Same-day reuse makes re-runs identical (deterministic) and instant, and
    avoids re-hammering yfinance."""
    path = os.path.join(_PRICE_CACHE_DIR, f"{key}.pkl")
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("date") == date.today().isoformat():
            return blob["close"], blob["volume"], blob["high"], blob["low"]
    except Exception:
        pass
    return None


def _save_price_cache(key: str, close, volume, high, low):
    try:
        os.makedirs(_PRICE_CACHE_DIR, exist_ok=True)
        with open(os.path.join(_PRICE_CACHE_DIR, f"{key}.pkl"), "wb") as f:
            pickle.dump({"date": date.today().isoformat(), "close": close,
                         "volume": volume, "high": high, "low": low}, f)
    except Exception:
        pass


def download_data(tickers: list[str], years: int = 5,
                  include_hl: bool = False, benchmark: str | None = None,
                  progress_cb=None):
    """
    Download daily OHLCV for all tickers + benchmark.
    Returns (close, volume) by default; (close, volume, high, low) when include_hl=True.
    Batched to avoid yfinance rate limiting; benchmark fetched first.
    progress_cb(done_batches, total_batches, msg) is called as batches complete.
    """
    benchmark = benchmark or BENCHMARK
    def _dlp(done, total, msg):
        if progress_cb: progress_cb(done, total, msg)

    # Same-day price cache → deterministic, instant re-runs, no re-download.
    cache_key = _price_cache_key(tickers, benchmark, years)
    cached = _load_price_cache(cache_key)
    if cached is not None:
        _dlp(1, 1, "Loaded prices from today's cache")
        close, volume, high, low = cached
        return (close, volume, high, low) if include_hl else (close, volume)

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
    _dlp(0, 1, f"Downloading benchmark {benchmark} …")
    close_bench, vol_bench, high_bench, low_bench = None, None, pd.DataFrame(), pd.DataFrame()
    for attempt in range(2):
        try:
            close_bench, vol_bench, high_bench, low_bench = _fetch([benchmark])
            if not close_bench.empty:
                break
        except Exception:
            pass
        time.sleep(10)
    if close_bench is None or close_bench.empty:
        raise RuntimeError(f"Cannot download benchmark {benchmark} — check internet.")

    batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
    n_batches = len(batches)
    close_frames  = [close_bench]
    volume_frames = [vol_bench]
    high_frames   = [high_bench]
    low_frames    = [low_bench]
    failed_batches = []
    for idx, batch in enumerate(batches):
        if idx > 0:
            time.sleep(_BATCH_PAUSE)
        _dlp(idx, n_batches, f"Downloading prices… batch {idx + 1}/{n_batches}")
        # Retry each batch with backoff. Dropping a batch silently changes the
        # universe between runs (yfinance rate-limits unevenly), which makes the
        # backtest non-deterministic — so we retry instead of skipping.
        got = False
        for attempt in range(3):
            try:
                c, v, h, l = _fetch(batch)
                if not c.empty:
                    close_frames.append(c)
                    volume_frames.append(v)
                    high_frames.append(h)
                    low_frames.append(l)
                    got = True
                    break
            except Exception:
                pass
            if attempt < 2:
                time.sleep(3 * (attempt + 1))   # 3s, 6s backoff between retries
        if not got:
            failed_batches.append(idx + 1)
        _dlp(idx + 1, n_batches, f"Downloaded {idx + 1}/{n_batches} batches")
    if failed_batches:
        print(f"[download_data] WARNING: {len(failed_batches)} batch(es) failed "
              f"after retries: {failed_batches} — universe may be incomplete this run")

    close  = pd.concat(close_frames,  axis=1)
    volume = pd.concat(volume_frames, axis=1)
    close  = close.loc[:,  ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]

    close.index  = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)

    # Filter on recent activity (last ~1 year), not total download length.
    recent_win = min(252, len(close))
    recent_ok  = close.iloc[-recent_win:].notna().sum() >= (recent_win // 2)
    keep_bench = close[[benchmark]].copy() if benchmark in close.columns else None
    close  = close.loc[:, recent_ok]
    if keep_bench is not None and benchmark not in close.columns:
        close[benchmark] = keep_bench[benchmark]
    volume = volume.reindex(columns=close.columns).fillna(0)

    # Always build high/low so the cached panel is complete regardless of caller.
    high = pd.concat(high_frames, axis=1)
    low  = pd.concat(low_frames,  axis=1)
    high = high.loc[:, ~high.columns.duplicated()]
    low  = low.loc[:,  ~low.columns.duplicated()]
    high.index = pd.to_datetime(high.index).tz_localize(None)
    low.index  = pd.to_datetime(low.index).tz_localize(None)
    high = high.reindex(columns=close.columns)
    low  = low.reindex(columns=close.columns)

    # Cache only a reasonably-complete download (don't persist a run badly
    # truncated by rate-limiting).
    if not failed_batches:
        _save_price_cache(cache_key, close, volume, high, low)

    return (close, volume, high, low) if include_hl else (close, volume)

