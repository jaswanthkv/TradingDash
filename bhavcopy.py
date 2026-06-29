"""
bhavcopy.py — download & cache NSE F&O daily bhavcopy (REAL option settlement prices).

Handles both NSE layouts:
  • UDiFF  (>= 2024-07-08): BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
  • legacy (before):        fo{DDMONYYYY}bhav.csv.zip

Each trading day is filtered to index options and stored as a compact pickle in
.bhavcopy/.  This is the data source for a true (non-modelled) options backtest.
"""
import glob
import io
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

_DIR    = os.path.join(os.path.dirname(__file__), ".bhavcopy")
_INDEX  = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
_MON    = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_HDRS   = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/all-reports",
}

_session = None


def _sess():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(_HDRS)
        try:
            _session.get("https://www.nseindia.com", timeout=10)   # warm cookies
        except Exception:
            pass
    return _session


def _udiff_url(d: date) -> str:
    return (f"https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")


def _legacy_url(d: date) -> str:
    mon = _MON[d.month - 1]
    return (f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
            f"{d.year}/{mon}/fo{d.day:02d}{mon}{d.year}bhav.csv.zip")


def _parse(content: bytes, d: date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), compression="zip")
    if "TckrSymb" in df.columns:                          # UDiFF
        df = df[df["OptnTp"].isin(["CE", "PE"]) & df["TckrSymb"].isin(_INDEX)]
        return pd.DataFrame({
            "date":       pd.to_datetime(df["TradDt"]).dt.date,
            "symbol":     df["TckrSymb"].values,
            "expiry":     pd.to_datetime(df["XpryDt"]).dt.date,
            "strike":     df["StrkPric"].astype(float).values,
            "opt_type":   df["OptnTp"].values,
            "close":      df["ClsPric"].astype(float).values,
            "settle":     df["SttlmPric"].astype(float).values,
            "underlying": df["UndrlygPric"].astype(float).values,
            "oi":         df["OpnIntrst"].astype(float).values,
        })
    # legacy
    df = df[df["OPTION_TYP"].isin(["CE", "PE"]) & df["SYMBOL"].isin(_INDEX)]
    return pd.DataFrame({
        "date":       d,
        "symbol":     df["SYMBOL"].values,
        "expiry":     pd.to_datetime(df["EXPIRY_DT"]).dt.date,
        "strike":     df["STRIKE_PR"].astype(float).values,
        "opt_type":   df["OPTION_TYP"].values,
        "close":      df["CLOSE"].astype(float).values,
        "settle":     df["SETTLE_PR"].astype(float).values,
        "underlying": pd.NA,
        "oi":         df["OPEN_INT"].astype(float).values,
    })


def fetch_day(d: date, force: bool = False) -> str:
    """Download+cache one day. Returns 'cached' | 'ok' | 'miss' (holiday/weekend/NA)."""
    os.makedirs(_DIR, exist_ok=True)
    pq = os.path.join(_DIR, f"{d:%Y%m%d}.pkl")
    if os.path.exists(pq) and not force:
        return "cached"
    s = _sess()
    for url in (_udiff_url(d), _legacy_url(d)):
        for attempt in range(2):
            try:
                r = s.get(url, timeout=25)
                if r.status_code == 200 and r.content[:2] == b"PK":
                    _parse(r.content, d).to_pickle(pq)
                    return "ok"
                if r.status_code == 404:
                    break                      # genuinely not there → try other format
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))      # backoff (likely throttled)
    return "miss"


def download_range(start: date, end: date, pause: float = 0.4, progress=None) -> dict:
    """Download every trading day in [start, end]. Skips cached days; weekends skipped."""
    d, stats = start, {"ok": 0, "cached": 0, "miss": 0}
    while d <= end:
        if d.weekday() < 5:
            st = fetch_day(d)
            stats[st] += 1
            if progress:
                progress(d, st)
            if st == "ok":
                time.sleep(pause)
        d += timedelta(days=1)
    return stats


def status() -> dict:
    files = sorted(glob.glob(os.path.join(_DIR, "*.pkl")))
    return {"days": len(files),
            "first": os.path.basename(files[0])[:8] if files else None,
            "last":  os.path.basename(files[-1])[:8] if files else None}


def load(symbol: str = None, start: date = None, end: date = None) -> pd.DataFrame:
    """Concatenate cached days into one DataFrame, optionally filtered."""
    frames = [pd.read_pickle(pq) for pq in sorted(glob.glob(os.path.join(_DIR, "*.pkl")))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if symbol:
        df = df[df["symbol"] == symbol]
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df
