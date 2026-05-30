"""
ha_backtest.py — Heikin-Ashi 30-min positional backtest for NIFTY.

Strategy:
  - Bullish HA bar (no lower wick): go LONG at next bar's open
  - Bearish HA bar (no upper wick): go SHORT at next bar's open
  - Reverse on opposite signal; stop-loss optional
  - Modes: futures | options_sell | options_buy (synthetic Black-Scholes + VIX)

Data source: Kite Connect historical API (30-min candles).
"""
from __future__ import annotations

import math
import sys
import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

NIFTY_TOKEN    = 256265
NIFTY_LOT_SIZE = 65
ATM_STEP       = 50
EPS            = 0.05
IST            = ZoneInfo("Asia/Kolkata")


# ── Heikin-Ashi ───────────────────────────────────────────────────────────────

def _ha(df: pd.DataFrame) -> pd.DataFrame:
    ha       = df.copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open  = [0.0] * len(df)
    ha_open[0] = (float(df["open"].iloc[0]) + float(df["close"].iloc[0])) / 2.0
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + float(ha_close.iloc[i - 1])) / 2.0
    ha["ha_close"] = ha_close.values
    ha["ha_open"]  = ha_open
    ha["ha_high"]  = ha[["high", "ha_open", "ha_close"]].max(axis=1)
    ha["ha_low"]   = ha[["low",  "ha_open", "ha_close"]].min(axis=1)
    return ha


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_data(kite, from_date: date, to_date: date) -> pd.DataFrame:
    """Fetch 30-min NIFTY candles in 60-day chunks."""
    rows = []
    cur  = from_date
    while cur <= to_date:
        end     = min(cur + timedelta(days=59), to_date)
        from_dt = datetime.combine(cur, time(9, 15), tzinfo=IST)
        to_dt   = datetime.combine(end, time(15, 30), tzinfo=IST)
        raw     = kite.historical_data(NIFTY_TOKEN, from_dt, to_dt, "30minute")
        if raw:
            rows.extend(raw)
        cur = end + timedelta(days=1)

    if not rows:
        return pd.DataFrame()
    df         = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["day"]  = df["date"].dt.date
    df["t"]    = df["date"].dt.time
    return df.reset_index(drop=True)


# ── Futures backtest ──────────────────────────────────────────────────────────

def _run_futures(df: pd.DataFrame, sl_pts: float, lots: int) -> list[dict]:
    ha          = _ha(df.reset_index(drop=True))
    trades      = []
    pos         = None
    pending_dir = None

    def _fmt(dt) -> str:
        """Format datetime as 'YYYY-MM-DD HH:MM'."""
        if hasattr(dt, "strftime"):
            return dt.strftime("%Y-%m-%d %H:%M")
        return str(dt)

    def _close(p, dt, price, reason):
        mult = 1 if p["dir"] == "LONG" else -1
        entry_dt = p["entry_dt"]
        days_held = (dt.date() - entry_dt.date()).days if hasattr(dt, "date") else 0
        return {
            "entry_date":  _fmt(entry_dt),
            "exit_date":   _fmt(dt),
            "days_held":   days_held,
            "direction":   p["dir"],
            "entry_price": round(p["entry_price"], 2),
            "exit_price":  round(price, 2),
            "qty":         NIFTY_LOT_SIZE * lots,
            "pnl":         round(mult * (price - p["entry_price"]) * NIFTY_LOT_SIZE * lots, 2),
            "exit_reason": reason,
        }

    for i in range(len(ha)):
        row     = ha.iloc[i]
        dt      = row["date"]   # full datetime
        open_p  = float(row["open"])
        ha_o    = float(row["ha_open"])
        bullish = abs(ha_o - float(row["ha_low"]))  <= EPS
        bearish = abs(ha_o - float(row["ha_high"])) <= EPS

        if pending_dir is not None:
            if pos is not None:
                trades.append(_close(pos, dt, open_p, "reversal"))
            sl_val = open_p - sl_pts if pending_dir == "LONG" else open_p + sl_pts
            pos = {"dir": pending_dir, "entry_dt": dt,
                   "entry_price": open_p, "sl": sl_val}
            pending_dir = None

        if pos is None:
            if bullish:   pending_dir = "LONG"
            elif bearish: pending_dir = "SHORT"
            continue

        close_p  = float(row["close"])
        sl_hit   = sl_pts > 0 and (
            (pos["dir"] == "LONG"  and close_p <= pos["sl"]) or
            (pos["dir"] == "SHORT" and close_p >= pos["sl"])
        )
        reversal = (pos["dir"] == "LONG" and bearish) or \
                   (pos["dir"] == "SHORT" and bullish)

        if sl_hit:
            trades.append(_close(pos, dt, pos["sl"], "SL"))
            pos = None
        elif reversal:
            pending_dir = "LONG" if bullish else "SHORT"

    if pos is not None:
        last = ha.iloc[-1]
        trades.append(_close(pos, last["date"], float(last["close"]), "series_end"))
    return trades


# ── KPIs ──────────────────────────────────────────────────────────────────────

def _compute_kpis(trades: list[dict]) -> dict:
    if not trades:
        return {"total_trades": 0, "win_rate_pct": 0, "total_pnl": 0,
                "avg_pnl": 0, "avg_win": 0, "avg_loss": 0,
                "max_win": 0, "max_loss": 0, "profit_factor": 0,
                "max_drawdown_pct": 0, "expectancy": 0}

    pnls    = [t["pnl"] for t in trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]

    total_pnl    = sum(pnls)
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))

    # Running max drawdown on cumulative PnL
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = float(dd.min()) if len(dd) else 0

    return {
        "total_trades":    len(trades),
        "win_rate_pct":    round(len(wins) / len(trades) * 100, 1),
        "total_pnl":       round(total_pnl, 2),
        "avg_pnl":         round(total_pnl / len(trades), 2),
        "avg_win":         round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss":        round(sum(losses) / len(losses), 2) if losses else 0,
        "max_win":         round(max(pnls), 2),
        "max_loss":        round(min(pnls), 2),
        "profit_factor":   round(gross_profit / gross_loss, 2) if gross_loss else 0,
        "max_drawdown":    round(max_dd, 2),
        "expectancy":      round(total_pnl / len(trades), 2),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run_backtest(
    from_date: date,
    to_date:   date,
    mode:      str   = "futures",   # futures | options_sell | options_buy
    sl_pts:    float = 0,
    sl_pct:    float = 0.5,
    lots:      int   = 1,
) -> dict:
    import kite_auth
    kite = kite_auth.get_kite()

    df = fetch_data(kite, from_date, to_date)
    if df.empty:
        raise ValueError("No data returned from Kite — check dates and token.")

    if mode == "futures":
        trades = _run_futures(df, sl_pts, lots)
    else:
        # Import options engine from the straddle project
        sys.path.insert(0, os.path.expanduser("~/straddle_vwap_forward"))
        from backtest.engine import run_options_synthetic
        trades = run_options_synthetic(kite, df, sl_pct, lots,
                                       mode="sell" if mode == "options_sell" else "buy")
        # Normalise date fields to strings
        for t in trades:
            t["entry_date"] = str(t.get("entry_date", ""))
            t["exit_date"]  = str(t.get("exit_date",  ""))

    kpis = _compute_kpis(trades)

    # Equity curve (cumulative PnL)
    cum_pnl = list(np.cumsum([t["pnl"] for t in trades]).tolist()) if trades else []

    return {
        "params": {
            "from_date": str(from_date), "to_date": str(to_date),
            "mode": mode, "sl_pts": sl_pts, "sl_pct": sl_pct, "lots": lots,
        },
        "kpis":       kpis,
        "trades":     trades,
        "equity_curve": [round(v, 2) for v in cum_pnl],
        "run_at":     datetime.now().isoformat(timespec="seconds"),
    }
