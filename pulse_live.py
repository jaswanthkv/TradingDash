"""
pulse_live.py — Live signal and options execution for Pulse strategy.

Signal logic (same as backtest):
  Bullish HA bar (no lower wick) → LONG  → Sell ATM-500 PE (monthly)
  Bearish HA bar (no upper wick) → SHORT → Sell ATM+500 CE (monthly)

Monthly expiry = last Thursday of the current/next month.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

from ha_backtest import _ha, fetch_data, EPS, ATM_STEP, NIFTY_LOT_SIZE
import kite_auth


def _kite():
    return kite_auth.get_kite()


# ── Monthly expiry ─────────────────────────────────────────────────────────────

def monthly_expiry(kite) -> date | None:
    """Return the nearest upcoming monthly NIFTY options expiry (last Thursday)."""
    instruments = kite.instruments("NFO")
    today = date.today()

    expiries = sorted(set(
        i["expiry"] for i in instruments
        if i["name"] == "NIFTY"
        and i["instrument_type"] in ("CE", "PE")
        and i["expiry"] >= today
    ))

    for exp in expiries:
        if exp.weekday() == 1:                          # Tuesday
            next_tue = exp + timedelta(days=7)
            if next_tue.month != exp.month:             # last Tuesday of month
                return exp
    return None


# ── Live signal ────────────────────────────────────────────────────────────────

def current_signal(kite) -> dict:
    """
    Fetch last 10 trading days of NIFTY 30-min data, compute HA.

    Returns:
      - candle_signal: what the latest candle says (LONG / SHORT / FLAT)
      - active_signal: most recent definitive signal — this is the current position direction.
                       FLAT only if no signal has ever fired in the window.
      - active_since:  datetime of the candle that triggered the active signal
    """
    today     = date.today()
    from_date = today - timedelta(days=10)

    df = fetch_data(kite, from_date, today)
    if df.empty:
        return {"candle_signal": "FLAT", "active_signal": "FLAT", "reason": "No data from Kite"}

    ha = _ha(df.reset_index(drop=True))

    def _sig(row):
        ha_o = float(row["ha_open"])
        bullish = abs(ha_o - float(row["ha_low"]))  <= EPS
        bearish = abs(ha_o - float(row["ha_high"])) <= EPS
        return "LONG" if bullish else ("SHORT" if bearish else "FLAT")

    last = ha.iloc[-1]
    ha_o = float(last["ha_open"])
    ha_h = float(last["ha_high"])
    ha_l = float(last["ha_low"])
    ha_c = float(last["ha_close"])
    candle_signal = _sig(last)

    # Walk back to find the most recent non-FLAT candle
    active_signal = "FLAT"
    active_since  = None
    for _, row in ha.iloc[::-1].iterrows():
        s = _sig(row)
        if s != "FLAT":
            active_signal = s
            active_since  = str(row["date"])
            break

    return {
        "candle_signal": candle_signal,
        "active_signal": active_signal,
        "active_since":  active_since,
        "candle_time":   str(last["date"]),
        "nifty_price":   round(float(last["close"]), 2),
        "ha_open":       round(ha_o, 2),
        "ha_close":      round(ha_c, 2),
        "ha_high":       round(ha_h, 2),
        "ha_low":        round(ha_l, 2),
    }


# ── Option selection ───────────────────────────────────────────────────────────

def option_details(kite, signal: str, nifty_price: float, expiry: date) -> dict | None:
    """Return instrument info + LTP for the option to sell."""
    if signal == "FLAT":
        return None

    atm    = round(nifty_price / ATM_STEP) * ATM_STEP
    strike = (atm - 500) if signal == "LONG" else (atm + 500)
    otype  = "PE" if signal == "LONG" else "CE"

    instruments = kite.instruments("NFO")
    opt = next(
        (i for i in instruments
         if i["name"] == "NIFTY"
         and i["instrument_type"] == otype
         and i["expiry"] == expiry
         and int(i["strike"]) == int(strike)),
        None,
    )
    if not opt:
        return {"error": f"Instrument not found: NIFTY {expiry} {strike} {otype}"}

    sym   = opt["tradingsymbol"]
    quote = kite.quote(f"NFO:{sym}")
    ltp   = quote.get(f"NFO:{sym}", {}).get("last_price", 0)

    return {
        "tradingsymbol": sym,
        "strike":        int(strike),
        "opt_type":      otype,
        "expiry":        str(expiry),
        "atm":           int(atm),
        "ltp":           round(ltp, 2),
    }


# ── Current NFO positions ──────────────────────────────────────────────────────

def nifty_positions(kite) -> list[dict]:
    """Return open NIFTY option positions (net qty != 0)."""
    pos = kite.positions().get("net", [])
    result = []
    for p in pos:
        if "NIFTY" not in p.get("tradingsymbol", ""):
            continue
        qty = p.get("quantity", 0)
        if qty == 0:
            continue
        result.append({
            "tradingsymbol": p["tradingsymbol"],
            "quantity":      qty,
            "average_price": round(p.get("average_price", 0), 2),
            "last_price":    round(p.get("last_price", 0), 2),
            "pnl":           round(p.get("pnl", 0), 2),
        })
    return result


# ── Order placement ────────────────────────────────────────────────────────────

def _nfo_ltp(kite, tradingsymbol: str) -> float:
    quote = kite.quote(f"NFO:{tradingsymbol}")
    return quote.get(f"NFO:{tradingsymbol}", {}).get("last_price", 0.0)


def _tick(price: float, buffer_pct: float) -> float:
    """Round to nearest 0.05 tick after applying a buffer."""
    raw = price * buffer_pct
    return max(round(round(raw / 0.05) * 0.05, 2), 0.05)


def sell_option(kite, tradingsymbol: str, lots: int) -> str:
    """Sell (write) a NIFTY option at limit = LTP * 0.98. Returns order_id."""
    qty   = lots * NIFTY_LOT_SIZE
    ltp   = _nfo_ltp(kite, tradingsymbol)
    price = _tick(ltp, 0.98)   # 2% below LTP — fills quickly, avoids rejection
    return kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NFO,
        tradingsymbol=tradingsymbol,
        transaction_type=kite.TRANSACTION_TYPE_SELL,
        quantity=qty,
        order_type=kite.ORDER_TYPE_LIMIT,
        price=price,
        product=kite.PRODUCT_NRML,
    )


def close_option(kite, tradingsymbol: str, qty: int) -> str:
    """Buy back a short option position at limit = LTP * 1.02. qty should be positive."""
    ltp   = _nfo_ltp(kite, tradingsymbol)
    price = _tick(ltp, 1.02)   # 2% above LTP — fills quickly
    return kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NFO,
        tradingsymbol=tradingsymbol,
        transaction_type=kite.TRANSACTION_TYPE_BUY,
        quantity=abs(qty),
        order_type=kite.ORDER_TYPE_LIMIT,
        price=price,
        product=kite.PRODUCT_NRML,
    )
