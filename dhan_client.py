"""
dhan_client.py — thin wrapper around the DhanHQ v2 REST API.

Deliberately raw `requests` calls, not the third-party `dhanhq` SDK — every
request this module sends with your real money is visible in this file, not
hidden inside a dependency.

Requires DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in .env (see config.py). Order
placement/modification/cancellation additionally require a static IP
whitelisted with Dhan — GET-only calls (holdings, funds, LTP) don't.
"""
import os
import pickle
import time
from datetime import date

import pandas as pd
import requests

from config import DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN

BASE_URL     = "https://api.dhan.co/v2"
_MASTER_URL  = "https://images.dhan.co/api-data/api-scrip-master.csv"
_CACHE_DIR   = os.path.join(os.path.dirname(__file__), ".dhan_cache")
_ORDER_DELAY = 0.2   # seconds between orders in a batch — Dhan allows 10/sec, this stays well under it


class DhanAPIError(RuntimeError):
    """A Dhan API call failed. `status` is the HTTP status code; `detail` is
    Dhan's own error message when it returned one (e.g. an auth failure, an
    order rejected for an invalid tick, an unwhitelisted IP)."""
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"Dhan API error {status}: {detail}")


def _headers() -> dict:
    return {
        "access-token": DHAN_ACCESS_TOKEN,
        "client-id":    DHAN_CLIENT_ID,
        "Content-Type": "application/json",
    }


def _raise_for_status(r: requests.Response) -> None:
    if r.ok:
        return
    try:
        body = r.json()
        detail = body.get("errorMessage") or body.get("remarks") or body.get("message") or str(body)
    except Exception:
        detail = r.text[:300]
    raise DhanAPIError(r.status_code, detail)


def _require_configured() -> None:
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        raise DhanAPIError(0, "DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set in .env")


def _get(path: str, **kw) -> dict:
    _require_configured()
    r = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=15, **kw)
    _raise_for_status(r)
    return r.json()


def _post(path: str, body: dict) -> dict:
    _require_configured()
    r = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=body, timeout=15)
    _raise_for_status(r)
    return r.json()


def _instrument_master() -> pd.DataFrame:
    """NSE cash-equity instruments: symbol -> security_id, tick_size (rupees),
    lot_size. Cached on disk for the day (this file changes rarely intraday).

    Dhan's SEM_TICK_SIZE for equities is in paise (divide by 100 for rupees)
    — confirmed empirically, not from docs (which are inconsistent on this):
    the NIFTY index row already reports 0.05 correctly with no scaling, while
    equity rows report whole numbers (1, 5, 10, 50, 100, 500) that only match
    NSE's real published tick-size price bands after dividing by 100."""
    cache_path = os.path.join(_CACHE_DIR, "instrument_master.pkl")
    today = date.today().isoformat()
    try:
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("date") == today:
            return blob["df"]
    except Exception:
        pass

    raw = pd.read_csv(_MASTER_URL, low_memory=False)
    eq = raw[(raw["SEM_EXM_EXCH_ID"] == "NSE") &
             (raw["SEM_SEGMENT"] == "E") &
             (raw["SEM_SERIES"] == "EQ")].copy()
    eq["tick_size"] = eq["SEM_TICK_SIZE"] / 100.0
    df = eq.set_index("SEM_TRADING_SYMBOL")[["SEM_SMST_SECURITY_ID", "tick_size", "SEM_LOT_UNITS"]] \
           .rename(columns={"SEM_SMST_SECURITY_ID": "security_id", "SEM_LOT_UNITS": "lot_size"})

    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"date": today, "df": df}, f)
    except Exception:
        pass
    return df


def security_id(symbol: str) -> str:
    """Bare NSE symbol (no .NS suffix) -> Dhan security ID."""
    row = _instrument_master().loc[symbol]
    return str(int(row["security_id"]))


def _snap_to_tick(price: float, symbol: str) -> float:
    tick = float(_instrument_master().loc[symbol, "tick_size"])
    if tick <= 0:
        return round(price, 2)
    return round(round(price / tick) * tick, 2)


def get_holdings() -> dict:
    """Real current holdings from Dhan. {symbol: {'qty': int, 'avg_cost': float}}
    — source of truth, never a locally-cached ledger that can drift."""
    rows = _get("/holdings")
    return {r["tradingSymbol"]: {"qty": int(r["totalQty"]), "avg_cost": float(r["avgCostPrice"])}
            for r in rows}


def get_available_cash() -> float:
    r = _get("/fundlimit")
    return float(r["availabelBalance"])   # sic — Dhan's own field name is misspelled


def get_ltp(symbols: list[str]) -> dict:
    """Live last-traded price for bare NSE symbols. {symbol: price}. Skips
    any symbol not in the instrument master rather than raising."""
    master = _instrument_master()
    sec_ids = {}
    for s in symbols:
        if s in master.index:
            sec_ids[str(int(master.loc[s, "security_id"]))] = s

    if not sec_ids:
        return {}
    resp = _post("/marketfeed/ltp", {"NSE_EQ": [int(i) for i in sec_ids]})
    out = {}
    for sec_id, data in resp.get("data", {}).get("NSE_EQ", {}).items():
        sym = sec_ids.get(sec_id)
        if sym:
            out[sym] = float(data["last_price"])
    return out


def place_order(symbol: str, transaction_type: str, quantity: int, price: float,
                 product_type: str = "CNC") -> dict:
    """transaction_type: 'BUY' or 'SELL'. Snaps `price` to the instrument's
    real tick size before sending — an unsnapped limit price gets rejected."""
    sec_id = security_id(symbol)
    limit_price = _snap_to_tick(price, symbol)
    body = {
        "dhanClientId":     DHAN_CLIENT_ID,
        "transactionType":  transaction_type,
        "exchangeSegment":  "NSE_EQ",
        "productType":      product_type,
        "orderType":        "LIMIT",
        "validity":         "DAY",
        "securityId":       sec_id,
        "quantity":         int(quantity),
        "price":            limit_price,
    }
    return _post("/orders", body)


def place_orders(orders: list[dict]) -> list[dict]:
    """Place a batch of {'symbol','side','qty','price'} orders sequentially,
    pausing between each (well under Dhan's 10/sec limit). One rejected order
    doesn't block the rest — each result is tracked independently so a failed
    order can be identified and retried."""
    results = []
    for i, o in enumerate(orders):
        try:
            resp = place_order(o["symbol"], o["side"], o["qty"], o["price"])
            results.append({**o, "status": "sent",
                             "order_id": resp.get("orderId"),
                             "order_status": resp.get("orderStatus")})
        except Exception as e:
            results.append({**o, "status": "failed", "error": str(e)})
        if i < len(orders) - 1:
            time.sleep(_ORDER_DELAY)
    return results
