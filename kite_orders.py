"""
kite_orders.py — Order placement via Kite Connect.

Robustness:
  - Market hours check before every order
  - Margin sufficiency check before buy
  - 3-retry with exponential backoff on transient errors
  - SL-M order placed automatically after buy (if stop_loss provided)
  - Dry-run mode simulates without placing
"""
import time, datetime, logging
from kiteconnect import KiteConnect
from kite_auth import get_kite

logger = logging.getLogger(__name__)

_EXCHANGE  = "NSE"
_MAX_RETRY = 3
_BACKOFF   = 1.5   # seconds; retries at 0s, 1.5s, 2.25s


# ── market hours ──────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return datetime.time(9, 15) <= t <= datetime.time(15, 30)


def market_status() -> dict:
    now  = datetime.datetime.now()
    open_ = is_market_open()
    return {
        "open": open_,
        "message": (
            "Market open" if open_
            else "Market closed — NSE: 9:15 AM – 3:30 PM, Mon–Fri"
        ),
        "weekday": now.strftime("%A"),
        "time":    now.strftime("%H:%M"),
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _sym(ticker: str) -> str:
    return ticker.replace(".NS", "").replace(".BO", "")


def _retry_place(kite: KiteConnect, **params) -> str:
    last_err = None
    for attempt in range(_MAX_RETRY):
        try:
            return kite.place_order(variety=kite.VARIETY_REGULAR, **params)
        except Exception as exc:
            last_err = exc
            logger.warning("Order attempt %d/%d failed: %s", attempt + 1, _MAX_RETRY, exc)
            if attempt < _MAX_RETRY - 1:
                time.sleep(_BACKOFF ** attempt)
    raise last_err


# ── quote / margins ───────────────────────────────────────────────────────────

def get_quote(ticker: str) -> dict:
    kite = get_kite()
    key  = f"NSE:{_sym(ticker)}"
    q    = kite.quote([key])[key]
    depth = q.get("depth", {})
    buys  = depth.get("buy",  [])
    sells = depth.get("sell", [])
    return {
        "last_price":     q["last_price"],
        "best_buy":       buys[0]["price"]  if buys  else q["last_price"],
        "best_sell":      sells[0]["price"] if sells else q["last_price"],
        "volume":         q.get("volume", 0),
        "upper_circuit":  q.get("upper_circuit_limit"),
        "lower_circuit":  q.get("lower_circuit_limit"),
    }


def get_margins() -> dict:
    kite = get_kite()
    m    = kite.margins("equity")
    return {
        "available": round(float(m["net"]), 2),
        "used":      round(float(m.get("utilised", {}).get("debits", 0)), 2),
    }


# ── buy ───────────────────────────────────────────────────────────────────────

def place_buy(ticker: str, qty: int,
              order_type: str   = "MARKET",
              limit_price: float = 0,
              stop_loss: float   = 0,
              place_sl: bool     = True,
              dry_run: bool      = False) -> dict:
    """
    Place a CNC buy order on NSE.
    If stop_loss > 0 and place_sl is True, also places an SL-M sell order.
    order_type: "MARKET" | "LIMIT"
    """
    if not is_market_open():
        raise ValueError("Market closed. NSE hours: 9:15 AM – 3:30 PM, Mon–Fri.")

    symbol  = _sym(ticker)
    kite    = get_kite()
    quote   = get_quote(ticker)
    price   = limit_price or quote["last_price"]
    est_val = round(price * qty, 2)

    margins = get_margins()
    if margins["available"] < est_val:
        raise ValueError(
            f"Insufficient margin: need ₹{est_val:,.0f}, "
            f"available ₹{margins['available']:,.0f}. "
            f"Free up funds or reduce qty."
        )

    if dry_run:
        return {
            "dry_run":   True,
            "side":      "BUY",
            "ticker":    ticker,
            "symbol":    symbol,
            "qty":       qty,
            "order_type": order_type,
            "price":     round(price, 2),
            "est_value": est_val,
            "stop_loss": stop_loss,
        }

    buy_params = dict(
        tradingsymbol    = symbol,
        exchange         = _EXCHANGE,
        transaction_type = kite.TRANSACTION_TYPE_BUY,
        quantity         = qty,
        order_type       = (kite.ORDER_TYPE_MARKET
                            if order_type == "MARKET"
                            else kite.ORDER_TYPE_LIMIT),
        product          = kite.PRODUCT_CNC,
        validity         = kite.VALIDITY_DAY,
    )
    if order_type == "LIMIT" and limit_price:
        buy_params["price"] = round(limit_price, 2)

    order_id = _retry_place(kite, **buy_params)
    logger.info("BUY placed: %s qty=%d order_id=%s", symbol, qty, order_id)

    sl_order_id = None
    if stop_loss and place_sl:
        try:
            sl_params = dict(
                tradingsymbol    = symbol,
                exchange         = _EXCHANGE,
                transaction_type = kite.TRANSACTION_TYPE_SELL,
                quantity         = qty,
                order_type       = kite.ORDER_TYPE_SLM,
                product          = kite.PRODUCT_CNC,
                validity         = kite.VALIDITY_DAY,
                trigger_price    = round(stop_loss, 2),
            )
            sl_order_id = _retry_place(kite, **sl_params)
            logger.info("SL-M placed: %s trigger=%.2f order_id=%s",
                        symbol, stop_loss, sl_order_id)
        except Exception as exc:
            logger.error("SL-M order failed (buy already placed): %s", exc)

    return {
        "status":      "placed",
        "side":        "BUY",
        "ticker":      ticker,
        "symbol":      symbol,
        "qty":         qty,
        "order_type":  order_type,
        "est_value":   est_val,
        "order_id":    str(order_id),
        "sl_order_id": str(sl_order_id) if sl_order_id else None,
    }


# ── sell ──────────────────────────────────────────────────────────────────────

def place_sell(ticker: str, qty: int,
               order_type: str    = "MARKET",
               limit_price: float  = 0,
               dry_run: bool       = False) -> dict:
    """Place a CNC sell order on NSE."""
    if not is_market_open():
        raise ValueError("Market closed. NSE hours: 9:15 AM – 3:30 PM, Mon–Fri.")

    symbol  = _sym(ticker)
    kite    = get_kite()
    quote   = get_quote(ticker)
    price   = limit_price or quote["last_price"]
    est_val = round(price * qty, 2)

    if dry_run:
        return {
            "dry_run":    True,
            "side":       "SELL",
            "ticker":     ticker,
            "symbol":     symbol,
            "qty":        qty,
            "order_type": order_type,
            "price":      round(price, 2),
            "est_value":  est_val,
        }

    sell_params = dict(
        tradingsymbol    = symbol,
        exchange         = _EXCHANGE,
        transaction_type = kite.TRANSACTION_TYPE_SELL,
        quantity         = qty,
        order_type       = (kite.ORDER_TYPE_MARKET
                            if order_type == "MARKET"
                            else kite.ORDER_TYPE_LIMIT),
        product          = kite.PRODUCT_CNC,
        validity         = kite.VALIDITY_DAY,
    )
    if order_type == "LIMIT" and limit_price:
        sell_params["price"] = round(limit_price, 2)

    order_id = _retry_place(kite, **sell_params)
    logger.info("SELL placed: %s qty=%d order_id=%s", symbol, qty, order_id)

    return {
        "status":     "placed",
        "side":       "SELL",
        "ticker":     ticker,
        "symbol":     symbol,
        "qty":        qty,
        "order_type": order_type,
        "est_value":  est_val,
        "order_id":   str(order_id),
    }


# ── order status ──────────────────────────────────────────────────────────────

def get_order_status(order_id: str) -> dict:
    kite = get_kite()
    for o in kite.orders():
        if str(o["order_id"]) == str(order_id):
            return {
                "order_id":   str(o["order_id"]),
                "status":     o["status"],
                "filled_qty": o.get("filled_quantity", 0),
                "avg_price":  round(float(o.get("average_price") or 0), 2),
                "message":    o.get("status_message", ""),
            }
    return {"order_id": order_id, "status": "NOT_FOUND"}


def get_today_orders() -> list[dict]:
    kite = get_kite()
    return [
        {
            "order_id":   str(o["order_id"]),
            "ticker":     o.get("tradingsymbol", "") + ".NS",
            "side":       o["transaction_type"],
            "qty":        o["quantity"],
            "filled_qty": o.get("filled_quantity", 0),
            "avg_price":  round(float(o.get("average_price") or 0), 2),
            "status":     o["status"],
            "order_type": o["order_type"],
            "placed_at":  str(o.get("order_timestamp", "")),
            "message":    o.get("status_message", ""),
        }
        for o in kite.orders()
    ]
