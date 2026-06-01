"""
kite_orders.py — Rebalance order execution via Kite Connect.

Flow:
  1. preview(to_buy, to_sell, capital_per_stock) — fetch LTP + holdings, return order list
  2. execute(to_buy, to_sell, capital_per_stock) — place limit CNC orders (sells first)

Limit order pricing:
  BUY  → ceil(LTP × 1.003 / tick) × tick  — rounds UP to next valid tick
  SELL → floor(LTP × 0.997 / tick) × tick — rounds DOWN to next valid tick

Tick sizes (NSE): 0.05 for most equities; 0.10 / 0.50 / 1.00 for some.
We fetch the instruments list from Kite on first call and cache for 1 hour.

Rate limiting: 0.4 s pause between each order (Kite allows ~3/s).
"""
import math
import time
import kite_auth

_ORDER_DELAY = 0.4   # seconds between consecutive orders

_tick_cache: dict  = {}
_tick_ts:    float = 0.0
_TICK_TTL          = 3600   # refresh tick map every hour


def _kite():
    return kite_auth.get_kite()


def _load_tick_map(kite) -> dict:
    """Return {tradingsymbol: tick_size} for NSE EQ instruments (cached 1 h)."""
    global _tick_cache, _tick_ts
    if time.time() - _tick_ts > _TICK_TTL:
        instruments = kite.instruments("NSE")
        _tick_cache = {
            i["tradingsymbol"]: float(i.get("tick_size") or 0.05)
            for i in instruments
            if i.get("instrument_type") == "EQ"
        }
        _tick_ts = time.time()
    return _tick_cache


def _limit_price(ltp: float, action: str, tick: float = 0.05) -> float:
    """
    Limit price with 0.3% buffer snapped to the instrument's tick size.
    BUY  → ceil  so we never undershoot the ask.
    SELL → floor so we never overshoot the bid.
    """
    if tick <= 0:
        tick = 0.05
    raw     = ltp * (1.003 if action == "BUY" else 0.997)
    snapped = (math.ceil if action == "BUY" else math.floor)(raw / tick) * tick
    return round(snapped, 4)   # keep extra precision; Kite truncates internally


def get_holdings() -> dict:
    """Returns {symbol: holding} for all NSE holdings with qty > 0."""
    kite = _kite()
    return {h["tradingsymbol"]: h for h in kite.holdings()
            if h.get("quantity", 0) > 0}


def _ltp(kite, symbols: list) -> dict:
    """Returns {symbol: last_price} for NSE symbols."""
    if not symbols:
        return {}
    quotes = kite.quote([f"NSE:{s}" for s in symbols])
    return {s: quotes.get(f"NSE:{s}", {}).get("last_price", 0) for s in symbols}


def preview(to_buy: list, to_sell: list, capital_per_stock: float) -> dict:
    kite      = _kite()
    buy_syms  = [t.replace(".NS", "") for t in to_buy]
    sell_syms = [t.replace(".NS", "") for t in to_sell]
    holdings  = get_holdings()
    prices    = _ltp(kite, buy_syms)
    ticks     = _load_tick_map(kite)

    orders = []

    for sym in sell_syms:
        held  = holdings.get(sym, {})
        qty   = held.get("quantity", 0)
        ltp   = held.get("last_price", 0)
        tick  = ticks.get(sym, 0.05)
        price = _limit_price(ltp, "SELL", tick)
        orders.append({
            "symbol": sym, "action": "SELL",
            "quantity": qty, "price": price,
            "estimated_amount": round(qty * price, 2),
            "note": "Not in holdings" if qty == 0 else "",
        })

    for sym in buy_syms:
        ltp        = prices.get(sym, 0)
        tick       = ticks.get(sym, 0.05)
        target_qty = math.floor(capital_per_stock / ltp) if ltp > 0 else 0
        held_qty   = holdings.get(sym, {}).get("quantity", 0)
        qty        = max(0, target_qty - held_qty)
        price      = _limit_price(ltp, "BUY", tick)
        note       = ""
        if ltp == 0:
            note = "Price unavailable"
        elif held_qty >= target_qty:
            note = f"Already holding {held_qty}"
        elif held_qty > 0:
            note = f"Held {held_qty}, buying {qty}"
        orders.append({
            "symbol": sym, "action": "BUY",
            "quantity": qty, "price": price,
            "estimated_amount": round(qty * price, 2),
            "note": note,
        })

    total_buy  = sum(o["estimated_amount"] for o in orders if o["action"] == "BUY")
    total_sell = sum(o["estimated_amount"] for o in orders if o["action"] == "SELL")
    return {
        "orders": orders,
        "summary": {
            "total_buy":   round(total_buy, 2),
            "total_sell":  round(total_sell, 2),
            "net_outflow": round(total_buy - total_sell, 2),
        },
    }


def execute(to_buy: list, to_sell: list, capital_per_stock: float) -> dict:
    """Place limit CNC orders — sells first, then buys, with rate-limit delay."""
    kite      = _kite()
    buy_syms  = [t.replace(".NS", "") for t in to_buy]
    sell_syms = [t.replace(".NS", "") for t in to_sell]
    holdings  = get_holdings()
    ticks     = _load_tick_map(kite)

    # Fresh LTP for both sides in one call
    all_prices = _ltp(kite, buy_syms + sell_syms)

    results = []

    for sym in sell_syms:
        qty = holdings.get(sym, {}).get("quantity", 0)
        if qty <= 0:
            results.append({"symbol": sym, "action": "SELL", "quantity": 0,
                            "status": "skipped", "note": "Not in holdings"})
            continue
        ltp   = all_prices.get(sym) or holdings.get(sym, {}).get("last_price", 0)
        tick  = ticks.get(sym, 0.05)
        price = _limit_price(ltp, "SELL", tick)
        try:
            oid = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=sym,
                transaction_type=kite.TRANSACTION_TYPE_SELL,
                quantity=qty,
                order_type=kite.ORDER_TYPE_LIMIT,
                price=price,
                product=kite.PRODUCT_CNC,
            )
            results.append({"symbol": sym, "action": "SELL", "quantity": qty,
                            "price": price, "status": "placed", "order_id": oid})
        except Exception as e:
            results.append({"symbol": sym, "action": "SELL", "quantity": qty,
                            "price": price, "status": "failed", "error": str(e)})
        time.sleep(_ORDER_DELAY)

    for sym in buy_syms:
        ltp        = all_prices.get(sym, 0)
        tick       = ticks.get(sym, 0.05)
        target_qty = math.floor(capital_per_stock / ltp) if ltp > 0 else 0
        held_qty   = holdings.get(sym, {}).get("quantity", 0)
        qty        = max(0, target_qty - held_qty)

        if qty <= 0:
            if ltp == 0:
                note = "Symbol not found on NSE"
            elif held_qty >= target_qty:
                note = f"Already holding {held_qty} share{'s' if held_qty != 1 else ''}"
            else:
                note = "Qty rounds to 0 — increase capital"
            results.append({"symbol": sym, "action": "BUY", "quantity": 0,
                            "status": "skipped", "note": note})
            continue

        price = _limit_price(ltp, "BUY", tick)
        try:
            oid = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=sym,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=qty,
                order_type=kite.ORDER_TYPE_LIMIT,
                price=price,
                product=kite.PRODUCT_CNC,
            )
            results.append({"symbol": sym, "action": "BUY", "quantity": qty,
                            "price": price, "status": "placed", "order_id": oid,
                            "note": f"Held {held_qty}, buying {qty}" if held_qty else ""})
        except Exception as e:
            results.append({"symbol": sym, "action": "BUY", "quantity": qty,
                            "price": price, "status": "failed", "error": str(e)})
        time.sleep(_ORDER_DELAY)

    placed  = sum(1 for r in results if r["status"] == "placed")
    failed  = sum(1 for r in results if r["status"] == "failed")
    skipped = len(results) - placed - failed
    return {
        "results": results,
        "summary": {"placed": placed, "failed": failed, "skipped": skipped},
    }
