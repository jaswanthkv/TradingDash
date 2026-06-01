"""
kite_orders.py — Rebalance order execution via Kite Connect.

Flow:
  1. preview(to_buy, to_sell, capital_per_stock) — fetch LTP + holdings, return order list
  2. execute(to_buy, to_sell, capital_per_stock) — place limit CNC orders (sells first)

Limit order pricing:
  BUY  → LTP × 1.003  (+0.3%) so the order fills at or below this price
  SELL → LTP × 0.997  (−0.3%) so the order fills at or above this price

Rate limiting: 0.4 s pause between each order (Kite allows ~3/s).
"""
import math
import time
import kite_auth

_ORDER_DELAY = 0.4   # seconds between consecutive orders


def _limit_price(ltp: float, action: str) -> float:
    """Return a limit price with a small buffer to ensure quick fill."""
    buf = 1.003 if action == "BUY" else 0.997
    return round(ltp * buf, 2)


def _kite():
    return kite_auth.get_kite()


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
    kite     = _kite()
    buy_syms  = [t.replace(".NS", "") for t in to_buy]
    sell_syms = [t.replace(".NS", "") for t in to_sell]
    holdings  = get_holdings()
    prices    = _ltp(kite, buy_syms)

    orders = []

    for sym in sell_syms:
        held  = holdings.get(sym, {})
        qty   = held.get("quantity", 0)
        ltp   = held.get("last_price", 0)
        price = _limit_price(ltp, "SELL")
        orders.append({
            "symbol": sym, "action": "SELL",
            "quantity": qty, "price": price,
            "estimated_amount": round(qty * price, 2),
            "note": "Not in holdings" if qty == 0 else "",
        })

    for sym in buy_syms:
        ltp   = prices.get(sym, 0)
        qty   = math.floor(capital_per_stock / ltp) if ltp > 0 else 0
        price = _limit_price(ltp, "BUY")
        orders.append({
            "symbol": sym, "action": "BUY",
            "quantity": qty, "price": price,
            "estimated_amount": round(qty * price, 2),
            "note": "Price unavailable" if ltp == 0 else "",
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

    # Fetch fresh LTP for both sides in one call
    all_prices = _ltp(kite, buy_syms + sell_syms)

    results = []

    for sym in sell_syms:
        qty = holdings.get(sym, {}).get("quantity", 0)
        if qty <= 0:
            results.append({"symbol": sym, "action": "SELL", "quantity": 0,
                            "status": "skipped", "note": "Not in holdings"})
            continue
        ltp   = all_prices.get(sym) or holdings.get(sym, {}).get("last_price", 0)
        price = _limit_price(ltp, "SELL")
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
        ltp   = all_prices.get(sym, 0)
        qty   = math.floor(capital_per_stock / ltp) if ltp > 0 else 0
        price = _limit_price(ltp, "BUY")
        if qty <= 0:
            results.append({"symbol": sym, "action": "BUY", "quantity": 0,
                            "status": "skipped", "note": "Price unavailable or qty 0"})
            continue
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
                            "price": price, "status": "placed", "order_id": oid})
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
