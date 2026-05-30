"""
kite_orders.py — Rebalance order execution via Kite Connect.

Flow:
  1. preview(to_buy, to_sell, capital_per_stock) — fetch LTP + holdings, return order list
  2. execute(to_buy, to_sell, capital_per_stock) — place market CNC orders (sells first)
"""
import math
import kite_auth


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
        held = holdings.get(sym, {})
        qty  = held.get("quantity", 0)
        price = held.get("last_price", 0)
        orders.append({
            "symbol": sym, "action": "SELL",
            "quantity": qty, "price": round(price, 2),
            "estimated_amount": round(qty * price, 2),
            "note": "Not in holdings" if qty == 0 else "",
        })

    for sym in buy_syms:
        price = prices.get(sym, 0)
        qty   = math.floor(capital_per_stock / price) if price > 0 else 0
        orders.append({
            "symbol": sym, "action": "BUY",
            "quantity": qty, "price": round(price, 2),
            "estimated_amount": round(qty * price, 2),
            "note": "Price unavailable" if price == 0 else "",
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
    """Place market CNC orders — sells first, then buys."""
    kite     = _kite()
    buy_syms  = [t.replace(".NS", "") for t in to_buy]
    sell_syms = [t.replace(".NS", "") for t in to_sell]
    holdings  = get_holdings()
    prices    = _ltp(kite, buy_syms)

    results = []

    for sym in sell_syms:
        qty = holdings.get(sym, {}).get("quantity", 0)
        if qty <= 0:
            results.append({"symbol": sym, "action": "SELL", "quantity": 0,
                            "status": "skipped", "note": "Not in holdings"})
            continue
        try:
            oid = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=sym,
                transaction_type=kite.TRANSACTION_TYPE_SELL,
                quantity=qty,
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_CNC,
            )
            results.append({"symbol": sym, "action": "SELL", "quantity": qty,
                            "status": "placed", "order_id": oid})
        except Exception as e:
            results.append({"symbol": sym, "action": "SELL", "quantity": qty,
                            "status": "failed", "error": str(e)})

    for sym in buy_syms:
        price = prices.get(sym, 0)
        qty   = math.floor(capital_per_stock / price) if price > 0 else 0
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
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_CNC,
            )
            results.append({"symbol": sym, "action": "BUY", "quantity": qty,
                            "status": "placed", "order_id": oid})
        except Exception as e:
            results.append({"symbol": sym, "action": "BUY", "quantity": qty,
                            "status": "failed", "error": str(e)})

    placed  = sum(1 for r in results if r["status"] == "placed")
    failed  = sum(1 for r in results if r["status"] == "failed")
    skipped = len(results) - placed - failed
    return {
        "results": results,
        "summary": {"placed": placed, "failed": failed, "skipped": skipped},
    }
