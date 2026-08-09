"""
trade_live.py — Equity (cash market) order execution via Kite for the SEPA screener's
"Buy Top 20" flow.

Buy: equal-rupee allocation across the given symbols, sized against a live LTP snapshot,
executed as CNC market buys.
Sell: closes (fully or partially) an existing equity holding at market.
"""
from __future__ import annotations


def _ltp(kite, symbol: str) -> float:
    quote = kite.quote(f"NSE:{symbol}")
    return quote.get(f"NSE:{symbol}", {}).get("last_price", 0.0)


def preview_buy(kite, symbols: list[str], budget: float) -> list[dict]:
    """Live LTP + quantity for an equal split of budget across symbols.
    A symbol priced above its allocated share gets qty=0 (skipped)."""
    if not symbols:
        return []
    allocation = budget / len(symbols)
    rows = []
    for sym in symbols:
        ltp = _ltp(kite, sym)
        qty = int(allocation // ltp) if ltp > 0 else 0
        rows.append({
            "symbol":  sym,
            "ltp":     round(ltp, 2),
            "qty":     qty,
            "cost":    round(qty * ltp, 2),
            "skipped": qty == 0,
        })
    return rows


def place_buy_orders(kite, orders: list[dict]) -> list[dict]:
    """orders: [{symbol, qty}]. CNC market buy per symbol; failures don't block the rest."""
    results = []
    for o in orders:
        sym, qty = o["symbol"], int(o["qty"])
        if qty <= 0:
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
            results.append({"symbol": sym, "qty": qty, "order_id": oid})
        except Exception as exc:
            results.append({"symbol": sym, "qty": qty, "error": str(exc)})
    return results


def get_holdings(kite) -> list[dict]:
    """Current NSE equity holdings with non-zero quantity."""
    result = []
    for h in kite.holdings():
        qty = h.get("quantity", 0)
        if qty == 0:
            continue
        result.append({
            "symbol":        h["tradingsymbol"],
            "quantity":      qty,
            "average_price": round(h.get("average_price", 0), 2),
            "last_price":    round(h.get("last_price", 0), 2),
            "pnl":           round(h.get("pnl", 0), 2),
        })
    return result


def sell_holding(kite, symbol: str, quantity: int) -> str:
    """Sell (part or all of) an equity holding at market. Returns order_id."""
    return kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NSE,
        tradingsymbol=symbol,
        transaction_type=kite.TRANSACTION_TYPE_SELL,
        quantity=int(quantity),
        order_type=kite.ORDER_TYPE_MARKET,
        product=kite.PRODUCT_CNC,
    )
