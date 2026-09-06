"""
live_trading.py — bridges the SEPA Top 20 selection to real orders via Dhan.

Unlike portfolio_tracker.py's paper journal (which keeps its own local
ledger), this always reads real holdings and cash from Dhan first — there is
no local ledger here that can drift out of sync with what you actually own.

Two-step, review-and-confirm by design: plan_rebalance() computes proposed
orders and sends nothing to Dhan except read-only calls (holdings, funds,
LTP). execute_rebalance() recomputes that same plan fresh — it never trusts
a client-supplied order list — and only then places orders.
"""
import minervini as mv
import dhan_client as dhan

TOP_N = 20


def _compute_top20(get_prices=None) -> list[str]:
    """SEPA full-pass NSE stocks (≥500 Cr), ranked by RS Rating, top 20."""
    rows = mv.screen_market("india", use_cache=True, get_prices=get_prices)["rows"]
    candidates = [r for r in rows if r.get("sepa_pass") and r.get("rs_rating") is not None]
    candidates.sort(key=lambda r: r["rs_rating"], reverse=True)
    return [r["symbol"] for r in candidates[:TOP_N]]


def plan_rebalance(get_prices=None) -> dict:
    """Compute the buy/sell orders needed to move real Dhan holdings to the
    current Top 20, equal-weight. Read-only — sends nothing to Dhan besides
    holdings/funds/LTP lookups."""
    top20    = _compute_top20(get_prices=get_prices)
    holdings = dhan.get_holdings()
    cash     = dhan.get_available_cash()
    live_px  = dhan.get_ltp(sorted(set(top20) | set(holdings)))

    total_value  = cash + sum(pos["qty"] * live_px.get(sym, 0) for sym, pos in holdings.items())
    priced_top20 = [s for s in top20 if s in live_px]
    per_stock    = total_value / len(priced_top20) if priced_top20 else 0

    target_qty = {}
    for sym in priced_top20:
        qty = int(per_stock // live_px[sym])
        if qty > 0:
            target_qty[sym] = qty

    orders = []
    for sym in sorted(set(target_qty) | set(holdings)):
        cur   = holdings.get(sym, {}).get("qty", 0)
        tgt   = target_qty.get(sym, 0)
        delta = tgt - cur
        if delta > 0:
            orders.append({"symbol": sym, "side": "BUY", "qty": delta, "price": live_px[sym]})
        elif delta < 0:
            # Selling out of an existing holding that dropped out of the Top 20
            # — use the live price if we have it, else fall back to avg_cost
            # so the plan is still shown (a missing LTP shouldn't hide a sell).
            px = live_px.get(sym) or holdings[sym]["avg_cost"]
            orders.append({"symbol": sym, "side": "SELL", "qty": -delta, "price": px})

    return {
        "top20":       top20,
        "orders":      orders,
        "total_value": round(total_value, 2),
        "cash":        round(cash, 2),
        "holdings":    holdings,
    }


def execute_rebalance(get_prices=None) -> dict:
    """Recompute plan_rebalance() fresh and place each order via Dhan,
    tracking per-order success/failure so one rejection doesn't block the
    rest and a failed order can be retried individually."""
    plan = plan_rebalance(get_prices=get_prices)
    results = dhan.place_orders(plan["orders"])
    return {**plan, "results": results}
