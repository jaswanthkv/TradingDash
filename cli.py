"""
cli.py — Command-line interface (thin wrapper around engine.py)

Usage:
  python cli.py              # scan + Claude picks (prints results)
  python cli.py --positions  # review open positions only
"""
import os, sys, argparse, json, datetime
sys.path.insert(0, os.path.dirname(__file__))

import engine
from config import CAPITAL, MAX_POSITIONS, POSITIONS_FILE

# ── display helpers ───────────────────────────────────────────────────────────

def print_positions(positions, price_data=None):
    sep = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  OPEN POSITIONS — {len(positions)} trades")
    print(f"{'═'*72}")
    total_r = 0.0
    for pos in positions:
        ticker = pos["ticker"]
        df     = price_data.get(ticker) if price_data else None
        if df is None:
            print(f"  ⚠ {ticker:<18} — no price data")
            continue
        status = engine.compute_5ma_status(df, entry_price=pos["entry_price"])
        flag   = {"safe":"✓","warning":"⚠","exit":"✗"}[status["urgency"]]
        action = {"safe":"HOLD","warning":"ALERT","exit":"SELL next open"}[status["urgency"]]
        print(f"\n  {flag} {ticker:<18}  {status['five_ma_status']}")
        print(f"     Entry ₹{pos['entry_price']:<9.2f}  →  CMP ₹{status['current_price']:<9.2f}"
              f"  ({status.get('pnl_pct',0):+.1f}%   {status.get('r_multiple',0):+.2f}R)")
        print(f"     Trail stop: ₹{status.get('trailing_stop','—')}  (5EMA ₹{status['ema5']})")
        print(f"     → Action: {action}")
        total_r += status.get("r_multiple", 0)
    print(f"\n{sep}")
    print(f"  Portfolio total R: {total_r:+.2f}")
    print(f"{'═'*72}")


def print_scan_result(result: dict):
    picks    = result.get("picks", [])
    watchlist= result.get("watchlist", [])
    sep      = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  SWING ENTRY PICKS  |  {result['run_date']}  |  Capital ₹{CAPITAL:,.0f}")
    print(f"{'═'*72}")
    print(f"  Regime  : {result.get('regime_summary','—')}")
    print(f"  Stance  : {result.get('market_stance','—')}")
    print(f"  Themes  : {', '.join(result.get('sector_themes',[]) or ['—'])}")
    print(sep)
    for i, p in enumerate(picks, 1):
        print(f"\n  [{i}] {p['ticker']}  |  {p.get('signal_type','?')}  |  RS {p.get('rs_score','?')}  |  {p.get('conviction','?')}")
        print(f"      Thesis : {p.get('thesis','—')}")
        print(f"      Entry  : {p.get('entry_zone','—')}  Stop: ₹{p.get('initial_stop','?')}  ({p.get('stop_loss_pct','?')}%)")
        print(f"      Targets: T1 +{p.get('target1_pct','?')}%  T2 +{p.get('target2_pct','?')}%")
        print(f"      Size   : {p.get('position_pct','?')}%  →  {p.get('qty',0)} shares @ ₹{p.get('cmp','?')}  =  ₹{p.get('actual_inr',0):,.0f}")
        print(f"      Risk   : {p.get('key_risk','—')}")
    if watchlist:
        print(f"\n{sep}\n  WATCHLIST:")
        for w in watchlist:
            print(f"    • {w['ticker']:<18} — {w['reason']}")
    if result.get("exits"):
        print(f"\n{sep}\n  REBALANCE — EXITS:")
        for e in result["exits"]:
            pnl = e.get('pnl_pct','?')
            print(f"    ✗ SELL {e['ticker']:<18} reason: {e['exit_reason']}"
                  + (f"  P&L: {pnl:+.1f}%" if isinstance(pnl, float) else ""))
    print(f"\n{sep}")
    print(f"  Rationale: {result.get('portfolio_rationale','—')}")
    print(f"{'═'*72}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TradeBoard CLI")
    parser.add_argument("--positions", action="store_true", help="Review open positions only")
    args = parser.parse_args()

    if args.positions:
        positions = engine.load_positions()
        if not positions:
            print("No open positions.")
            return
        tickers    = [p["ticker"] for p in positions]
        price_data = engine.fetch_prices(tickers, days=60, min_bars=20)
        print_positions(positions, price_data)
        return

    # full scan
    print("\n  Running universe scan…")
    result = engine.run_scan()
    print_scan_result(result)

    confirm = input("Commit rebalance to open_positions.json? [y/N] ").strip().lower()
    if confirm != "y":
        return

    exit_tickers   = [e["ticker"] for e in result.get("exits", [])]
    approved_picks = result.get("picks", [])
    updated        = engine.apply_rebalance(exit_tickers, approved_picks)
    print(f"  Positions updated ({len(updated)}/{MAX_POSITIONS}) → {POSITIONS_FILE}")


if __name__ == "__main__":
    main()
