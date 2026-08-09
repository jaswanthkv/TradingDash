#!/usr/bin/env python3
"""
Paper trading engine for the Nifty Microcap 250 LLM portfolio experiment.

This script does NO networking. It takes prices supplied via a JSON file
(fetched separately, e.g. by Claude via Yahoo Finance quote pages) and a
list of proposed decisions, validates them against the hard rules in
docs/decision_rules.md, executes what's valid, and updates state.

Usage:
    python3 paper_trader.py mark --prices prices.json
        -> mark-to-market current holdings, append NAV history point, no trades.

    python3 paper_trader.py apply --decisions decisions.json --prices prices.json
        -> validate + execute a batch of decisions, log trades, update NAV.

decisions.json format:
[
  {"symbol": "TDPOWERSYS", "action": "buy", "qty": 120, "reasoning": "..."},
  {"symbol": "MTARTECH",   "action": "sell", "qty": 40,  "reasoning": "..."}
]

prices.json format:
{"TDPOWERSYS": 650.5, "MTARTECH": 7000.0, ...}   # last close, INR, one entry per symbol touched
Also expects a special key "__BENCHMARK__" with the current NIFTY_MICROCAP250.NS level, if available.
"""
import json
import csv
import sys
import argparse
import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE / "state" / "portfolio.json"
TRADE_LOG_PATH = BASE / "state" / "trade_log.csv"
UNIVERSE_PATH = BASE / "universe.csv"

# ---- Hard rules (see docs/decision_rules.md) ----
MAX_POSITION_PCT = 0.10
MIN_HOLDINGS_WHEN_FULLY_INVESTED = 15
MIN_CASH_BUFFER_PCT = 0.02
MAX_WEEKLY_TURNOVER_PCT = 0.30


def load_universe():
    symbols = set()
    with open(UNIVERSE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            symbols.add(row["symbol"].strip().upper())
    return symbols


def load_portfolio():
    with open(PORTFOLIO_PATH) as f:
        return json.load(f)


def save_portfolio(state):
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(state, f, indent=2)


def compute_nav(state, prices):
    holdings_value = 0.0
    missing = []
    for sym, pos in state["holdings"].items():
        if sym in prices:
            holdings_value += pos["qty"] * prices[sym]
        else:
            missing.append(sym)
    nav = state["cash_inr"] + holdings_value
    return nav, holdings_value, missing


def append_nav_history(state, nav, benchmark_value=None):
    today = datetime.date.today().isoformat()
    entry = {"date": today, "nav": round(nav, 2)}
    if benchmark_value is not None:
        entry["benchmark"] = benchmark_value
        if state.get("benchmark_inception_value") is None:
            state["benchmark_inception_value"] = benchmark_value
    # replace if same-day entry already exists (idempotent re-runs)
    state["nav_history"] = [h for h in state["nav_history"] if h["date"] != today]
    state["nav_history"].append(entry)


def log_trade(date, symbol, action, qty, price, cash_after, reasoning):
    is_new = not TRADE_LOG_PATH.exists()
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["date", "symbol", "action", "qty", "price", "value_inr", "cash_after", "reasoning_summary"])
        value = round(qty * price, 2)
        w.writerow([date, symbol, action, qty, price, value, round(cash_after, 2), reasoning[:300]])


def validate_and_execute(state, decisions, prices, universe, inception=False):
    today = datetime.date.today().isoformat()
    accepted, rejected = [], []

    nav_before, _, _ = compute_nav(state, prices)
    turnover_used = 0.0
    turnover_cap_pct = 1.01 if inception else MAX_WEEKLY_TURNOVER_PCT  # inception: building from 100% cash, cap doesn't apply

    for d in decisions:
        sym = d["symbol"].strip().upper()
        action = d["action"].strip().lower()
        qty = int(d["qty"])
        reasoning = d.get("reasoning", "")

        if sym not in universe:
            rejected.append((d, f"{sym} not in Nifty Microcap 250 universe"))
            continue
        if sym not in prices:
            rejected.append((d, f"no price supplied for {sym}"))
            continue
        price = prices[sym]
        trade_value = qty * price

        if action == "buy":
            if trade_value > state["cash_inr"]:
                rejected.append((d, f"insufficient cash: need {trade_value:.0f}, have {state['cash_inr']:.0f}"))
                continue
            existing_qty = state["holdings"].get(sym, {}).get("qty", 0)
            existing_cost = state["holdings"].get(sym, {}).get("avg_cost", 0)
            new_position_value = (existing_qty * existing_cost) + trade_value
            if new_position_value > MAX_POSITION_PCT * nav_before:
                rejected.append((d, f"position limit breach: {sym} would be "
                                     f"{new_position_value/nav_before:.1%} of NAV (max {MAX_POSITION_PCT:.0%})"))
                continue
            if (turnover_used + trade_value) > turnover_cap_pct * nav_before:
                rejected.append((d, "weekly turnover cap reached"))
                continue

            new_qty = existing_qty + qty
            new_avg_cost = new_position_value / new_qty
            state["holdings"][sym] = {"qty": new_qty, "avg_cost": round(new_avg_cost, 2)}
            state["cash_inr"] -= trade_value
            turnover_used += trade_value
            log_trade(today, sym, "buy", qty, price, state["cash_inr"], reasoning)
            accepted.append(d)

        elif action == "sell":
            existing_qty = state["holdings"].get(sym, {}).get("qty", 0)
            if qty > existing_qty:
                rejected.append((d, f"cannot sell {qty}, only hold {existing_qty}"))
                continue
            state["cash_inr"] += trade_value
            remaining = existing_qty - qty
            if remaining == 0:
                del state["holdings"][sym]
            else:
                state["holdings"][sym]["qty"] = remaining
            turnover_used += trade_value
            log_trade(today, sym, "sell", qty, price, state["cash_inr"], reasoning)
            accepted.append(d)
        else:
            rejected.append((d, f"unknown action {action}"))

    # post-trade checks (informational, not blocking retroactively)
    cash_pct = state["cash_inr"] / nav_before if nav_before else 0
    warnings = []
    if cash_pct < MIN_CASH_BUFFER_PCT:
        warnings.append(f"cash buffer below target: {cash_pct:.1%} < {MIN_CASH_BUFFER_PCT:.0%}")
    if len(state["holdings"]) < MIN_HOLDINGS_WHEN_FULLY_INVESTED and cash_pct < 0.5:
        warnings.append(f"only {len(state['holdings'])} holdings, below diversification floor of {MIN_HOLDINGS_WHEN_FULLY_INVESTED}")

    state["last_rebalance_date"] = today
    return accepted, rejected, warnings


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_mark = sub.add_parser("mark")
    p_mark.add_argument("--prices", required=True)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--decisions", required=True)
    p_apply.add_argument("--prices", required=True)
    p_apply.add_argument("--inception", action="store_true",
                          help="One-time flag for initial portfolio construction: bypasses the weekly turnover cap.")

    args = ap.parse_args()
    state = load_portfolio()
    universe = load_universe()

    with open(args.prices) as f:
        prices = json.load(f)
    benchmark_value = prices.pop("__BENCHMARK__", None)

    if args.cmd == "apply":
        with open(args.decisions) as f:
            decisions = json.load(f)
        accepted, rejected, warnings = validate_and_execute(state, decisions, prices, universe, inception=args.inception)
        nav, holdings_value, missing = compute_nav(state, prices)
        append_nav_history(state, nav, benchmark_value)
        save_portfolio(state)

        print(f"Accepted {len(accepted)} trade(s), rejected {len(rejected)}")
        for d, reason in rejected:
            print(f"  REJECTED {d['action']} {d['qty']} {d['symbol']}: {reason}")
        for w in warnings:
            print(f"  WARNING: {w}")
        if missing:
            print(f"  NOTE: no price for {len(missing)} held symbol(s), excluded from NAV: {missing}")
        print(f"NAV: Rs.{nav:,.2f}  (cash Rs.{state['cash_inr']:,.2f} + holdings Rs.{holdings_value:,.2f})")

    elif args.cmd == "mark":
        nav, holdings_value, missing = compute_nav(state, prices)
        append_nav_history(state, nav, benchmark_value)
        save_portfolio(state)
        if missing:
            print(f"NOTE: no price for {len(missing)} held symbol(s), excluded from NAV: {missing}")
        print(f"NAV: Rs.{nav:,.2f}  (cash Rs.{state['cash_inr']:,.2f} + holdings Rs.{holdings_value:,.2f})")


if __name__ == "__main__":
    main()
