#!/usr/bin/env python3
"""
Builds a self-contained HTML dashboard from the current portfolio state.
Run after every weekly cycle: python3 scripts/build_dashboard.py

No network calls - just renders state/portfolio.json + state/trade_log.csv +
universe.csv into a static report (Chart.js loaded from CDN for the NAV chart).
"""
import json
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PORTFOLIO_PATH = BASE / "state" / "portfolio.json"
TRADE_LOG_PATH = BASE / "state" / "trade_log.csv"
UNIVERSE_PATH = BASE / "universe.csv"
OUT_PATH = BASE / "dashboard.html"


def load_company_names():
    names = {}
    with open(UNIVERSE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            names[row["symbol"].strip().upper()] = row["company_name"].strip().title()
    return names


def load_trades():
    if not TRADE_LOG_PATH.exists():
        return []
    with open(TRADE_LOG_PATH, newline="") as f:
        return list(csv.DictReader(f))


def main():
    with open(PORTFOLIO_PATH) as f:
        state = json.load(f)
    names = load_company_names()
    trades = load_trades()

    nav_history = state["nav_history"]
    latest_nav = nav_history[-1]["nav"] if nav_history else state["starting_capital_inr"]
    total_return_pct = (latest_nav / state["starting_capital_inr"] - 1) * 100

    bench_inception = state.get("benchmark_inception_value")
    bench_latest = nav_history[-1].get("benchmark") if nav_history else None
    bench_return_pct = None
    if bench_inception and bench_latest:
        bench_return_pct = (bench_latest / bench_inception - 1) * 100

    holdings_rows = ""
    for sym, pos in sorted(state["holdings"].items(), key=lambda kv: -kv[1]["qty"] * kv[1]["avg_cost"]):
        value = pos["qty"] * pos["avg_cost"]
        weight = value / latest_nav * 100 if latest_nav else 0
        holdings_rows += f"""<tr>
            <td>{sym}</td>
            <td class="dim">{names.get(sym, '')}</td>
            <td class="num">{pos['qty']}</td>
            <td class="num">Rs.{pos['avg_cost']:,.2f}</td>
            <td class="num">Rs.{value:,.0f}</td>
            <td class="num">{weight:.1f}%</td>
        </tr>"""

    trade_rows = ""
    for t in reversed(trades[-40:]):
        action_cls = "buy" if t["action"] == "buy" else "sell"
        trade_rows += f"""<tr>
            <td class="dim">{t['date']}</td>
            <td>{t['symbol']}</td>
            <td class="{action_cls}">{t['action'].upper()}</td>
            <td class="num">{t['qty']}</td>
            <td class="num">Rs.{float(t['price']):,.2f}</td>
            <td class="num">Rs.{float(t['value_inr']):,.0f}</td>
            <td class="dim reasoning">{t['reasoning_summary']}</td>
        </tr>"""

    nav_dates = json.dumps([h["date"] for h in nav_history])
    nav_values = json.dumps([h["nav"] for h in nav_history])
    bench_values = json.dumps([
        (h.get("benchmark") / bench_inception * state["starting_capital_inr"]) if h.get("benchmark") and bench_inception else None
        for h in nav_history
    ])

    return_color = "#16a34a" if total_return_pct >= 0 else "#dc2626"
    bench_color = "#16a34a" if (bench_return_pct or 0) >= 0 else "#dc2626"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Nifty Microcap 250 - LLM Portfolio</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f7f7f8; margin: 0; padding: 32px; color: #1a1a1a; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #6b7280; font-size: 14px; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 28px; }}
  .card {{ background: white; border-radius: 12px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }}
  .card .value {{ font-size: 26px; font-weight: 600; margin-top: 4px; }}
  .chart-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 28px; }}
  .section-title {{ font-size: 16px; font-weight: 600; margin: 0 0 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 10px; color: #6b7280; font-weight: 600; border-bottom: 1px solid #e5e7eb; font-size: 11px; text-transform: uppercase; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #f0f0f1; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.dim {{ color: #6b7280; }}
  td.reasoning {{ max-width: 320px; font-size: 12px; }}
  .buy {{ color: #16a34a; font-weight: 600; }}
  .sell {{ color: #dc2626; font-weight: 600; }}
  .table-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 28px; overflow-x: auto; }}
  .footer-note {{ font-size: 12px; color: #9ca3af; margin-top: 8px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Nifty Microcap 250 &mdash; LLM Portfolio (Claude)</h1>
  <div class="subtitle">Paper trading experiment &middot; inception {state['inception_date']} &middot; last rebalance {state.get('last_rebalance_date') or '-'}</div>

  <div class="cards">
    <div class="card">
      <div class="label">NAV</div>
      <div class="value">Rs.{latest_nav:,.0f}</div>
    </div>
    <div class="card">
      <div class="label">Total Return</div>
      <div class="value" style="color:{return_color}">{total_return_pct:+.2f}%</div>
    </div>
    <div class="card">
      <div class="label">Nifty Microcap 250</div>
      <div class="value" style="color:{bench_color}">{f'{bench_return_pct:+.2f}%' if bench_return_pct is not None else 'n/a'}</div>
    </div>
    <div class="card">
      <div class="label">Holdings / Cash</div>
      <div class="value">{len(state['holdings'])} &middot; Rs.{state['cash_inr']:,.0f}</div>
    </div>
  </div>

  <div class="chart-card">
    <div class="section-title">NAV vs Nifty Microcap 250 (rebased to starting capital)</div>
    <canvas id="navChart" height="90"></canvas>
  </div>

  <div class="table-card">
    <div class="section-title">Current Holdings</div>
    <table>
      <thead><tr><th>Symbol</th><th>Company</th><th class="num">Qty</th><th class="num">Avg Cost</th><th class="num">Value</th><th class="num">Weight</th></tr></thead>
      <tbody>{holdings_rows}</tbody>
    </table>
  </div>

  <div class="table-card">
    <div class="section-title">Recent Trades</div>
    <table>
      <thead><tr><th>Date</th><th>Symbol</th><th>Action</th><th class="num">Qty</th><th class="num">Price</th><th class="num">Value</th><th>Reasoning</th></tr></thead>
      <tbody>{trade_rows}</tbody>
    </table>
  </div>

  <div class="footer-note">
    Paper trading only, no real orders. Universe: Nifty Microcap 250 proxy (SEBI/AMFI avg market-cap ranks 501-750, Jul-Dec 2024).
    Prices from Yahoo/Google Finance quote pages, not tick-accurate. Regenerated by scripts/build_dashboard.py.
  </div>
</div>

<script>
const ctx = document.getElementById('navChart');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {nav_dates},
    datasets: [
      {{ label: 'Claude Portfolio', data: {nav_values}, borderColor: '#4f46e5', backgroundColor: 'rgba(79,70,229,0.08)', fill: true, tension: 0.2 }},
      {{ label: 'Nifty Microcap 250 (rebased)', data: {bench_values}, borderColor: '#9ca3af', borderDash: [5,4], fill: false, tension: 0.2 }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ position: 'bottom' }} }},
    scales: {{ y: {{ ticks: {{ callback: (v) => 'Rs.' + v.toLocaleString() }} }} }}
  }}
}});
</script>
</body>
</html>
"""
    OUT_PATH.write_text(html)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
