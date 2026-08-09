# Decision Rules — Nifty Microcap 250 LLM Portfolio

This is a **paper-trading** experiment: Claude acts as portfolio manager over a virtual
₹10,00,000 account, picking only from the Nifty Microcap 250 universe (`universe.csv`,
sourced from SEBI/AMFI average-market-cap ranks 501–750, Jul–Dec 2024 classification —
refresh periodically for accuracy). No real orders are ever placed from this workflow.

## Hard constraints (validated programmatically before any trade is logged)

1. **Universe only** — every symbol traded must appear in `universe.csv`.
2. **No leverage, no shorting** — cash_inr must never go negative; only long positions.
3. **Position limit** — no single holding may exceed 10% of NAV at the time of purchase.
4. **Diversification floor** — once fully invested, hold at least 15 distinct names.
5. **Cash buffer** — keep at least 2% of NAV in cash after each rebalance (execution slack).
6. **Turnover cap** — no more than 30% of NAV traded (buys + sells) in a single weekly cycle,
   to keep this a realistic, not hyperactive, strategy and to respect microcap liquidity.
7. **Liquidity check** — skip any name where the intended trade value exceeds a small
   fraction of visible daily traded value (avoid moving illiquid microcap prices unrealistically).

## Weekly process

1. Mark-to-market current holdings using latest close prices (Yahoo Finance `SYMBOL.NS`).
2. Screen the universe (fundamentals + price action) for candidates.
3. Produce buy/sell/hold decisions with **written reasoning per trade** — this reasoning is
   the actual point of the public experiment, so it must be specific (not generic boilerplate).
4. Validate decisions against the hard constraints above; reject/resize anything that violates them.
5. Simulate fills at last close price (no slippage model in v1 — flagged as a known simplification).
6. Update `state/portfolio.json`, append to `state/trade_log.csv`, write
   `state/weekly_decisions/YYYY-MM-DD.md` with the full reasoning.
7. Refresh the dashboard artifact (NAV vs Nifty Microcap 250 index).

## Known limitations (v1)

- Prices are fetched via Yahoo Finance quote pages (delayed, not tick-accurate) — fine for
  weekly marks, not for intraday precision.
- No slippage/impact cost modeling yet — real fills on illiquid microcaps would be worse.
- Universe market-cap ranks are as of Jul–Dec 2024; NSE reconstitutes the real index
  semi-annually, so the true Nifty Microcap 250 list may have drifted slightly since.
