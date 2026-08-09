# Weekly Runbook (what Claude does each cycle, manually or via scheduled task)

1. **Load state**: read `state/portfolio.json`, `universe.csv`, recent `state/trade_log.csv`.
2. **Fetch prices**: for every current holding + any candidate under consideration, fetch
   `https://finance.yahoo.com/quote/<SYMBOL>.NS/` via web_fetch and pull the last price.
   Also fetch `https://finance.yahoo.com/quote/NIFTY_MICROCAP250.NS/` for the benchmark level.
   Write all of this to `state/prices_YYYY-MM-DD.json` (format: `{"SYMBOL": price, ..., "__BENCHMARK__": value}`).
3. **Screen + decide**: review holdings and a slice of the universe, write buy/sell/hold
   decisions with real reasoning per `docs/decision_rules.md`. Save as
   `state/decisions_YYYY-MM-DD.json`.
4. **Execute**: run
   `python3 scripts/paper_trader.py apply --decisions state/decisions_YYYY-MM-DD.json --prices state/prices_YYYY-MM-DD.json`
   This validates against hard rules, updates `portfolio.json`, appends `trade_log.csv`.
   Read the script's output for any REJECTED trades or WARNINGs and note them in the write-up.
5. **Write the log**: save `state/weekly_decisions/YYYY-MM-DD.md` with full reasoning for
   every trade (accepted and rejected) — this is the public-facing artifact that makes the
   experiment meaningful, not just the returns number.
6. **Refresh dashboard**: update the Cowork artifact with the new NAV point, holdings table,
   and latest reasoning.

On weeks with no new prices needed for a trade decision (holding through), still run
`paper_trader.py mark --prices ...` so the NAV history has a continuous weekly series for
the benchmark comparison chart.
