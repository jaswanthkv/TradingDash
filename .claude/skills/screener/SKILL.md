---
name: screener
description: "Loads full context for the QuantDesk SEPA screener before working on it. Use whenever making changes to the screener — criteria logic, UI table, CSV export, filters, sorting, universe, or API endpoint. Triggers: screener, SEPA, screener table, screener criteria, screener filter, screener sort, screener CSV, screener market, screener universe, RS rating, minervini, trend template, screen stocks"
---

# Screener Context Loader

Read these files **in full before doing anything else**. Each one owns a distinct layer
of the screener — changes often touch more than one layer.

## Files to read (always)

```
minervini.py          — SEPA criteria computation + screen_on_date()
server.py             — /api/screener endpoint (lines 65–120)
strategy.py           — universe loading, price download, MCap data
index.html            — screener UI: HTML ~line 448-515, JS ~line 1085-1540
```

## Architecture at a glance

```
strategy.py
  load_universe(market)         → list of tickers (NSE: mcap_fy25h2.xlsx ≥500Cr)
  download_data(tickers, ...)   → (close, volume, high, low) DataFrames, same-day cache

minervini.py
  compute_sepa(close, high, low) → dict of DataFrames (sma50/150/200, rs_rating, c1–c10 …)
  screen_on_date(sepa, close, ref_date, stocks) → list[dict] sorted full-pass first, then RS desc

server.py  GET /api/screener?market=india|us&fresh=bool
  → calls strategy + minervini, attaches company names, returns JSON {rows, as_of, full_pass, …}

index.html
  _screenData[]       in-memory row cache (cleared on market switch)
  loadScreener(fresh) fetches /api/screener, stores into _screenData, calls renderScreen()
  renderScreen()      applies MCap filter + sort, writes <tbody id="screenBody">
  downloadScreenerCSV() converts _screenData → CSV blob, triggers browser download
```

## Row schema (each element of `rows` / `_screenData`)

| Field | Type | Notes |
|---|---|---|
| ticker | str | e.g. `RELIANCE.NS` |
| symbol | str | `.NS` stripped |
| name | str | company display name |
| price | float | last close |
| rs_rating | float | 0–99 cross-sectional 12m return percentile |
| ema9, ema21 | float | exponential MAs |
| sma50, sma150, sma200 | float | simple MAs (forward-filled ≤5 gaps) |
| high52w, low52w | float | 252-bar intraday H/L rolling max/min |
| pct_from_52h | float | % below 52w high (positive = below) |
| pct_above_52l | float | % above 52w low |
| criteria | dict | `{c1:bool … c10:bool}` |
| passing | int | criteria passed (0–10) |
| sepa_pass | bool | passing == 10 |
| mcap_cr | float\|null | NSE average MCap in Crores (India only) |

## SEPA criteria (C1–C10)

| Key | Rule |
|---|---|
| c1 | Price > 150d SMA |
| c2 | Price > 200d SMA |
| c3 | 150d SMA > 200d SMA |
| c4 | 200d SMA slope positive over last 25 days |
| c5 | 50d SMA > 150d SMA |
| c6 | 50d SMA > 200d SMA |
| c7 | Price > 50d SMA |
| c8 | Price ≥ 75% of 52w High (within 25% of high) |
| c9 | Price ≥ 130% of 52w Low (30%+ above low) |
| c10 | RS Rating ≥ 70 |

## Key invariants — don't break these

- `compute_sepa` forward-fills closes up to 5 days (`ffill(limit=5)`) before rolling MAs
  so a single missing yfinance bar doesn't blank sma50/150/200.
- `screen_on_date` evaluates **each stock at its own last valid close** on/before `ref_date`,
  not the single most-recent frame date — prevents silently dropping stocks with data gaps.
- `rs_rating` is a cross-sectional percentile across ALL stocks for that date, so adding or
  removing tickers from the universe shifts everyone's score.
- Same-day price cache key = `(market, years=5)`. `fresh=true` bypasses it; `fresh=false`
  reuses it. The screener and backtest share the same cache.
- MCap filter (`mcapMin` / `mcapMax`) is applied **client-side** in `renderScreen()`, not
  server-side, so the full universe is always in `_screenData`.

## Common tasks

**Add a new column to the table**
1. Compute the value in `screen_on_date()` (minervini.py) and add it to the row dict.
2. Add a `<th>` in `index.html` screener thead (~line 498).
3. Add the `<td>` in the `rows.map()` inside `renderScreen()` (~line 1130).
4. Add it to `downloadScreenerCSV()` headers + row builder if it should appear in the CSV.

**Change a SEPA criterion**
- Edit `compute_sepa()` in minervini.py.
- Update `CRITERIA_LABELS` and `CRITERIA_SHORT` dicts if the label changes.
- Check `screen_on_date()` — it reads criteria keys from `CRITERIA_LABELS`.

**Change the universe**
- India: edit `load_universe("india")` in strategy.py — reads `mcap_fy25h2.xlsx`.
- US: edit `load_universe("us")` in strategy.py — reads `sp500_constituents.csv`.

**Add a server-side filter**
- Add a query param to `GET /api/screener` in server.py and filter `rows` before returning.
- Update `loadScreener()` in index.html to pass the param in the fetch URL.
