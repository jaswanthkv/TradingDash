# QuantDesk

A self-hosted stock screening and options trading dashboard for Indian (NSE) and US (S&P 500) markets. Built with FastAPI and vanilla JS — no paid data subscriptions required.

---

## Features

### Screener
- **Minervini SEPA trend template** — 10 criteria: price vs 50/150/200-day SMAs, 200-day slope, 52-week range gates, RS Rating ≥ 70
- **1,690-stock NSE universe** built from official NSE market cap data (≥ 500 Cr), covering the full small/mid-cap range that standard indices miss
- **Market cap filter** — default 1,000–10,000 Cr; updates instantly on input change
- **Column sorting** by RS Rating or Market Cap (asc/desc toggle)
- **US market** — same criteria against the S&P 500 universe
- Same-day price cache: first run downloads ~5 years of daily OHLCV, subsequent runs are instant
- **Buy Top 20** — equal-rupee buy across the top 20 SEPA full-pass stocks by RS Rating; enter a budget, review a live-priced order preview, confirm to place CNC market orders on Kite (India only)
- **Holdings panel** — view current NSE equity holdings with live P&L and sell (full or partial) at market

### Pulse (NIFTY Options Signal)
- **Live Heikin-Ashi signal** — 30-minute candle HA direction for NIFTY (LONG / SHORT / FLAT)
- **One-click option sell** — fetches ATM weekly expiry, shows LTP and premium, places NRML order on Kite
- **Auto-execute mode** — fires 3 minutes after each 30-min candle close during market hours; flips position automatically on signal change
- Full execution log with timestamps

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python · FastAPI · Uvicorn |
| Market data | yfinance · NSE CSV universe · SEBI MCap Excel |
| Broker integration | Zerodha Kite Connect API |
| Frontend | Vanilla JS · HTML/CSS · Chart.js |
| Data processing | pandas · NumPy |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/jaswanthkv/TradingDash.git
cd TradingDash
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file:

```env
PORT=8000

# Required only for Pulse (options trading)
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret

# Optional: override the NSE universe CSV
# UNIVERSE_CSV=~/Downloads/ind_niftytotalmarket_list.csv
```

Get a Kite API key at [kite.trade](https://kite.trade). The screener works without it.

### 3. Run

```bash
uvicorn server:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

On first load the screener downloads ~5 years of daily prices for 1,690 NSE stocks (~3–5 min). Subsequent loads use the same-day cache and are instant.

---

## Architecture

```
server.py          FastAPI app — screener + pulse endpoints
minervini.py       SEPA criteria computation (vectorised pandas)
strategy.py        Universe loading, price download, MCap data
trade_live.py      Kite integration — equity buy/sell, holdings (Buy Top 20)
pulse_live.py      Kite integration — signal, positions, orders
kite_auth.py       OAuth flow for Kite Connect
config.py          Env-based configuration
mcap_fy25h2.xlsx   NSE official average MCap (Jul–Dec 2025)
```

The screener pipeline:
1. Load universe from `mcap_fy25h2.xlsx` (all NSE stocks ≥ 500 Cr)
2. Batch-download OHLCV via yfinance (100 tickers/batch, same-day cache)
3. Vectorised SEPA computation across all dates × stocks (pandas rolling)
4. Client-side MCap filter and RS/MCap column sort

---

## Screener Criteria (Minervini SEPA)

| # | Criterion |
|---|---|
| C1 | Price > 150-day SMA |
| C2 | Price > 200-day SMA |
| C3 | 150-day SMA > 200-day SMA |
| C4 | 200-day SMA trending up (25-day slope > 0) |
| C5 | 50-day SMA > 150-day SMA |
| C6 | 50-day SMA > 200-day SMA |
| C7 | Price > 50-day SMA |
| C8 | Price within 25% of 52-week high |
| C9 | Price ≥ 30% above 52-week low |
| C10 | RS Rating ≥ 70 (cross-sectional 12-month return percentile) |

---

## License

MIT
