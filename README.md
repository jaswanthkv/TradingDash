# QuantDesk

A self-hosted stock screener for Indian (NSE) and US (S&P 500) markets. Built with FastAPI and vanilla JS — no paid data subscriptions required, no broker integration.

---

## Features

### Screener
- **Minervini SEPA trend template** — 10 criteria: price vs 50/150/200-day SMAs, 200-day slope, 52-week range gates, RS Rating ≥ 70
- **1,690-stock NSE universe** built from official NSE market cap data (≥ 500 Cr), covering the full small/mid-cap range that standard indices miss
- **Market cap filter** — default 1,000–10,000 Cr; updates instantly on input change
- **Column sorting** by RS Rating or Market Cap (asc/desc toggle)
- **US market** — same criteria against the S&P 500 universe
- Same-day price cache: first run downloads ~5 years of daily OHLCV, subsequent runs are instant

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python · FastAPI · Uvicorn |
| Market data | yfinance · NSE CSV universe · SEBI MCap Excel |
| Frontend | Vanilla JS · HTML/CSS |
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

# Optional: override the NSE universe CSV
# UNIVERSE_CSV=~/Downloads/ind_niftytotalmarket_list.csv
```

### 3. Run

```bash
uvicorn server:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

On first load the screener downloads ~5 years of daily prices for 1,690 NSE stocks (~3–5 min). Subsequent loads use the same-day cache and are instant.

---

## Architecture

```
server.py          FastAPI app — screener endpoint
minervini.py       SEPA criteria computation (vectorised pandas)
strategy.py        Universe loading, price download, MCap data
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
