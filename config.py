"""
config.py — All settings in one place. Edit this file to change behaviour.
"""
import os

# ── Anthropic ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL             = "claude-sonnet-4-6"

# ── Capital & risk ────────────────────────────────────────────────────────────
CAPITAL       = 500_000   # ₹5 Lakhs total capital
MAX_POSITIONS = 10        # hard cap on concurrent open positions
MIN_ADTV_CR   = 5.0       # minimum avg daily turnover in crores
MAX_ATR_PCT   = 4.0       # skip stocks more volatile than this

# ── Scanner ───────────────────────────────────────────────────────────────────
HISTORY_DAYS    = 504     # ~2 years of price history
TOP_N_TO_CLAUDE = 25      # top N candidates sent to Claude after scoring
BENCHMARK       = "^CRSLDX"  # Nifty 500 Total Return Index

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = "/Users/jaswanth"
POSITIONS_FILE = f"{BASE_DIR}/tradeboard/open_positions.json"
TRADE_LOG_FILE = f"{BASE_DIR}/tradeboard/trade_log.json"
PICKS_DIR      = f"{BASE_DIR}/tradeboard/scans"
UNIVERSE_CSV   = f"{BASE_DIR}/Downloads/MW-NIFTY-MICROCAP-250-18-Apr-2026.csv"

# ── Kite Connect ─────────────────────────────────────────────────────────────
KITE_API_KEY    = os.environ.get("KITE_API_KEY",    "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_TOKEN_FILE = f"{BASE_DIR}/tradeboard/.kite_token.json"

# ── Server ────────────────────────────────────────────────────────────────────
PORT           = 8000
PRICE_CACHE_TTL = 300     # seconds before yfinance is re-fetched
SSE_INTERVAL    = 60      # seconds between live price pushes
