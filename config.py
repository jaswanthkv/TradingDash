"""
config.py — All settings in one place.
Keys are loaded from .env file → environment variables → defaults below.
"""
import os

# ── Load .env automatically ───────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Anthropic ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL             = "claude-sonnet-4-6"

# ── Capital & risk ────────────────────────────────────────────────────────────
CAPITAL       = 200_000   # ₹2 Lakhs total capital
MAX_POSITIONS = 10        # hard cap on concurrent open positions
MIN_ADTV_CR   = 5.0       # minimum avg daily turnover in crores
MAX_ATR_PCT   = 4.0       # skip stocks more volatile than this

# ── Scanner ───────────────────────────────────────────────────────────────────
HISTORY_DAYS    = 504     # ~2 years of price history
TOP_N_TO_CLAUDE = 25      # top N candidates sent to Claude after scoring
BENCHMARK       = "^CRSLDX"  # Nifty 500 Total Return Index

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(__file__)
POSITIONS_FILE = os.path.join(BASE_DIR, "open_positions.json")
TRADE_LOG_FILE = os.path.join(BASE_DIR, "trade_log.json")
PICKS_DIR      = os.path.join(BASE_DIR, "scans")
UNIVERSE_CSV   = os.environ.get(
    "UNIVERSE_CSV",
    os.path.expanduser("~/Downloads/MW-NIFTY-MICROCAP-250-18-Apr-2026.csv")
)

# ── Kite Connect ─────────────────────────────────────────────────────────────
KITE_API_KEY    = os.environ.get("KITE_API_KEY",    "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_TOKEN_FILE = os.path.join(BASE_DIR, ".kite_token.json")

# ── Server ────────────────────────────────────────────────────────────────────
PORT            = int(os.environ.get("PORT", 8000))
PRICE_CACHE_TTL = 300     # seconds before yfinance is re-fetched
SSE_INTERVAL    = 60      # seconds between live price pushes
