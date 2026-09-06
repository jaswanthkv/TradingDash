"""
config.py — Configuration for the QuantDesk dashboard.
"""
import os

_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

PORT         = int(os.environ.get("PORT", 8000))
UNIVERSE_CSV = os.environ.get(
    "UNIVERSE_CSV",
    os.path.expanduser("~/Downloads/ind_niftymicrocap250_list.csv")
)
UNIVERSE_CSV_2 = os.environ.get("UNIVERSE_CSV_2", "")

# Dhan broker API — only needed for /api/live/* real-money endpoints.
# Order placement requires a static IP whitelisted with Dhan.
DHAN_CLIENT_ID    = os.environ.get("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
