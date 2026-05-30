"""
config.py — Configuration for the ML Strategy dashboard.
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

KITE_API_KEY    = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".kite_token.json")
