"""
kite_auth.py — Kite Connect authentication.
Token is persisted to a JSON file and checked daily — Kite invalidates tokens at midnight.
"""
import json, os, datetime, logging
from kiteconnect import KiteConnect
from config import KITE_API_KEY, KITE_API_SECRET, KITE_TOKEN_FILE

logger = logging.getLogger(__name__)


def get_kite() -> KiteConnect:
    """Return authenticated KiteConnect instance. Raises RuntimeError if not logged in."""
    kite = KiteConnect(api_key=KITE_API_KEY)
    data = _load_token()
    if not data:
        raise RuntimeError("Kite not authenticated. Visit /api/kite/login to connect.")
    kite.set_access_token(data["access_token"])
    return kite


def kite_status() -> dict:
    """Safe status check — never raises."""
    if not KITE_API_KEY or not KITE_API_SECRET:
        return {"connected": False, "reason": "KITE_API_KEY / KITE_API_SECRET not set in config.py"}
    data = _load_token()
    if not data:
        return {"connected": False, "reason": "Not logged in today — click Connect Kite"}
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(data["access_token"])
        profile = kite.profile()
        return {
            "connected":  True,
            "user_name":  profile["user_name"],
            "user_id":    profile["user_id"],
            "token_date": data["date"],
        }
    except Exception as exc:
        return {"connected": False, "reason": str(exc)}


def get_login_url() -> str:
    return KiteConnect(api_key=KITE_API_KEY).login_url()


def complete_login(request_token: str) -> dict:
    kite    = KiteConnect(api_key=KITE_API_KEY)
    session = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
    _save_token(session["access_token"])
    logger.info("Kite login OK for %s", session.get("user_id", ""))
    return {"user_id": session.get("user_id", ""), "user_name": session.get("user_name", "")}


# ── internal ──────────────────────────────────────────────────────────────────

def _load_token() -> dict | None:
    if not os.path.exists(KITE_TOKEN_FILE):
        return None
    try:
        with open(KITE_TOKEN_FILE) as f:
            data = json.load(f)
        if data.get("date") != datetime.date.today().isoformat():
            return None   # expired
        return data
    except Exception:
        return None


def _save_token(access_token: str):
    os.makedirs(os.path.dirname(os.path.abspath(KITE_TOKEN_FILE)), exist_ok=True)
    with open(KITE_TOKEN_FILE, "w") as f:
        json.dump({
            "access_token": access_token,
            "date": datetime.date.today().isoformat(),
        }, f)
