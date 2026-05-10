"""
options.py — Option chain, expiries, short positions for 0DTE/1DTE selling.
"""
import datetime, json, logging, re
import warnings; warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

NIFTY_LOT   = 25
SENSEX_LOT  = 10
NIFTY_STEP  = 50
SENSEX_STEP = 100

_instruments_cache: dict = {}   # segment → (date, list)


def _instruments(segment: str) -> list:
    today = datetime.date.today()
    if segment in _instruments_cache:
        d, insts = _instruments_cache[segment]
        if d == today:
            return insts
    from kite_auth import get_kite
    insts = get_kite().instruments(segment)
    _instruments_cache[segment] = (today, insts)
    logger.info("Loaded %d instruments for %s", len(insts), segment)
    return insts


def get_spot(symbol: str) -> float:
    from kite_auth import get_kite
    key = "NSE:NIFTY 50" if symbol == "NIFTY" else "BSE:SENSEX"
    q = get_kite().quote([key])
    return float(q[key]["last_price"])


def atm_strike(spot: float, symbol: str) -> int:
    step = NIFTY_STEP if symbol == "NIFTY" else SENSEX_STEP
    return int(round(spot / step) * step)


def get_expiries(symbol: str) -> list[str]:
    segment = "NFO" if symbol == "NIFTY" else "BFO"
    name    = symbol  # "NIFTY" or "SENSEX"
    today   = datetime.date.today()
    expiries = sorted({
        inst["expiry"] for inst in _instruments(segment)
        if inst["name"] == name
        and inst["instrument_type"] in ("CE", "PE")
        and inst["expiry"] and inst["expiry"] >= today
    })
    return [str(e) for e in expiries[:6]]


def get_chain(symbol: str, expiry_str: str, n: int = 8) -> dict:
    from kite_auth import get_kite
    kite    = get_kite()
    segment = "NFO" if symbol == "NIFTY" else "BFO"
    step    = NIFTY_STEP if symbol == "NIFTY" else SENSEX_STEP
    expiry  = datetime.date.fromisoformat(expiry_str)

    spot = get_spot(symbol)
    atm  = atm_strike(spot, symbol)

    valid_strikes = {atm + i * step for i in range(-n, n + 1)}

    relevant = [
        inst for inst in _instruments(segment)
        if inst["name"] == symbol
        and inst["expiry"] == expiry
        and inst["instrument_type"] in ("CE", "PE")
        and int(inst["strike"]) in valid_strikes
    ]
    if not relevant:
        return {"chain": [], "spot": round(spot, 2), "atm": atm}

    ts_keys = [f"{segment}:{i['tradingsymbol']}" for i in relevant]
    quotes  = {}
    for start in range(0, len(ts_keys), 500):
        try:
            quotes.update(kite.quote(ts_keys[start:start + 500]))
        except Exception as exc:
            logger.error("Quote error: %s", exc)

    chain: dict[int, dict] = {}
    for inst in relevant:
        s = int(inst["strike"])
        if s not in chain:
            chain[s] = {"strike": s, "CE": None, "PE": None}
        key = f"{segment}:{inst['tradingsymbol']}"
        q   = quotes.get(key, {})
        d   = q.get("depth", {})
        bids = d.get("buy",  [])
        asks = d.get("sell", [])
        chain[s][inst["instrument_type"]] = {
            "tradingsymbol": inst["tradingsymbol"],
            "exchange":      segment,
            "lot_size":      int(inst["lot_size"]),
            "ltp":   round(float(q.get("last_price", 0)), 2),
            "bid":   round(float(bids[0]["price"]) if bids else 0, 2),
            "ask":   round(float(asks[0]["price"]) if asks else 0, 2),
            "oi":    int(q.get("oi", 0)),
            "volume":int(q.get("volume", 0)),
        }

    rows = []
    for s in sorted(chain.keys()):
        r = chain[s]
        r["atm"]         = s == atm
        r["distance"]    = s - atm
        r["pct_from_atm"]= round((s - atm) / spot * 100, 2)
        rows.append(r)

    return {"chain": rows, "spot": round(spot, 2), "atm": atm}


_HEDGE_SYMBOLS: set[str] = set()   # symbols to exclude (covered calls, long hedges)

def mark_as_hedge(tradingsymbol: str):
    _HEDGE_SYMBOLS.add(tradingsymbol)

def unmark_hedge(tradingsymbol: str):
    _HEDGE_SYMBOLS.discard(tradingsymbol)


def get_short_positions() -> list[dict]:
    from kite_auth import get_kite
    pos = get_kite().positions()
    out = []
    for p in pos.get("net", []):
        if p["quantity"] >= 0 or p["exchange"] not in ("NFO", "BFO"):
            continue
        if p["tradingsymbol"] in _HEDGE_SYMBOLS:
            continue   # exclude covered calls / hedges from active shorts panel
        qty   = abs(p["quantity"])
        entry = round(float(p["average_price"]), 2)
        cmp   = round(float(p["last_price"]),    2)
        pnl   = round((entry - cmp) * qty, 2)
        pct   = round((entry - cmp) / entry * 100, 2) if entry > 0 else 0
        ratio = cmp / entry if entry > 0 else 0
        status = "breach" if ratio >= 2.0 else "warning" if ratio >= 1.5 else "safe"
        out.append({
            "tradingsymbol": p["tradingsymbol"],
            "exchange":      p["exchange"],
            "qty":           qty,
            "entry_price":   entry,
            "cmp":           cmp,
            "pnl":           pnl,
            "pnl_pct":       pct,
            "day_pnl":       round(float(p.get("day_m2m", 0)), 2),
            "status":        status,
            "sl_2x":         round(entry * 2, 2),
            "sl_3x":         round(entry * 3, 2),
        })
    return out


def claude_adjust_position(positions: list, symbol: str, spot: float, chain: list, hedge_notes: str = "") -> dict:
    """
    Claude analyzes active short positions and recommends mid-trade adjustments.
    positions: list of short position dicts from get_short_positions()
    """
    from config import ANTHROPIC_API_KEY, MODEL
    import anthropic

    vix = get_vix()
    today = datetime.date.today()

    # summarize positions for Claude
    pos_text = ""
    for p in positions:
        decay_pct = p["pnl_pct"]
        cmp_ratio = p["cmp"] / p["entry_price"] if p["entry_price"] > 0 else 0
        pos_text += (
            f"  {p['tradingsymbol']}: Entry ₹{p['entry_price']} → CMP ₹{p['cmp']} "
            f"({'+' if decay_pct>=0 else ''}{decay_pct:.1f}% decay) | "
            f"P&L ₹{p['pnl']:+.0f} | "
            f"2× SL at ₹{p['sl_2x']} | Status: {p['status'].upper()}\n"
        )

    # nearby chain for roll suggestions
    chain_rows = []
    for r in chain:
        ce = r.get("CE") or {}
        pe = r.get("PE") or {}
        ce_bid = ce.get("bid") or ce.get("ltp") or 0
        pe_bid = pe.get("bid") or pe.get("ltp") or 0
        if ce_bid > 5 or pe_bid > 5:
            chain_rows.append(
                f"  Strike {r['strike']:>7} (dist {r['distance']:>+5}): "
                f"CE ₹{ce_bid} OI {ce.get('oi',0)//100000:.1f}L | "
                f"PE ₹{pe_bid} OI {pe.get('oi',0)//100000:.1f}L"
            )
    chain_text = "\n".join(chain_rows[:20])

    prompt = f"""You are an expert options trader managing short theta positions in Indian index options.

CURRENT MARKET:
  {symbol} Spot: {spot}
  India VIX: {vix if vix else 'unavailable'}
  Date: {today}

ACTIVE SHORT POSITIONS:
{pos_text}

NEARBY OPTION CHAIN (for roll suggestions):
{chain_text}

EXISTING HEDGES / CONTEXT:
{hedge_notes if hedge_notes.strip() else "None provided"}

ADJUSTMENT RULES:
- HOLD: position is safe, premium decayed >30%, spot far from strikes
- CLOSE_LEG: one leg is threatened (CMP approaching 1.5× entry or strike nearly breached) — close that leg, keep profitable one
- CLOSE_ALL: CMP has reached 2× entry on any leg OR spot has breached a strike — immediate exit
- ROLL: close threatened leg and re-sell at a safer strike further OTM (only if premium is still meaningful ≥ ₹25)
- HEDGE: buy a closer strike to cap loss without closing (use if directional move but mean-reversion likely)

Analyze the position and recommend ONE action.

Return ONLY this JSON:
{{
  "action": "HOLD" | "CLOSE_LEG" | "CLOSE_ALL" | "ROLL" | "HEDGE",
  "urgency": "low" | "medium" | "high",
  "affected_leg": "<tradingsymbol or null>",
  "roll_to_strike": <number or null>,
  "roll_to_tradingsymbol": "<string or null>",
  "roll_to_exchange": "<string or null>",
  "roll_premium": <number or null>,
  "reasoning": "<2-3 sentences explaining what's happening and why this action>",
  "specific_steps": ["step 1", "step 2", ...]
}}"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=MODEL, max_tokens=600, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"Claude returned no JSON: {raw}")
    return json.loads(m.group())


def get_vix() -> float:
    """Fetch India VIX from Kite."""
    try:
        from kite_auth import get_kite
        q = get_kite().quote(["NSE:INDIA VIX"])
        return round(float(q["NSE:INDIA VIX"]["last_price"]), 2)
    except Exception:
        return 0.0


def claude_select_strikes(symbol: str, expiry_str: str, strategy: str, chain: list, spot: float, atm: int) -> dict:
    """
    Ask Claude to select optimal strikes for the short options strategy.
    Returns {ce_strike, pe_strike, ce_ts, pe_ts, reasoning}.
    """
    from config import ANTHROPIC_API_KEY, MODEL
    import anthropic

    expiry = datetime.date.fromisoformat(expiry_str)
    today  = datetime.date.today()
    dte    = (expiry - today).days
    vix    = get_vix()

    # Build a compact chain table for Claude (only strikes with valid data)
    rows = []
    for r in chain:
        ce = r.get("CE") or {}
        pe = r.get("PE") or {}
        ce_bid = ce.get("bid") or ce.get("ltp") or 0
        pe_bid = pe.get("bid") or pe.get("ltp") or 0
        if ce_bid == 0 and pe_bid == 0:
            continue
        ce_oi = ce.get("oi", 0)
        pe_oi = pe.get("oi", 0)
        rows.append({
            "strike":   r["strike"],
            "dist":     r["distance"],
            "ce_bid":   ce_bid,
            "ce_oi":    ce_oi,
            "pe_bid":   pe_bid,
            "pe_oi":    pe_oi,
            "ce_ts":    ce.get("tradingsymbol", ""),
            "pe_ts":    pe.get("tradingsymbol", ""),
            "ce_exch":  ce.get("exchange", "NFO"),
            "pe_exch":  pe.get("exchange", "NFO"),
            "lot_size": ce.get("lot_size") or pe.get("lot_size") or 25,
        })

    chain_text = "Strike | Dist | CE Bid | CE OI    | PE Bid | PE OI\n"
    chain_text += "-" * 65 + "\n"
    for r in rows:
        ce_oi_f = f"{r['ce_oi']/1e5:.1f}L" if r["ce_oi"] > 1e5 else str(r["ce_oi"])
        pe_oi_f = f"{r['pe_oi']/1e5:.1f}L" if r["pe_oi"] > 1e5 else str(r["pe_oi"])
        chain_text += f"{r['strike']:>7} | {r['dist']:>+5} | {r['ce_bid']:>6} | {ce_oi_f:>8} | {r['pe_bid']:>6} | {pe_oi_f}\n"

    prompt = f"""You are an expert Indian options trader specializing in 0DTE and 1DTE theta-selling strategies.

Instrument: {symbol}
Spot: {spot}
ATM: {atm}
Expiry: {expiry_str} ({dte} DTE)
Strategy: Short {strategy}
India VIX: {vix if vix else 'unavailable'}

Option Chain:
{chain_text}

Select the BEST strikes to short for maximum theta decay with acceptable gamma risk.

Rules to follow:
- 0 DTE: short strikes must be at least 150 pts OTM from spot (gamma is lethal near ATM)
- 1 DTE: short strikes should be 80–150 pts OTM
- Prefer strikes just BEYOND the highest OI level (OI wall = natural resistance/support)
- CE strike: prefer just above major CE OI cluster
- PE strike: prefer just below major PE OI cluster
- Minimum premium to collect: ₹20 per leg (below this not worth the risk)
- If VIX > 18: go further OTM for safety
- For STRADDLE: both CE and PE at same strike (ATM)
- For STRANGLE: CE above ATM, PE below ATM

Return ONLY this JSON (no explanation outside JSON):
{{
  "ce_strike": <number>,
  "pe_strike": <number>,
  "ce_tradingsymbol": "<string>",
  "pe_tradingsymbol": "<string>",
  "ce_exchange": "<NFO or BFO>",
  "pe_exchange": "<NFO or BFO>",
  "lot_size": <number>,
  "ce_premium": <number>,
  "pe_premium": <number>,
  "total_premium": <number>,
  "reasoning": "<2-3 sentences explaining the strike selection>"
}}"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()

    # extract JSON
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"Claude returned no JSON: {raw}")
    return json.loads(m.group())


# ── intraday directional pick ──────────────────────────────────────────────────

def get_intraday_candles(symbol: str) -> list[dict]:
    """Fetch last 2 days of 15-min OHLCV candles via yfinance."""
    import yfinance as yf
    ticker = "^NSEI" if symbol == "NIFTY" else "^BSESN"
    df = yf.download(ticker, period="2d", interval="15m",
                     auto_adjust=True, progress=False)
    if df.empty:
        return []
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "time":  ts.strftime("%H:%M"),
            "open":  round(float(row["Open"]),  2),
            "high":  round(float(row["High"]),  2),
            "low":   round(float(row["Low"]),   2),
            "close": round(float(row["Close"]), 2),
            "vol":   int(row.get("Volume", 0)),
        })
    return candles[-20:]   # last 20 candles (5 hours)


def claude_intraday_pick(symbol: str, chain: list, spot: float, atm: int) -> dict:
    """
    Claude reads recent price candles + option chain and suggests
    ONE directional short leg for intraday: sell CE (bearish) or sell PE (bullish).
    Returns action, strike, tradingsymbol, premium, SL, reasoning.
    """
    from config import ANTHROPIC_API_KEY, MODEL
    import anthropic

    vix     = get_vix()
    now     = datetime.datetime.now().strftime("%H:%M")
    candles = get_intraday_candles(symbol)

    if not candles:
        raise ValueError("Could not fetch intraday price data")

    prev_close = candles[0]["open"]   # approximate
    day_open   = next((c for c in candles if c["time"] <= "09:30"), candles[0])["open"]
    current    = candles[-1]["close"]
    day_high   = max(c["high"]  for c in candles)
    day_low    = min(c["low"]   for c in candles)
    day_change = round((current - day_open) / day_open * 100, 2)

    candle_text = "Time  | Open    | High    | Low     | Close\n"
    candle_text += "-" * 55 + "\n"
    for c in candles:
        candle_text += f"{c['time']}  | {c['open']:>7} | {c['high']:>7} | {c['low']:>7} | {c['close']:>7}\n"

    # compact chain — CE side above ATM, PE side below
    ce_rows = [(r["strike"], r["CE"].get("bid") or r["CE"].get("ltp") or 0,
                r["CE"].get("oi", 0))
               for r in chain if r["distance"] > 0 and r.get("CE")][:6]
    pe_rows = [(r["strike"], r["PE"].get("bid") or r["PE"].get("ltp") or 0,
                r["PE"].get("oi", 0))
               for r in chain if r["distance"] < 0 and r.get("PE")][-6:]

    chain_text  = "CE side (above ATM):\n"
    chain_text += "\n".join(f"  {s}: bid ₹{p} OI {o//100000:.1f}L" for s, p, o in ce_rows)
    chain_text += "\nPE side (below ATM):\n"
    chain_text += "\n".join(f"  {s}: bid ₹{p} OI {o//100000:.1f}L" for s, p, o in pe_rows)

    # resolve tradingsymbols for chain rows
    ts_map = {}
    for r in chain:
        if r.get("CE"):
            ts_map[f"CE_{r['strike']}"] = {
                "ts": r["CE"].get("tradingsymbol",""),
                "exch": r["CE"].get("exchange","NFO"),
                "lot": r["CE"].get("lot_size", 25),
                "prem": r["CE"].get("bid") or r["CE"].get("ltp") or 0,
            }
        if r.get("PE"):
            ts_map[f"PE_{r['strike']}"] = {
                "ts": r["PE"].get("tradingsymbol",""),
                "exch": r["PE"].get("exchange","NFO"),
                "lot": r["PE"].get("lot_size", 25),
                "prem": r["PE"].get("bid") or r["PE"].get("ltp") or 0,
            }

    prompt = f"""You are an expert intraday options trader in Indian markets.

TIME: {now} IST
INSTRUMENT: {symbol} | Spot: {spot} | ATM: {atm}
Day: Open {day_open} | High {day_high} | Low {day_low} | Change {day_change:+.2f}%
India VIX: {vix if vix else 'unavailable'}

RECENT 15-MIN CANDLES (latest at bottom):
{candle_text}

OPTION CHAIN:
{chain_text}

YOUR TASK:
Based ONLY on price action, determine intraday directional bias and suggest ONE option to SHORT (sell):
- BEARISH price action → Sell CE (collect premium, profit if market stays below strike)
- BULLISH price action → Sell PE (collect premium, profit if market stays above strike)
- FLAT/UNCLEAR → return action: "NO_TRADE" with reason

Strike selection for the short:
- Pick a strike that is OTM enough to be safe (delta 0.20–0.30 range)
- Must have minimum ₹25 premium to collect
- Prefer strikes near major OI walls (resistance for CE, support for PE)
- Stop loss: when premium doubles (2× entry)
- Target: 50% premium decay

Key price analysis to consider:
- Is there a clear trend in the candles?
- Where is the market relative to day open and ATM?
- Are there rejection candles, breakouts, or consolidation?
- VIX > 16 = go further OTM

Return ONLY this JSON:
{{
  "action": "SELL_CE" | "SELL_PE" | "NO_TRADE",
  "direction": "BEARISH" | "BULLISH" | "NEUTRAL",
  "strike": <number or null>,
  "type": "CE" | "PE" | null,
  "premium": <number or null>,
  "sl_premium": <number or null>,
  "target_premium": <number or null>,
  "confidence": "low" | "medium" | "high",
  "reasoning": "<2-3 sentences on what the price action shows and why this strike>",
  "key_levels": ["level 1 to watch", "level 2 to watch"]
}}"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=MODEL, max_tokens=512, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"No JSON in response: {raw}")
    result = json.loads(m.group())

    # attach tradingsymbol to result
    key = f"{result.get('type','CE')}_{result.get('strike','')}"
    if key in ts_map:
        result["tradingsymbol"] = ts_map[key]["ts"]
        result["exchange"]      = ts_map[key]["exch"]
        result["lot_size"]      = ts_map[key]["lot"]

    return result
