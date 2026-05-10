"""
engine.py — All trading logic. No prints, no user input. Import-safe.
"""
import os, json, re, datetime, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import anthropic

from config import (
    ANTHROPIC_API_KEY, MODEL, CAPITAL, MAX_POSITIONS,
    MIN_ADTV_CR, MAX_ATR_PCT, HISTORY_DAYS, TOP_N_TO_CLAUDE,
    BENCHMARK, POSITIONS_FILE, TRADE_LOG_FILE, PICKS_DIR, UNIVERSE_CSV,
)

os.makedirs(PICKS_DIR, exist_ok=True)

# ── Universe ──────────────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    symbols = set()
    try:
        df = pd.read_csv(UNIVERSE_CSV)
        df.columns = [c.strip() for c in df.columns]
        col = next((c for c in df.columns if c.upper() == "SYMBOL"), None)
        if col:
            for s in df[col].dropna():
                s = str(s).strip()
                if s and " " not in s and s.upper() != "SYMBOL":
                    symbols.add(s + ".NS")
    except Exception:
        pass
    return sorted(symbols)

# ── Price fetch ───────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], days: int, min_bars: int = 100) -> dict[str, pd.DataFrame]:
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    raw   = yf.download(tickers, start=str(start), end=str(end),
                        auto_adjust=True, progress=False, threads=True)
    data = {}
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
            else:
                df = raw.copy().dropna(how="all")
            if len(df) >= min_bars:
                data[ticker] = df
        except Exception:
            pass
    return data

# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def compute_stage(df: pd.DataFrame) -> tuple[str, int]:
    close     = df["Close"].squeeze().dropna()
    sma30     = close.rolling(30).mean().iloc[-1]
    sma150    = close.rolling(150).mean().iloc[-1]
    sma200    = close.rolling(200).mean().iloc[-1]
    sma200_21 = close.rolling(200).mean().iloc[-21]
    price     = close.iloc[-1]
    criteria  = [price > sma30, price > sma150, price > sma200,
                 sma30 > sma150, sma150 > sma200, sma200 > sma200_21]
    score = sum(bool(c) for c in criteria)
    if score >= 5: return "Stage2", 100
    if score >= 3: return "Late1",  75
    if score >= 1: return "Early1", 50
    return "Stage4", 25


def compute_rs_scores(price_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    returns = {}
    for ticker, df in price_data.items():
        close = df["Close"].squeeze()
        if len(close) >= 252:
            q1 = close.iloc[-1]   / close.iloc[-63]  - 1
            q2 = close.iloc[-63]  / close.iloc[-126] - 1
            q3 = close.iloc[-126] / close.iloc[-189] - 1
            q4 = close.iloc[-189] / close.iloc[-252] - 1
            returns[ticker] = 0.63*q1 + 0.125*q2 + 0.125*q3 + 0.125*q4
        elif len(close) >= 63:
            returns[ticker] = close.iloc[-1] / close.iloc[-63] - 1
    if not returns:
        return {}
    series = pd.Series(returns)
    return (series.rank(pct=True) * 98 + 1).round(1).to_dict()


def compute_5ma_status(df: pd.DataFrame, entry_price: float | None = None) -> dict:
    close  = df["Close"].squeeze()
    ema5   = close.ewm(span=5,  adjust=False).mean()
    ema10  = close.ewm(span=10, adjust=False).mean()
    sma20  = close.rolling(20).mean()

    cmp          = float(close.iloc[-1])
    last_ema5    = float(ema5.iloc[-1])
    last_ema10   = float(ema10.iloc[-1])

    # Exit: 2 consecutive closes below 10EMA
    below_ema10_now  = cmp < last_ema10
    below_ema10_prev = float(close.iloc[-2]) < float(ema10.iloc[-2])
    # Warning: below 5EMA but not yet 2 closes below 10EMA
    below_ema5_now   = cmp < last_ema5

    if below_ema10_now and below_ema10_prev:
        urgency, status = "exit",    "10MA Break"
    elif below_ema5_now:
        urgency, status = "warning", "Below 5MA"
    else:
        urgency, status = "safe",    "10MA Safe"

    days_above = 0
    for i in range(len(close) - 1, -1, -1):
        if float(close.iloc[i]) > float(ema10.iloc[i]):
            days_above += 1
        else:
            break

    out = dict(urgency=urgency, five_ma_status=status, days_above_5ma=days_above,
               ema5=round(last_ema5, 2), ema10=round(last_ema10, 2),
               sma20=round(float(sma20.iloc[-1]), 2), current_price=round(cmp, 2))

    if entry_price:
        atr        = compute_atr(df)
        init_stop  = entry_price - 2 * atr
        risk       = max(entry_price - init_stop, 0.01)
        trail_stop = max(init_stop, last_ema10)
        out.update(
            initial_stop  = round(init_stop, 2),
            trailing_stop = round(trail_stop, 2),
            pnl_pct       = round((cmp - entry_price) / entry_price * 100, 2),
            r_multiple    = round((cmp - entry_price) / risk, 2),
        )
    return out

# ── Entry signal classifier ───────────────────────────────────────────────────

def _vcp_contractions(prices: np.ndarray) -> int:
    n, seg = len(prices), max(len(prices) // 4, 5)
    ranges = [(prices[i:i+seg].max() - prices[i:i+seg].min()) / prices[i:i+seg].min() * 100
              for i in range(0, n - seg + 1, seg)]
    if len(ranges) < 3:
        return 0
    return sum(1 for i in range(1, len(ranges)) if ranges[i] < ranges[i-1] * 0.90)


class EntrySignalEngine:
    def __init__(self, df: pd.DataFrame):
        self.df     = df
        self.close  = df["Close"].squeeze().dropna()
        self.high   = df["High"].squeeze()
        self.low    = df["Low"].squeeze()
        self.volume = df["Volume"].squeeze()
        self.last   = float(self.close.iloc[-1])

    def find_pivot_high(self, lookback: int = 40, confirm: int = 3) -> float | None:
        window = self.high.tail(lookback)
        if len(window) < confirm * 2 + 1:
            return None
        arr = window.values
        for i in range(len(arr) - confirm - 1, confirm - 1, -1):
            val = arr[i]
            if (all(arr[i-j] < val for j in range(1, confirm+1)) and
                    all(arr[i+j] < val for j in range(1, confirm+1))):
                return float(val)
        return None

    def tightness(self, window: int = 10) -> float:
        last_n = self.close.tail(window)
        return float((last_n.max() - last_n.min()) / last_n.mean() * 100)

    def vol_dry_up(self, window: int = 10) -> float:
        avg50 = self.volume.tail(50).mean()
        return float(self.volume.tail(window).mean() / avg50) if avg50 else 1.0

    def pocket_pivot(self) -> tuple[bool, float]:
        recent    = self.df.tail(11)
        today_up  = recent["Close"].iloc[-1] > recent["Open"].iloc[-1]
        if not today_up:
            return False, 0.0
        down_days = recent.iloc[:-1]
        down_days = down_days[down_days["Close"] < down_days["Open"]]
        if len(down_days) == 0:
            return False, 0.0
        today_vol    = float(recent["Volume"].iloc[-1])
        max_down_vol = float(down_days["Volume"].max())
        ratio = today_vol / max_down_vol if max_down_vol else 0
        return (today_vol > max_down_vol), round(ratio, 2)

    def breakout(self, pivot: float | None) -> tuple[bool, float]:
        if pivot is None:
            return False, 0.0
        avg50     = float(self.volume.tail(50).mean())
        today_vol = float(self.volume.iloc[-1])
        ratio     = today_vol / avg50 if avg50 else 0
        broke     = pivot <= self.last < pivot * 1.03
        return (broke and ratio > 1.2), round(ratio, 2)

    def stage2_reclaim(self) -> bool:
        ema5  = self.close.ewm(span=5,  adjust=False).mean()
        ema10 = self.close.ewm(span=10, adjust=False).mean()
        sma20 = self.close.rolling(20).mean()
        sma50 = self.close.rolling(50).mean()
        stack    = ema5.iloc[-1] > ema10.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]
        above    = self.last > ema5.iloc[-1]
        rising   = sma20.iloc[-1] > sma20.iloc[-10]
        tight    = (self.last - ema5.iloc[-1]) / ema5.iloc[-1] < 0.05
        return bool(stack and above and rising and tight)

    def vcp_ready(self, pivot: float | None) -> tuple[bool, float]:
        if pivot is None:
            return False, 99.0
        dist = (pivot - self.last) / self.last * 100
        if not (0 <= dist < 5):
            return False, round(dist, 2)
        base = self.close.tail(min(100, len(self.close))).values
        return (_vcp_contractions(base) >= 3 and self.vol_dry_up() < 0.85), round(dist, 2)

    def weekly_trend(self) -> bool:
        weekly = self.df.resample("W").agg(
            {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
        ).dropna()
        if len(weekly) < 10:
            return False
        close = weekly["Close"].squeeze()
        ma10w = close.rolling(min(10, len(close))).mean()
        last  = float(close.iloc[-1])
        # Must be above 10-week MA and MA must be rising over last 4 weeks
        return bool(last > float(ma10w.iloc[-1]) and
                    float(ma10w.iloc[-1]) > float(ma10w.iloc[-min(4, len(ma10w)-1)]))

    def stage2_reclaim(self) -> bool:
        ema5  = self.close.ewm(span=5,  adjust=False).mean()
        ema10 = self.close.ewm(span=10, adjust=False).mean()
        sma20 = self.close.rolling(20).mean()
        sma50 = self.close.rolling(50).mean()
        stack  = ema5.iloc[-1] > ema10.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]
        above  = self.last > ema5.iloc[-1]
        rising = sma20.iloc[-1] > sma20.iloc[-10]
        # Removed tight constraint — allow up to 10% above EMA5
        return bool(stack and above and rising)

    def overextended(self) -> bool:
        ema20 = self.close.ewm(span=20).mean().iloc[-1]
        ema50 = self.close.ewm(span=50).mean().iloc[-1]
        return ((self.last - ema20) / ema20 * 100 > 15 or
                (self.last - ema50) / ema50 * 100 > 22)

    def _trending_momentum(self) -> bool:
        """Fallback: Stage2 stock above rising EMAs — no specific pattern required."""
        ema10 = self.close.ewm(span=10, adjust=False).mean()
        ema20 = self.close.ewm(span=20, adjust=False).mean()
        sma50 = self.close.rolling(50).mean()
        return bool(
            self.last > float(ema10.iloc[-1]) > float(ema20.iloc[-1]) > float(sma50.iloc[-1]) and
            float(ema10.iloc[-1]) > float(ema10.iloc[-5])
        )

    def classify(self) -> dict | None:
        if len(self.df) < 100 or self.overextended():
            return None
        if not self.weekly_trend():
            return None
        stage_name, _ = compute_stage(self.df)
        if stage_name == "Stage4":
            return None

        pivot                      = self.find_pivot_high()
        pp_flag,  pp_ratio         = self.pocket_pivot()
        br_flag,  br_vol           = self.breakout(pivot)
        s2_flag                    = self.stage2_reclaim()
        vcp_flag, dist_to_pivot    = self.vcp_ready(pivot)
        mom_flag                   = self._trending_momentum()

        if pp_flag and stage_name == "Stage2":
            sig, conv = "POCKET_PIVOT",       "High"
        elif br_flag:
            sig, conv = "BREAKOUT",           "High"
        elif s2_flag and vcp_flag:
            sig, conv = "STAGE2_RECLAIM",     "High"
        elif vcp_flag:
            sig, conv = "VCP_BREAKOUT_READY", "Medium"
        elif s2_flag:
            sig, conv = "STAGE2_RECLAIM",     "Medium"
        elif mom_flag and stage_name in ("Stage1", "Stage2"):
            sig, conv = "TRENDING_MOMENTUM",  "Medium"
        else:
            return None

        atr  = compute_atr(self.df)
        stop = self.last - 2 * atr
        return {
            "signal_type":       sig,
            "conviction":        conv,
            "stage":             stage_name,
            "pivot_high":        round(pivot, 2) if pivot else None,
            "dist_to_pivot_pct": dist_to_pivot,
            "pocket_pivot":      pp_flag,
            "pp_vol_ratio":      pp_ratio,
            "breakout_vol_ratio":br_vol,
            "stage2_reclaim":    s2_flag,
            "con_tightness_pct": round(self.tightness(), 2),
            "vol_dry_up_ratio":  round(self.vol_dry_up(), 2),
            "atr":               round(atr, 2),
            "atr_pct":           round(atr / self.last * 100, 2),
            "stop_atr":          round(stop, 2),
            "target1":           round(self.last + 2 * atr, 2),
            "target2":           round(self.last + 4 * atr, 2),
            "risk_reward":       round((2 * atr) / max(self.last - stop, 0.01), 2),
        }

# ── Scorecard ─────────────────────────────────────────────────────────────────

def build_scorecard(price_data: dict[str, pd.DataFrame],
                    rs_scores: dict[str, float]) -> list[dict]:
    scorecard = []
    for ticker, df in price_data.items():
        close  = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        last   = float(close.iloc[-1])

        adtv_cr = volume.tail(30).mean() * last / 1e7
        if adtv_cr < MIN_ADTV_CR:
            continue
        if compute_atr(df) / last * 100 > MAX_ATR_PCT:
            continue

        signal = EntrySignalEngine(df).classify()
        if signal is None:
            continue

        ma        = compute_5ma_status(df)
        week_ret  = (last / close.iloc[-5]  - 1) * 100 if len(close) >= 5  else 0
        month_ret = (last / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
        high52    = close.tail(252).max()

        scorecard.append({
            "ticker":            ticker,
            "last_price":        round(last, 2),
            "rs_score":          rs_scores.get(ticker, 50.0),
            "adtv_cr":           round(float(adtv_cr), 1),
            "pct_from_52w_high": round(float((last - high52) / high52 * 100), 1),
            "week_ret_pct":      round(float(week_ret), 2),
            "month_ret_pct":     round(float(month_ret), 2),
            **signal,
            **ma,
        })
    return scorecard


def get_market_regime(bench_df: pd.DataFrame | None) -> dict:
    if bench_df is None or len(bench_df) < 200:
        return {"regime": "unknown", "nifty500_price": 0,
                "nifty500_1m_ret_pct": 0, "nifty500_3m_ret_pct": 0,
                "nifty500_above_200ma": False}
    close  = bench_df["Close"].squeeze()
    last   = float(close.iloc[-1])
    ret3m  = (last / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else 0
    ret1m  = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0
    ma200  = close.rolling(200).mean().iloc[-1]
    return {
        "regime":               "bull" if ret3m > 5 else "bear" if ret3m < -5 else "sideways",
        "nifty500_price":       round(last, 2),
        "nifty500_1m_ret_pct":  round(float(ret1m), 1),
        "nifty500_3m_ret_pct":  round(float(ret3m), 1),
        "nifty500_above_200ma": bool(last > ma200),
    }

# ── Positions ─────────────────────────────────────────────────────────────────

def load_positions() -> list[dict]:
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return []


def save_positions(positions: list[dict]):
    os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def load_trade_log() -> list[dict]:
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE) as f:
            return json.load(f)
    return []


def append_trade_log(entries: list[dict]):
    log = load_trade_log()
    log.extend(entries)
    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

# ── Claude ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior Indian equity swing trader running a Minervini/O'Neil VCP + 5MA trailing stop system.
Your mandate: beat the Nifty 500 on a weekly rolling basis with tight risk control.

SIGNAL TYPES
============
POCKET_PIVOT        → Up-volume exceeds 10-day max down-volume. Institutional accumulation. HIGHEST priority.
BREAKOUT            → Price cleared prior pivot high on >1.2x avg volume. Classic O'Neil buy point.
STAGE2_RECLAIM      → 5EMA > 10EMA > 20SMA, all rising, price just above 5EMA.
VCP_BREAKOUT_READY  → Tight base, dry volume, within 3% of pivot. Stalk, don't chase yet.

SELECTION RULES
===============
1. Skip if adtv_cr < 5 or atr_pct > 4 (already filtered)
2. Strongly prefer rs_score > 80
3. VCP_BREAKOUT_READY: include only if dist_to_pivot_pct < 3, else watchlist
4. Cap at 5–10 picks. Hard limit: 10 total open positions.

POSITION SIZING
===============
Risk 1% of capital per trade. No single position > 20%. Sum ~100%.

5MA TRAILING STOP
=================
Initial stop: 2x ATR below entry. Trail at 5EMA once profitable.
2 consecutive closes below 5EMA = sell next open.

OUTPUT — valid JSON only, no markdown fences
============================================
{
  "regime_summary": "...",
  "market_stance": "aggressive|neutral|defensive",
  "picks": [
    {
      "ticker": "XXX.NS",
      "signal_type": "POCKET_PIVOT",
      "thesis": "Crisp: stage, RS, signal trigger.",
      "entry_zone": "₹X–₹Y",
      "initial_stop": 123.45,
      "stop_loss_pct": 7,
      "target1_pct": 12,
      "target2_pct": 22,
      "position_pct": 14,
      "conviction": "High|Medium",
      "rs_score": 85,
      "key_risk": "One sentence."
    }
  ],
  "watchlist": [{"ticker": "XXX.NS", "reason": "..."}],
  "sector_themes": ["..."],
  "portfolio_rationale": "2-3 sentences."
}"""


def call_claude(scorecard: list[dict], regime: dict,
                current_positions: list[dict] | None = None) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    holdings_block = ""
    if current_positions:
        holdings_block = (
            f"\nCURRENT HOLDINGS (already open — keep unless 5MA broken or materially worse than alternatives)\n"
            f"{'='*60}\n{json.dumps(current_positions, indent=2)}\n"
        )

    user_msg = (
        f"MARKET REGIME\n=============\n{json.dumps(regime, indent=2)}\n"
        f"{holdings_block}\n"
        f"ENTRY SIGNAL SCORECARD ({len(scorecard)} candidates)\n"
        f"{'='*60}\n{json.dumps(scorecard, indent=2)}\n\n"
        "Pick the best 5–10 swing trade entries for THIS WEEK. "
        "Include current holdings in your picks unless their 5MA is broken."
    )
    response = client.messages.create(
        model=MODEL, max_tokens=4096, temperature=0,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    return json.loads(raw)

# ── Pipeline ──────────────────────────────────────────────────────────────────

_SIGNAL_WEIGHT = {"POCKET_PIVOT": 100, "BREAKOUT": 90,
                  "STAGE2_RECLAIM": 70, "VCP_BREAKOUT_READY": 50,
                  "TRENDING_MOMENTUM": 40}


def run_scan() -> dict:
    """Full pipeline: universe → indicators → Claude. Returns structured dict."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    run_date    = datetime.date.today().isoformat()
    universe    = load_universe()
    positions   = load_positions()
    all_tickers = sorted(set(universe + [p["ticker"] for p in positions] + [BENCHMARK]))
    price_data  = fetch_prices(all_tickers, HISTORY_DAYS)
    bench_df    = price_data.pop(BENCHMARK, None)

    regime    = get_market_regime(bench_df)
    rs_scores = compute_rs_scores(price_data)
    scorecard = build_scorecard(price_data, rs_scores)

    if not scorecard:
        return {"run_date": run_date, "regime": regime, "error": "No entry signals found",
                "picks": [], "watchlist": [], "exits": [], "holds": [],
                "regime_summary": "", "market_stance": "neutral",
                "sector_themes": [], "portfolio_rationale": "",
                "scorecard_count": 0}

    for s in scorecard:
        s["_score"] = (_SIGNAL_WEIGHT[s["signal_type"]] * 0.40 +
                       s["rs_score"] * 0.30 +
                       (15 if s.get("stage2_reclaim") else 0) +
                       max(0, 20 - s["con_tightness_pct"]) * 0.75)
    scorecard.sort(key=lambda x: x["_score"], reverse=True)
    scorecard = scorecard[:TOP_N_TO_CLAUDE]
    for s in scorecard:
        del s["_score"]

    holdings_summary = [
        {"ticker": p["ticker"], "entry_price": p["entry_price"],
         "entry_date": p["entry_date"], "signal_type": p.get("signal_type", "")}
        for p in positions
    ]
    claude_result = call_claude(scorecard, regime, holdings_summary)

    # enrich picks with live CMP, qty, allocation
    enriched_picks = []
    for p in claude_result.get("picks", []):
        df        = price_data.get(p["ticker"])
        cmp       = round(float(df["Close"].squeeze().iloc[-1]), 2) if df is not None else None
        alloc_inr = round(CAPITAL * p["position_pct"] / 100)
        qty       = int(alloc_inr / cmp) if cmp else 0
        enriched_picks.append({**p, "cmp": cmp, "qty": qty,
                                "alloc_inr": alloc_inr,
                                "actual_inr": round(qty * cmp, 2) if cmp else 0})

    # rebalance diff — only exit on 5MA break, never just for "dropped from picks"
    new_tickers = {p["ticker"] for p in enriched_picks}
    exits, holds = [], []
    for pos in positions:
        df = price_data.get(pos["ticker"])
        if df is None:
            exits.append({**pos, "exit_reason": "no price data", "pnl_pct": 0, "r_multiple": 0})
            continue
        status = compute_5ma_status(df, entry_price=pos["entry_price"])
        if status["urgency"] == "exit":
            exits.append({**pos, **status, "exit_reason": "5MA EXIT"})
        else:
            holds.append({**pos, **status})

    result = {
        "run_date":           run_date,
        "regime":             regime,
        "scorecard_count":    len(scorecard),
        "picks":              enriched_picks,
        "watchlist":          claude_result.get("watchlist", []),
        "regime_summary":     claude_result.get("regime_summary", ""),
        "market_stance":      claude_result.get("market_stance", "neutral"),
        "sector_themes":      claude_result.get("sector_themes", []),
        "portfolio_rationale":claude_result.get("portfolio_rationale", ""),
        "exits":              exits,
        "holds":              holds,
    }

    out_path = f"{PICKS_DIR}/scan_{run_date}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def apply_rebalance(exit_tickers: list[str], approved_picks: list[dict],
                    exit_reasons: dict[str, str] | None = None) -> list[dict]:
    """Remove exits, add approved picks (up to MAX_POSITIONS). Saves and returns positions."""
    positions        = load_positions()
    run_date         = datetime.date.today().isoformat()
    exit_reasons     = exit_reasons or {}

    # fetch latest prices for exited positions to compute P&L
    exiting = [p for p in positions if p["ticker"] in exit_tickers]
    if exiting:
        price_data = fetch_prices([p["ticker"] for p in exiting], days=5)
        log_entries = []
        for pos in exiting:
            df        = price_data.get(pos["ticker"])
            exit_px   = round(float(df["Close"].squeeze().iloc[-1]), 2) if df is not None else pos["entry_price"]
            entry_px  = pos["entry_price"]
            qty       = pos["qty"]
            atr_stop  = pos.get("initial_stop") or entry_px
            risk      = max(entry_px - atr_stop, 0.01)
            holding   = (datetime.date.fromisoformat(run_date) -
                         datetime.date.fromisoformat(pos["entry_date"])).days
            log_entries.append({
                "ticker":       pos["ticker"],
                "signal_type":  pos.get("signal_type", ""),
                "entry_date":   pos["entry_date"],
                "exit_date":    run_date,
                "holding_days": holding,
                "entry_price":  entry_px,
                "exit_price":   exit_px,
                "qty":          qty,
                "pnl_inr":      round((exit_px - entry_px) * qty, 2),
                "pnl_pct":      round((exit_px - entry_px) / entry_px * 100, 2),
                "r_multiple":   round((exit_px - entry_px) / risk, 2),
                "exit_reason":  exit_reasons.get(pos["ticker"], "manual"),
                "position_pct": pos.get("position_pct", 0),
            })
        append_trade_log(log_entries)

    positions        = [p for p in positions if p["ticker"] not in exit_tickers]
    existing_tickers = {p["ticker"] for p in positions}

    for pick in approved_picks:
        if len(positions) >= MAX_POSITIONS:
            break
        ticker = pick["ticker"]
        if ticker not in existing_tickers:
            positions.append({
                "ticker":       ticker,
                "entry_price":  pick.get("cmp") or pick.get("entry_price", 0),
                "entry_date":   run_date,
                "initial_stop": pick.get("initial_stop"),
                "signal_type":  pick.get("signal_type"),
                "position_pct": pick.get("position_pct"),
                "qty":          pick.get("qty", 0),
                "alloc_inr":    pick.get("actual_inr", pick.get("alloc_inr", 0)),
            })
            existing_tickers.add(ticker)

    save_positions(positions)
    return positions
