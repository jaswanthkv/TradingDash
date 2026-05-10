"""
options_scorecard.py — 35-parameter options environment scorecard.

Scores each parameter -2 to +2:
  Positive = environment favors SELLING premium (Iron Butterfly / Iron Condor)
  Negative = environment favors BUYING premium (Long Straddle)

Composite:  >= +20 : Iron Butterfly
  +10 to +19 : Iron Condor
  -9 to +9   : Stand Aside
  <= -10     : Long Straddle
"""
import datetime, logging
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from blackscholes import bs_gamma, bs_delta, implied_vol
from options import get_chain, get_vix, get_spot, atm_strike as _atm_fn, get_expiries

logger = logging.getLogger(__name__)

_RISK_FREE = 0.065   # 6.5% annualized
_NIFTY_LOT = 25


# ─── helpers ──────────────────────────────────────────────────────────────────

def _score_band(val: float, thresholds: list) -> int:
    for lo, hi, sc in thresholds:
        if lo <= val < hi:
            return sc
    return 0


def _fetch_vix_history(days: int = 252) -> pd.Series:
    try:
        df = yf.download("^INDIAVIX", period=f"{days + 10}d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        return df["Close"].dropna()
    except Exception as exc:
        logger.warning("VIX history fetch failed: %s", exc)
        return pd.Series(dtype=float)


def _fetch_nifty_history(days: int = 35) -> pd.Series:
    try:
        df = yf.download("^NSEI", period=f"{days + 5}d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        return df["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)


def _realized_vol_pct(closes: pd.Series, window: int = 20) -> float:
    if len(closes) < window + 1:
        return float("nan")
    rets = np.log(closes / closes.shift(1)).dropna()
    return round(float(rets.iloc[-window:].std() * np.sqrt(252) * 100), 2)


def _compute_iv(mid: float, spot: float, strike: int, T: float, opt: str) -> float | None:
    if mid <= 0 or T <= 0:
        return None
    return implied_vol(mid, spot, strike, T, _RISK_FREE, opt)


def _atm_row(chain: list, atm: int) -> dict | None:
    for r in chain:
        if r["strike"] == atm:
            return r
    return min(chain, key=lambda r: abs(r["strike"] - atm)) if chain else None


def _mid(leg: dict | None) -> float:
    if not leg:
        return 0.0
    bid = leg.get("bid", 0) or 0
    ask = leg.get("ask", 0) or 0
    ltp = leg.get("ltp", 0) or 0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return ltp


def _max_pain(chain: list) -> int:
    strikes = [r["strike"] for r in chain]
    if not strikes:
        return 0
    best, min_val = strikes[0], float("inf")
    for target in strikes:
        total = 0
        for r in chain:
            k = r["strike"]
            ce_oi = (r.get("CE") or {}).get("oi", 0)
            pe_oi = (r.get("PE") or {}).get("oi", 0)
            if k < target:
                total += (target - k) * ce_oi
            elif k > target:
                total += (k - target) * pe_oi
        if total < min_val:
            min_val, best = total, target
    return best


def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


# ─── Dealer Greeks aggregate ──────────────────────────────────────────────────

def _compute_dealer_greeks(chain: list, spot: float, T: float) -> dict:
    total = {"gex": 0.0, "dex": 0.0, "vanna": 0.0, "charm": 0.0, "vomma": 0.0, "theta": 0.0}
    for r in chain:
        k = r["strike"]
        for side, opt_type in [("CE", "C"), ("PE", "P")]:
            leg = r.get(side)
            if not leg:
                continue
            oi, lot = leg.get("oi", 0), leg.get("lot_size", _NIFTY_LOT)
            mid = _mid(leg)
            if oi == 0 or mid <= 0 or T <= 0:
                continue
            iv = _compute_iv(mid, spot, k, T, opt_type)
            if not iv or iv <= 0:
                continue
            d1, d2 = _d1d2(spot, k, T, _RISK_FREE, iv)
            nd1 = float(norm.pdf(d1))
            gamma  = nd1 / (spot * iv * np.sqrt(T))
            delta  = float(norm.cdf(d1) if opt_type == "C" else norm.cdf(d1) - 1)
            vega_  = spot * nd1 * np.sqrt(T)
            theta_ = -(spot * nd1 * iv) / (2 * np.sqrt(T))
            vanna  = (-d2 * nd1 / iv) if iv > 0 else 0.0
            charm  = (-nd1 * (2 * _RISK_FREE * T - d2 * iv * np.sqrt(T))
                      / (2 * T * iv * np.sqrt(T))) if T > 0 else 0.0
            vomma  = (vega_ * d1 * d2 / iv) if iv > 0 else 0.0
            mult   = oi * lot * (-1)   # dealers are net short
            total["gex"]   += gamma * mult
            total["dex"]   += delta * mult
            total["vanna"] += vanna * mult
            total["charm"] += charm * mult
            total["vomma"] += vomma * mult
            total["theta"] += theta_ * mult
    return {k: round(v, 6) for k, v in total.items()}


# ─── 35 Parameter functions ──────────────────────────────────────────────────

def p01_straddle_pct(chain, spot, atm, T):
    row = _atm_row(chain, atm)
    if not row:
        return {"score": 0, "value": None, "label": "Straddle %", "unit": "%"}
    pct = round((_mid(row.get("CE")) + _mid(row.get("PE"))) / spot * 100, 2) if spot else 0
    score = _score_band(pct, [(3.0, 99, 2), (2.2, 3.0, 1), (1.5, 2.2, 0), (0.8, 1.5, -1), (0, 0.8, -2)])
    return {"score": score, "value": pct, "label": "Straddle %", "unit": "%"}


def p02_iv_rank(vix_history, vix_now):
    if vix_history.empty or not vix_now:
        return {"score": 0, "value": None, "label": "IV Rank", "unit": "%ile"}
    lo, hi = float(vix_history.min()), float(vix_history.max())
    rank = round((vix_now - lo) / (hi - lo) * 100, 1) if hi > lo else 50.0
    score = _score_band(rank, [(80, 101, 2), (60, 80, 1), (40, 60, 0), (20, 40, -1), (0, 20, -2)])
    return {"score": score, "value": rank, "label": "IV Rank", "unit": "%ile"}


def p03_iv_rv_spread(vix_now, nifty_history):
    rv = _realized_vol_pct(nifty_history, 20)
    if np.isnan(rv) or not vix_now:
        return {"score": 0, "value": None, "label": "IV-RV Spread", "unit": "%"}
    spread = round(vix_now - rv, 2)
    score = _score_band(spread, [(8, 99, 2), (4, 8, 1), (0, 4, 0), (-4, 0, -1), (-99, -4, -2)])
    return {"score": score, "value": spread, "label": "IV-RV Spread", "unit": "%"}


def p04_vix_level(vix_now):
    if not vix_now:
        return {"score": 0, "value": None, "label": "VIX Level"}
    score = _score_band(vix_now, [(22, 99, 2), (16, 22, 1), (13, 16, 0), (10, 13, -1), (0, 10, -2)])
    return {"score": score, "value": round(vix_now, 2), "label": "VIX Level"}


def p05_vix_change(vix_history, vix_now):
    if len(vix_history) < 2 or not vix_now:
        return {"score": 0, "value": None, "label": "VIX 1d Change", "unit": "pts"}
    change = round(vix_now - float(vix_history.iloc[-1]), 2)
    score = _score_band(change, [(-99, -1.5, 2), (-1.5, -0.5, 1), (-0.5, 0.5, 0),
                                  (0.5, 1.5, -1), (1.5, 99, -2)])
    return {"score": score, "value": change, "label": "VIX 1d Change", "unit": "pts"}


def p06_vix_vs_ma(vix_history, vix_now):
    if len(vix_history) < 10 or not vix_now:
        return {"score": 0, "value": None, "label": "VIX vs 10d MA", "unit": "ratio"}
    ma = float(vix_history.iloc[-10:].mean())
    ratio = round(vix_now / ma, 3) if ma else 1.0
    score = _score_band(ratio, [(0, 0.90, 2), (0.90, 0.97, 1), (0.97, 1.03, 0),
                                 (1.03, 1.10, -1), (1.10, 99, -2)])
    return {"score": score, "value": ratio, "label": "VIX vs 10d MA", "unit": "ratio"}


def p07_atm_iv_skew(chain, spot, atm, T):
    row = _atm_row(chain, atm)
    if not row or T <= 0:
        return {"score": 0, "value": None, "label": "ATM IV Skew", "unit": "%"}
    ce_iv = _compute_iv(_mid(row.get("CE")), spot, atm, T, "C")
    pe_iv = _compute_iv(_mid(row.get("PE")), spot, atm, T, "P")
    if ce_iv is None or pe_iv is None:
        return {"score": 0, "value": None, "label": "ATM IV Skew", "unit": "%"}
    skew = round((pe_iv - ce_iv) * 100, 2)
    score = _score_band(abs(skew), [(0, 1.5, 2), (1.5, 3, 1), (3, 5, 0), (5, 8, -1), (8, 99, -2)])
    return {"score": score, "value": skew, "label": "ATM IV Skew (PE-CE)", "unit": "%"}


def p08_pcr_oi(chain):
    ce = sum((r.get("CE") or {}).get("oi", 0) for r in chain)
    pe = sum((r.get("PE") or {}).get("oi", 0) for r in chain)
    if ce == 0:
        return {"score": 0, "value": None, "label": "PCR (OI)"}
    pcr = round(pe / ce, 3)
    dev = abs(pcr - 1.0)
    score = _score_band(dev, [(0, 0.1, 2), (0.1, 0.25, 1), (0.25, 0.5, 0),
                               (0.5, 0.8, -1), (0.8, 99, -2)])
    return {"score": score, "value": pcr, "label": "PCR (OI)"}


def p09_pcr_volume(chain):
    ce = sum((r.get("CE") or {}).get("volume", 0) for r in chain)
    pe = sum((r.get("PE") or {}).get("volume", 0) for r in chain)
    if ce == 0:
        return {"score": 0, "value": None, "label": "PCR (Volume)"}
    pcr = round(pe / ce, 3)
    dev = abs(pcr - 1.0)
    score = _score_band(dev, [(0, 0.15, 2), (0.15, 0.35, 1), (0.35, 0.6, 0),
                               (0.6, 0.9, -1), (0.9, 99, -2)])
    return {"score": score, "value": pcr, "label": "PCR (Volume)"}


def p10_max_pain(chain, spot):
    mp = _max_pain(chain)
    if not mp:
        return {"score": 0, "value": None, "label": "Max Pain", "unit": "pts from spot"}
    dist = abs(spot - mp)
    score = _score_band(dist, [(0, 50, 2), (50, 100, 1), (100, 200, 0),
                                (200, 350, -1), (350, 99999, -2)])
    return {"score": score, "value": mp, "detail": round(dist, 0), "label": "Max Pain", "unit": "pts from spot"}


def p11_max_pain_distance(chain, spot):
    mp = _max_pain(chain)
    if not mp:
        return {"score": 0, "value": None, "label": "Max Pain Distance", "unit": "pts"}
    dist = round(mp - spot, 1)
    score = _score_band(abs(dist), [(0, 75, 2), (75, 150, 1), (150, 250, 0),
                                     (250, 400, -1), (400, 99999, -2)])
    return {"score": score, "value": dist, "label": "Max Pain Distance", "unit": "pts"}


def p12_ce_oi_wall(chain, atm):
    oi_map = {r["strike"]: (r.get("CE") or {}).get("oi", 0)
              for r in chain if r["strike"] > atm and (r.get("CE") or {}).get("oi", 0) > 0}
    if not oi_map:
        return {"score": 0, "value": None, "label": "CE OI Wall"}
    top = max(oi_map, key=oi_map.get)
    dist = top - atm
    score = _score_band(dist, [(200, 99999, 2), (100, 200, 1), (50, 100, 0), (0, 50, -1)])
    return {"score": score, "value": top, "detail": dist, "label": "CE OI Wall", "unit": f"+{dist}pts"}


def p13_pe_oi_wall(chain, atm):
    oi_map = {r["strike"]: (r.get("PE") or {}).get("oi", 0)
              for r in chain if r["strike"] < atm and (r.get("PE") or {}).get("oi", 0) > 0}
    if not oi_map:
        return {"score": 0, "value": None, "label": "PE OI Wall"}
    top = max(oi_map, key=oi_map.get)
    dist = atm - top
    score = _score_band(dist, [(200, 99999, 2), (100, 200, 1), (50, 100, 0), (0, 50, -1)])
    return {"score": score, "value": top, "detail": dist, "label": "PE OI Wall", "unit": f"-{dist}pts"}


def p14_atm_oi_concentration(chain, atm):
    row = _atm_row(chain, atm)
    if not row:
        return {"score": 0, "value": None, "label": "ATM OI Conc.", "unit": "% total"}
    total_oi = sum((r.get("CE") or {}).get("oi", 0) + (r.get("PE") or {}).get("oi", 0)
                   for r in chain)
    atm_oi = (row.get("CE") or {}).get("oi", 0) + (row.get("PE") or {}).get("oi", 0)
    pct = round(atm_oi / total_oi * 100, 1) if total_oi else 0
    score = _score_band(pct, [(20, 100, 2), (12, 20, 1), (7, 12, 0), (3, 7, -1), (0, 3, -2)])
    return {"score": score, "value": pct, "label": "ATM OI Conc.", "unit": "% total"}


def p15_oi_skew(chain, atm):
    ce_oi = pe_oi = 0
    for r in chain:
        if abs(r["strike"] - atm) <= 250:
            ce_oi += (r.get("CE") or {}).get("oi", 0)
            pe_oi += (r.get("PE") or {}).get("oi", 0)
    if ce_oi == 0:
        return {"score": 0, "value": None, "label": "OI Skew (±5 strikes)"}
    ratio = round(pe_oi / ce_oi, 3)
    dev = abs(ratio - 1.0)
    score = _score_band(dev, [(0, 0.1, 2), (0.1, 0.25, 1), (0.25, 0.5, 0),
                               (0.5, 0.8, -1), (0.8, 99, -2)])
    return {"score": score, "value": ratio, "label": "OI Skew (±5 strikes)"}


def p16_term_structure(chain1, chain2, spot, atm, T1, T2):
    if T1 <= 0 or T2 <= 0 or not chain1 or not chain2:
        return {"score": 0, "value": None, "label": "Term Structure", "unit": "%"}
    r1, r2 = _atm_row(chain1, atm), _atm_row(chain2, atm)
    if not r1 or not r2:
        return {"score": 0, "value": None, "label": "Term Structure", "unit": "%"}
    mid1 = (_mid(r1.get("CE")) + _mid(r1.get("PE"))) / 2
    mid2 = (_mid(r2.get("CE")) + _mid(r2.get("PE"))) / 2
    iv1 = _compute_iv(mid1, spot, atm, T1, "C")
    iv2 = _compute_iv(mid2, spot, atm, T2, "C")
    if iv1 is None or iv2 is None:
        return {"score": 0, "value": None, "label": "Term Structure", "unit": "%"}
    spread = round((iv1 - iv2) * 100, 2)
    score = _score_band(spread, [(3, 99, 2), (1, 3, 1), (-1, 1, 0), (-3, -1, -1), (-99, -3, -2)])
    return {"score": score, "value": spread, "label": "Term Structure (Near-Far IV)", "unit": "%"}


def p17_near_far_oi(chain1, chain2):
    oi1 = sum((r.get("CE") or {}).get("oi", 0) + (r.get("PE") or {}).get("oi", 0) for r in chain1)
    oi2 = sum((r.get("CE") or {}).get("oi", 0) + (r.get("PE") or {}).get("oi", 0) for r in chain2)
    if oi2 == 0:
        return {"score": 0, "value": None, "label": "Near/Far OI Ratio"}
    ratio = round(oi1 / oi2, 3)
    score = _score_band(ratio, [(2.0, 99, 2), (1.3, 2.0, 1), (0.8, 1.3, 0),
                                 (0.5, 0.8, -1), (0, 0.5, -2)])
    return {"score": score, "value": ratio, "label": "Near/Far OI Ratio"}


def p18_price_skew(chain, spot, atm, T):
    if T <= 0:
        return {"score": 0, "value": None, "label": "Price Skew (OTM IV)", "unit": "%"}
    pe_strike = atm - 100
    ce_strike = atm + 100
    pe_row = next((r for r in chain if r["strike"] == pe_strike), None)
    ce_row = next((r for r in chain if r["strike"] == ce_strike), None)
    if not pe_row or not ce_row:
        return {"score": 0, "value": None, "label": "Price Skew (OTM IV)", "unit": "%"}
    pe_iv = _compute_iv(_mid(pe_row.get("PE")), spot, pe_strike, T, "P")
    ce_iv = _compute_iv(_mid(ce_row.get("CE")), spot, ce_strike, T, "C")
    if pe_iv is None or ce_iv is None:
        return {"score": 0, "value": None, "label": "Price Skew (OTM IV)", "unit": "%"}
    skew = round((pe_iv - ce_iv) * 100, 2)
    score = _score_band(skew, [(1, 4, 2), (0, 1, 1), (4, 7, 0), (-2, 0, -1),
                                (7, 99, -1), (-99, -2, -2)])
    return {"score": score, "value": skew, "label": "Price Skew (OTM PE-CE IV)", "unit": "%"}


def p19_rupee_skew(chain, atm):
    pe_row = next((r for r in chain if r["strike"] == atm - 100), None)
    ce_row = next((r for r in chain if r["strike"] == atm + 100), None)
    if not pe_row or not ce_row:
        return {"score": 0, "value": None, "label": "Rupee Skew", "unit": "₹"}
    skew = round(_mid(pe_row.get("PE")) - _mid(ce_row.get("CE")), 2)
    score = _score_band(skew, [(5, 20, 2), (0, 5, 1), (20, 50, 0),
                                (-10, 0, -1), (50, 99999, -2), (-99999, -10, -2)])
    return {"score": score, "value": skew, "label": "Rupee Skew (₹)", "unit": "₹"}


def p20_spot_vs_distribution(spot, atm, chain, T):
    row = _atm_row(chain, atm)
    if not row or T <= 0:
        return {"score": 0, "value": None, "label": "Spot vs Distribution", "unit": "σ"}
    em = _mid(row.get("CE")) + _mid(row.get("PE"))
    if em == 0:
        return {"score": 0, "value": None, "label": "Spot vs Distribution", "unit": "σ"}
    sigmas = round(abs(spot - atm) / em, 2)
    score = _score_band(sigmas, [(0, 0.3, 2), (0.3, 0.6, 1), (0.6, 1.0, 0),
                                  (1.0, 1.5, -1), (1.5, 99, -2)])
    return {"score": score, "value": sigmas, "label": "Spot vs Distribution", "unit": "σ"}


def p21_net_gex(dg):
    val = dg.get("gex", 0)
    score = 2 if val > 0.05 else 1 if val > 0 else 0 if abs(val) < 0.01 else -1 if val > -0.05 else -2
    return {"score": score, "value": round(val, 4), "label": "Net GEX (Dealer)"}


def p22_net_dex(dg, spot):
    val = dg.get("dex", 0)
    rel = abs(val) / spot if spot else 0
    score = _score_band(rel, [(0, 0.01, 2), (0.01, 0.03, 1), (0.03, 0.06, 0),
                               (0.06, 0.10, -1), (0.10, 99, -2)])
    return {"score": score, "value": round(val, 2), "label": "Net DEX (Dealer)"}


def p23_vanna(dg):
    val = dg.get("vanna", 0)
    score = _score_band(abs(val), [(0, 0.05, 2), (0.05, 0.15, 1), (0.15, 0.30, 0),
                                    (0.30, 0.50, -1), (0.50, 99, -2)])
    return {"score": score, "value": round(val, 4), "label": "Net Vanna (Dealer)"}


def p24_charm(dg):
    val = dg.get("charm", 0)
    score = _score_band(abs(val), [(0, 0.001, 2), (0.001, 0.005, 1), (0.005, 0.01, 0),
                                    (0.01, 0.02, -1), (0.02, 99, -2)])
    return {"score": score, "value": round(val, 6), "label": "Net Charm (Dealer)"}


def p25_vomma(dg):
    val = dg.get("vomma", 0)
    score = _score_band(abs(val), [(0, 0.05, 2), (0.05, 0.15, 1), (0.15, 0.30, 0),
                                    (0.30, 0.60, -1), (0.60, 99, -2)])
    return {"score": score, "value": round(val, 4), "label": "Net Vomma (Dealer)"}


def p26_net_theta(dg):
    val = dg.get("theta", 0)
    score = _score_band(abs(val), [(1000, 99999, 2), (500, 1000, 1), (200, 500, 0),
                                    (50, 200, -1), (0, 50, -2)])
    return {"score": score, "value": round(val, 2), "label": "Net Market Theta (₹/day)"}


def p27_atm_gamma(chain, spot, atm, T):
    row = _atm_row(chain, atm)
    if not row or T <= 0:
        return {"score": 0, "value": None, "label": "ATM Gamma/lot"}
    iv = _compute_iv(_mid(row.get("CE")), spot, atm, T, "C")
    if not iv:
        return {"score": 0, "value": None, "label": "ATM Gamma/lot"}
    g = float(bs_gamma(spot, atm, T, _RISK_FREE, iv))
    g_lot = round(g * _NIFTY_LOT, 6)
    score = _score_band(g_lot, [(0, 0.0005, 2), (0.0005, 0.001, 1), (0.001, 0.002, 0),
                                 (0.002, 0.004, -1), (0.004, 99, -2)])
    return {"score": score, "value": g_lot, "label": "ATM Gamma/lot"}


def p28_gamma_imbalance(chain, spot, atm, T):
    row = _atm_row(chain, atm)
    if not row or T <= 0:
        return {"score": 0, "value": None, "label": "Gamma Imbalance (CE-PE)"}
    ce_iv = _compute_iv(_mid(row.get("CE")), spot, atm, T, "C")
    pe_iv = _compute_iv(_mid(row.get("PE")), spot, atm, T, "P")
    if ce_iv is None or pe_iv is None:
        return {"score": 0, "value": None, "label": "Gamma Imbalance (CE-PE)"}
    imb = round(float(bs_gamma(spot, atm, T, _RISK_FREE, ce_iv))
                - float(bs_gamma(spot, atm, T, _RISK_FREE, pe_iv)), 6)
    score = _score_band(abs(imb), [(0, 0.00005, 2), (0.00005, 0.0001, 1), (0.0001, 0.0002, 0),
                                    (0.0002, 0.0005, -1), (0.0005, 99, -2)])
    return {"score": score, "value": imb, "label": "Gamma Imbalance (CE-PE)"}


def p29_dte_effect(T):
    dte = round(T * 365, 1)
    score = _score_band(dte, [(0, 1, 2), (1, 3, 1), (3, 7, 0), (7, 15, -1), (15, 99999, -2)])
    return {"score": score, "value": dte, "label": "DTE", "unit": "days"}


def p30_volume_oi_ratio(chain):
    vol = sum((r.get("CE") or {}).get("volume", 0) + (r.get("PE") or {}).get("volume", 0) for r in chain)
    oi  = sum((r.get("CE") or {}).get("oi", 0)     + (r.get("PE") or {}).get("oi", 0)     for r in chain)
    if oi == 0:
        return {"score": 0, "value": None, "label": "Volume/OI Ratio"}
    ratio = round(vol / oi, 4)
    score = _score_band(ratio, [(0, 0.10, 2), (0.10, 0.20, 1), (0.20, 0.40, 0),
                                 (0.40, 0.70, -1), (0.70, 99, -2)])
    return {"score": score, "value": ratio, "label": "Volume/OI Ratio"}


def p31_em_vs_rv(chain, spot, atm, T, nifty_history):
    row = _atm_row(chain, atm)
    if not row or T <= 0:
        return {"score": 0, "value": None, "label": "Implied EM / RV EM", "unit": "ratio"}
    impl_em = (_mid(row.get("CE")) + _mid(row.get("PE"))) / spot * 100
    rv = _realized_vol_pct(nifty_history, 20)
    if np.isnan(rv) or rv == 0:
        return {"score": 0, "value": None, "label": "Implied EM / RV EM", "unit": "ratio"}
    rv_em = rv / np.sqrt(252) * np.sqrt(T * 365) * 100
    ratio = round(impl_em / rv_em, 2) if rv_em > 0 else 1.0
    score = _score_band(ratio, [(1.4, 99, 2), (1.2, 1.4, 1), (0.9, 1.2, 0),
                                 (0.7, 0.9, -1), (0, 0.7, -2)])
    return {"score": score, "value": ratio, "label": "Implied EM / RV EM", "unit": "ratio"}


def p32_ce_pe_premium_ratio(chain, atm):
    row = _atm_row(chain, atm)
    if not row:
        return {"score": 0, "value": None, "label": "CE/PE Premium Ratio"}
    ce_m, pe_m = _mid(row.get("CE")), _mid(row.get("PE"))
    if pe_m == 0:
        return {"score": 0, "value": None, "label": "CE/PE Premium Ratio"}
    ratio = round(ce_m / pe_m, 3)
    dev = abs(ratio - 1.0)
    score = _score_band(dev, [(0, 0.1, 2), (0.1, 0.25, 1), (0.25, 0.5, 0),
                               (0.5, 0.8, -1), (0.8, 99, -2)])
    return {"score": score, "value": ratio, "label": "CE/PE Premium Ratio"}


def p33_nifty_trend(nifty_history):
    if len(nifty_history) < 21:
        return {"score": 0, "value": None, "label": "Nifty 5d/20d MA Ratio"}
    ratio = round(float(nifty_history.iloc[-5:].mean()) / float(nifty_history.iloc[-20:].mean()), 4)
    dev = abs(ratio - 1.0)
    score = _score_band(dev, [(0, 0.005, 2), (0.005, 0.01, 1), (0.01, 0.02, 0),
                               (0.02, 0.035, -1), (0.035, 99, -2)])
    return {"score": score, "value": ratio, "label": "Nifty 5d/20d MA Ratio"}


def p34_beta_regime(nifty_history, vix_history):
    if len(nifty_history) < 21 or len(vix_history) < 21:
        return {"score": 0, "value": None, "label": "Nifty-VIX Corr (20d)"}
    nr = nifty_history.pct_change().dropna().iloc[-20:].values
    vr = vix_history.pct_change().dropna().iloc[-20:].values
    n  = min(len(nr), len(vr))
    corr = round(float(np.corrcoef(nr[-n:], vr[-n:])[0, 1]), 3)
    score = _score_band(corr, [(-1, -0.6, 2), (-0.6, -0.3, 1), (-0.3, 0.1, 0),
                                (0.1, 0.4, -1), (0.4, 1.1, -2)])
    return {"score": score, "value": corr, "label": "Nifty-VIX Corr (20d)"}


def p35_atm_straddle_rupees(chain, atm):
    row = _atm_row(chain, atm)
    if not row:
        return {"score": 0, "value": None, "label": "ATM Straddle ₹/lot", "unit": "₹"}
    lot   = (row.get("CE") or {}).get("lot_size", _NIFTY_LOT) or _NIFTY_LOT
    total = round((_mid(row.get("CE")) + _mid(row.get("PE"))) * lot, 0)
    score = _score_band(total, [(3000, 99999, 2), (2000, 3000, 1), (1200, 2000, 0),
                                 (600, 1200, -1), (0, 600, -2)])
    return {"score": score, "value": total, "label": "ATM Straddle ₹/lot", "unit": "₹"}


# ─── strategy label ───────────────────────────────────────────────────────────

def _strategy(composite: int) -> str:
    if composite >= 20:
        return "Iron Butterfly"
    if composite >= 10:
        return "Iron Condor"
    if composite >= -9:
        return "Stand Aside"
    return "Long Straddle"


def _env_label(composite: int) -> str:
    if composite >= 20:
        return "SELL PREMIUM — tight strikes"
    if composite >= 10:
        return "SELL PREMIUM — wider strikes"
    if composite >= -9:
        return "NEUTRAL — wait for clarity"
    return "BUY VOLATILITY — avoid short premium"


# ─── main entry ───────────────────────────────────────────────────────────────

def run_scorecard(symbol: str = "NIFTY", expiry1: str = None, expiry2: str = None) -> dict:
    expiries = get_expiries(symbol)
    if not expiry1 and expiries:
        expiry1 = expiries[0]
    if not expiry2 and len(expiries) >= 2:
        expiry2 = expiries[1]

    today = datetime.date.today()

    def _T(exp_str):
        if not exp_str:
            return 0.0
        return max(0, (datetime.date.fromisoformat(exp_str) - today).days) / 365

    T1, T2 = _T(expiry1), _T(expiry2)

    vix_now     = get_vix()
    spot        = get_spot(symbol)
    atm         = _atm_fn(spot, symbol)
    vix_history = _fetch_vix_history(252)
    nifty_hist  = _fetch_nifty_history(35)

    chain1 = get_chain(symbol, expiry1)["chain"] if expiry1 else []
    chain2 = get_chain(symbol, expiry2)["chain"] if expiry2 else []

    dg = _compute_dealer_greeks(chain1, spot, T1) if chain1 else {}

    params = [
        p01_straddle_pct(chain1, spot, atm, T1),
        p02_iv_rank(vix_history, vix_now),
        p03_iv_rv_spread(vix_now, nifty_hist),
        p04_vix_level(vix_now),
        p05_vix_change(vix_history, vix_now),
        p06_vix_vs_ma(vix_history, vix_now),
        p07_atm_iv_skew(chain1, spot, atm, T1),
        p08_pcr_oi(chain1),
        p09_pcr_volume(chain1),
        p10_max_pain(chain1, spot),
        p11_max_pain_distance(chain1, spot),
        p12_ce_oi_wall(chain1, atm),
        p13_pe_oi_wall(chain1, atm),
        p14_atm_oi_concentration(chain1, atm),
        p15_oi_skew(chain1, atm),
        p16_term_structure(chain1, chain2, spot, atm, T1, T2),
        p17_near_far_oi(chain1, chain2),
        p18_price_skew(chain1, spot, atm, T1),
        p19_rupee_skew(chain1, atm),
        p20_spot_vs_distribution(spot, atm, chain1, T1),
        p21_net_gex(dg),
        p22_net_dex(dg, spot),
        p23_vanna(dg),
        p24_charm(dg),
        p25_vomma(dg),
        p26_net_theta(dg),
        p27_atm_gamma(chain1, spot, atm, T1),
        p28_gamma_imbalance(chain1, spot, atm, T1),
        p29_dte_effect(T1),
        p30_volume_oi_ratio(chain1),
        p31_em_vs_rv(chain1, spot, atm, T1, nifty_hist),
        p32_ce_pe_premium_ratio(chain1, atm),
        p33_nifty_trend(nifty_hist),
        p34_beta_regime(nifty_hist, vix_history),
        p35_atm_straddle_rupees(chain1, atm),
    ]

    for i, p in enumerate(params, 1):
        p["param"] = i

    composite = sum(p["score"] for p in params)

    return {
        "symbol":    symbol,
        "spot":      round(spot, 2),
        "atm":       atm,
        "vix":       round(vix_now, 2),
        "expiry1":   expiry1,
        "expiry2":   expiry2,
        "dte1":      round(T1 * 365, 1),
        "dte2":      round(T2 * 365, 1) if T2 else None,
        "composite": composite,
        "strategy":  _strategy(composite),
        "label":     _env_label(composite),
        "bullish":   sum(1 for p in params if p["score"] > 0),
        "neutral":   sum(1 for p in params if p["score"] == 0),
        "bearish":   sum(1 for p in params if p["score"] < 0),
        "sections": [
            {"name": "Volatility Context",        "params": params[0:7]},
            {"name": "OI Market Structure",        "params": params[7:15]},
            {"name": "Cross-Expiry Dynamics",      "params": params[15:20]},
            {"name": "Dealer Greeks & Flow",       "params": params[20:28]},
            {"name": "Pin Risk & Expiry Effects",  "params": params[28:35]},
        ],
        "params":    params,
        "timestamp": datetime.datetime.now().isoformat(),
    }
