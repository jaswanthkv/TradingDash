"""
options_seller.py — quant weekly option-selling strike suggester.

Pipeline:
  1. Spot + nearest weekly expiry + option chain (live, from Kite).
  2. Expected move from the ATM straddle. The ATM straddle ≈ 0.7979·S·σ·√t, so
     one standard deviation (to expiry) in points = straddle / 0.7979.
  3. Implied vs realized volatility — when implied > realized you're being paid a
     variance-risk premium, i.e. it's a favourable time to SELL premium.
  4. Probability of profit (POP) per strike from a normal model on S_T.
  5. Pick sell strikes by a target POP set by the risk level (not a fixed mult).

⚠️ Selling options carries large/undefined risk. Decision-support only, not advice.
"""
import math
from datetime import date, timedelta
# kite_auth is imported lazily inside the live functions so the backtest
# (which only needs bhavcopy/yfinance) works without kiteconnect installed.

UNDERLYINGS = {
    "NIFTY":     {"index": "NSE:NIFTY 50",          "name": "NIFTY",     "step": 50,  "lot": 75, "yf": "^NSEI"},
    "BANKNIFTY": {"index": "NSE:NIFTY BANK",        "name": "BANKNIFTY", "step": 100, "lot": 30, "yf": "^NSEBANK"},
    "FINNIFTY":  {"index": "NSE:NIFTY FIN SERVICE", "name": "FINNIFTY",  "step": 50,  "lot": 65, "yf": "NIFTY_FIN_SERVICE.NS"},
}
# Target probability the sold strike expires worthless (out-of-the-money).
RISK_POP = {"conservative": 0.90, "balanced": 0.84, "aggressive": 0.75}
_STRADDLE_TO_SD = 0.7979   # ATM straddle / S / (σ√t)

_nfo_cache = {"date": None, "instruments": None}


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs(S: float, K: float, sigma: float, t: float, call: bool) -> float:
    """Black-Scholes price (r=0). Used only to estimate historical premiums."""
    if sigma <= 0 or t <= 0:
        return max(S - K, 0) if call else max(K - S, 0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return (S * _ncdf(d1) - K * _ncdf(d2)) if call else (K * _ncdf(-d2) - S * _ncdf(-d1))


# SD distance for each risk level's strikes (z for the target POP).
RISK_Z = {"conservative": 1.28, "balanced": 1.0, "aggressive": 0.67}


def _trading_days(start: date, end: date) -> int:
    """Weekdays remaining from the day after `start` through `end` (≥1)."""
    n, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return max(n, 1)


def _nfo_instruments(kite):
    today = date.today().isoformat()
    if _nfo_cache["date"] == today and _nfo_cache["instruments"] is not None:
        return _nfo_cache["instruments"]
    inst = kite.instruments("NFO")
    _nfo_cache.update(date=today, instruments=inst)
    return inst


def _realized(underlying: str):
    """(daily_log_return_std, avg_weekly_range_pct) from recent daily history,
    or (None, None) if unavailable."""
    try:
        import yfinance as yf, numpy as np, pandas as pd
        sym = UNDERLYINGS[underlying]["yf"]
        df = yf.download(sym, period="4mo", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = df["Close"].dropna()
        if len(c) < 15:
            return None, None
        rets = np.log(c / c.shift(1)).dropna()
        sd = float(rets.tail(20).std())
        wk = df.resample("W").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
        wk_range = float(((wk["High"] - wk["Low"]) / wk["Close"]).tail(8).mean())
        return sd, wk_range
    except Exception:
        return None, None


def suggest(underlying: str = "NIFTY", risk: str = "balanced") -> dict:
    import kite_auth
    cfg = UNDERLYINGS.get(underlying, UNDERLYINGS["NIFTY"])
    target_pop = RISK_POP.get(risk, 0.84)
    kite = kite_auth.get_kite()

    spot = kite.ltp([cfg["index"]])[cfg["index"]]["last_price"]
    step = cfg["step"]

    opts = [i for i in _nfo_instruments(kite)
            if i["name"] == cfg["name"] and i["instrument_type"] in ("CE", "PE")]
    today    = date.today()
    expiries = sorted({i["expiry"] for i in opts if i["expiry"] and i["expiry"] >= today})
    if not expiries:
        raise RuntimeError(f"No live {underlying} option expiries found")
    expiry = expiries[0]
    chain  = [i for i in opts if i["expiry"] == expiry]
    td     = _trading_days(today, expiry)

    ce_strikes = sorted({i["strike"] for i in chain if i["instrument_type"] == "CE"})
    pe_strikes = sorted({i["strike"] for i in chain if i["instrument_type"] == "PE"})

    def _inst(strike, typ):
        for i in chain:
            if i["instrument_type"] == typ and abs(i["strike"] - strike) < 1e-6:
                return i
        return None

    def _ltp_map(insts):
        syms = [f"NFO:{i['tradingsymbol']}" for i in insts if i]
        q = kite.ltp(syms) if syms else {}
        return {i["strike"]: q.get(f"NFO:{i['tradingsymbol']}", {}).get("last_price")
                for i in insts if i}

    # ── Expected move from the ATM straddle ────────────────────────────────────
    atm     = round(spot / step) * step
    atm_px  = _ltp_map([_inst(atm, "CE"), _inst(atm, "PE")])
    straddle = sum(v for v in atm_px.values() if v) or 0.0
    if straddle <= 0:
        raise RuntimeError("Could not read ATM option prices (market closed?)")
    sd_pts   = straddle / _STRADDLE_TO_SD          # one standard deviation, in points
    implied_move = straddle                        # ~0.8 SD; the market's expected move
    iv_annual    = (sd_pts / spot) / math.sqrt(td) * math.sqrt(252)

    rv_daily, wk_range_pct = _realized(underlying)
    rv_pts    = rv_daily * math.sqrt(td) * spot if rv_daily else None
    rv_annual = rv_daily * math.sqrt(252) if rv_daily else None
    # Compare implied vs realized over the SAME horizon (to expiry) — annualizing a
    # near-expiry straddle is misleading. vrp>0 ⇒ market prices a bigger move than
    # realized lately ⇒ richer premium ⇒ more favourable to sell.
    vrp = (implied_move / rv_pts - 1) if rv_pts else None

    def pop_call(k): return _ncdf((k - spot) / sd_pts)   # P(S_T < strike)
    def pop_put(k):  return _ncdf((spot - k) / sd_pts)

    # ── Select sell strikes by target POP ──────────────────────────────────────
    ce_strike = next((s for s in ce_strikes if s > spot and pop_call(s) >= target_pop),
                     ce_strikes[-1] if ce_strikes else atm)
    pe_strike = next((s for s in reversed(pe_strikes) if s < spot and pop_put(s) >= target_pop),
                     pe_strikes[0] if pe_strikes else atm)

    sel = _ltp_map([_inst(ce_strike, "CE"), _inst(pe_strike, "PE")])
    ce_prem = sel.get(ce_strike)
    pe_prem = sel.get(pe_strike)
    total   = (ce_prem or 0) + (pe_prem or 0)
    lot     = cfg["lot"]

    # ── Candidate ladder (a few OTM strikes each side, with POP + premium) ──────
    def _candidates(strikes, side):
        out, picked = [], [s for s in strikes]
        if side == "CE":
            picked = [s for s in strikes if s >= ce_strike - 2 * step and s <= ce_strike + 3 * step]
        else:
            picked = [s for s in strikes if s <= pe_strike + 2 * step and s >= pe_strike - 3 * step]
        pxs = _ltp_map([_inst(s, side) for s in picked])
        for s in picked:
            pop = pop_call(s) if side == "CE" else pop_put(s)
            out.append({"strike": s, "premium": round(pxs[s], 2) if pxs.get(s) is not None else None,
                        "pop_pct": round(pop * 100, 1),
                        "distance_pct": round((s - spot) / spot * 100, 2)})
        return out

    # Entry edge: sell only when the market prices a bigger move than realized.
    if vrp is None:
        verdict, vreason = "unknown", "Realized vol unavailable."
    elif vrp > 0.10:
        verdict, vreason = "sell", "Premium is rich — market pricing a bigger move than realized. Edge favours selling."
    elif vrp < -0.10:
        verdict, vreason = "skip", "Premium is cheap vs recent moves — thin edge. Consider skipping this week."
    else:
        verdict, vreason = "caution", "Premium is fair — only a small edge."

    return {
        "underlying":     underlying,
        "spot":           round(spot, 2),
        "expiry":         expiry.isoformat(),
        "days_to_expiry": (expiry - today).days,
        "trading_days":   td,
        "risk":           risk,
        "target_pop":     round(target_pop * 100),
        "lot_size":       lot,
        "verdict":        verdict,
        "verdict_reason": vreason,
        "analysis": {
            "implied_move":      round(implied_move, 1),
            "implied_move_pct":  round(implied_move / spot * 100, 2),
            "sd1":               round(sd_pts, 1),
            "sd1_low":           round(spot - sd_pts), "sd1_high": round(spot + sd_pts),
            "sd2_low":           round(spot - 2 * sd_pts), "sd2_high": round(spot + 2 * sd_pts),
            "iv_annual_pct":     round(iv_annual * 100, 1),
            "rv_annual_pct":     round(rv_annual * 100, 1) if rv_annual else None,
            "realized_move":     round(rv_pts, 1) if rv_pts else None,
            "realized_move_pct": round(rv_pts / spot * 100, 2) if rv_pts else None,
            "vrp_pct":           round(vrp * 100, 1) if vrp is not None else None,
            "sell_signal":       ("rich" if vrp is not None and vrp > 0.10 else
                                  "cheap" if vrp is not None and vrp < -0.10 else
                                  "fair" if vrp is not None else "unknown"),
            "recent_weekly_range_pct": round(wk_range_pct * 100, 2) if wk_range_pct else None,
        },
        "call": {
            "strike": ce_strike, "tradingsymbol": (_inst(ce_strike, "CE") or {}).get("tradingsymbol"),
            "premium": round(ce_prem, 2) if ce_prem is not None else None,
            "distance_pct": round((ce_strike - spot) / spot * 100, 2),
            "pop_pct": round(pop_call(ce_strike) * 100, 1),
            "breakeven": round(ce_strike + total, 2),
        },
        "put": {
            "strike": pe_strike, "tradingsymbol": (_inst(pe_strike, "PE") or {}).get("tradingsymbol"),
            "premium": round(pe_prem, 2) if pe_prem is not None else None,
            "distance_pct": round((pe_strike - spot) / spot * 100, 2),
            "pop_pct": round(pop_put(pe_strike) * 100, 1),
            "breakeven": round(pe_strike - total, 2),
        },
        "range_low":       pe_strike,
        "range_high":      ce_strike,
        "total_premium":   round(total, 2),
        "premium_per_lot": round(total * lot, 2),
        "call_candidates": _candidates(ce_strikes, "CE"),
        "put_candidates":  _candidates(pe_strikes, "PE"),
    }


# ── Position monitoring + adjustment ────────────────────────────────────────────

def _ctx(kite, name, expiry):
    """Live context for an underlying+expiry: spot, step, chain, 1σ (points)."""
    cfg  = UNDERLYINGS[name]
    spot = kite.ltp([cfg["index"]])[cfg["index"]]["last_price"]
    step = cfg["step"]
    chain = [i for i in _nfo_instruments(kite)
             if i["name"] == name and i["expiry"] == expiry
             and i["instrument_type"] in ("CE", "PE")]
    atm = round(spot / step) * step
    def _f(s, t): return next((i for i in chain if i["instrument_type"] == t
                               and abs(i["strike"] - s) < 1e-6), None)
    syms = [f"NFO:{i['tradingsymbol']}" for i in [_f(atm, "CE"), _f(atm, "PE")] if i]
    q = kite.ltp(syms) if syms else {}
    straddle = sum((q.get(s, {}).get("last_price") or 0) for s in syms)
    sd = straddle / _STRADDLE_TO_SD if straddle > 0 else None
    return cfg, spot, step, chain, sd


def _ltp_for(kite, insts):
    syms = [f"NFO:{i['tradingsymbol']}" for i in insts if i]
    q = kite.ltp(syms) if syms else {}
    return {i["tradingsymbol"]: q.get(f"NFO:{i['tradingsymbol']}", {}).get("last_price")
            for i in insts if i}


def positions(target_pop: float = 0.84) -> dict:
    """Open short option legs from Kite with live health (safe/watch/tested/breached)
    and, for any threatened leg, the best adjustment (roll) with concrete numbers."""
    import kite_auth
    kite = kite_auth.get_kite()
    inst = _nfo_instruments(kite)
    by_sym = {i["tradingsymbol"]: i for i in inst}

    shorts = []
    for p in kite.positions().get("net", []):
        if p.get("quantity", 0) >= 0:
            continue                       # only short legs
        m = by_sym.get(p["tradingsymbol"])
        if m and m["instrument_type"] in ("CE", "PE"):
            shorts.append((m, p))

    groups: dict = {}
    for m, p in shorts:
        groups.setdefault((m["name"], m["expiry"]), []).append((m, p))

    out = []
    for (name, expiry), legs in groups.items():
        grp = {"underlying": name, "expiry": expiry.isoformat(),
               "known": name in UNDERLYINGS, "legs": [], "adjustment": None}
        if name not in UNDERLYINGS:
            for m, p in legs:
                grp["legs"].append({"tradingsymbol": m["tradingsymbol"], "type": m["instrument_type"],
                                    "strike": m["strike"], "qty": p["quantity"],
                                    "ltp": p.get("last_price"), "pnl": round(p.get("pnl", 0), 2),
                                    "status": "unknown"})
            out.append(grp); continue

        cfg, spot, step, chain, sd = _ctx(kite, name, expiry)
        grp.update(spot=round(spot, 2), sd=round(sd, 1) if sd else None,
                   lot_size=cfg["lot"])
        ce_strikes = sorted({i["strike"] for i in chain if i["instrument_type"] == "CE"})
        pe_strikes = sorted({i["strike"] for i in chain if i["instrument_type"] == "PE"})

        def _pick(side):
            ks = ce_strikes if side == "CE" else pe_strikes
            if side == "CE":
                return next((s for s in ks if s > spot and _ncdf((s - spot) / sd) >= target_pop), ks[-1] if ks else None)
            return next((s for s in reversed(ks) if s < spot and _ncdf((spot - s) / sd) >= target_pop), ks[0] if ks else None)

        cur = _ltp_for(kite, [m for m, _ in legs])
        leg_objs = []
        for m, p in legs:
            typ, K = m["instrument_type"], m["strike"]
            buf = (K - spot) / sd if sd and typ == "CE" else ((spot - K) / sd if sd else None)
            if buf is not None:
                status = ("breached" if buf <= 0 else "tested" if buf < 0.5 else
                          "watch" if buf < 1.0 else "safe")
            else:
                # No σ available — fall back to raw OTM distance.
                dp = ((K - spot) if typ == "CE" else (spot - K)) / spot
                status = ("breached" if dp <= 0 else "tested" if dp < 0.015 else
                          "watch" if dp < 0.03 else "safe")
            leg_objs.append({
                "tradingsymbol": m["tradingsymbol"], "type": typ, "strike": K,
                "qty": p["quantity"], "avg": round(p.get("average_price", 0), 2),
                "ltp": round(cur.get(m["tradingsymbol"]) or p.get("last_price") or 0, 2),
                "pnl": round(p.get("pnl", 0), 2),
                "distance_pct": round((K - spot) / spot * 100, 2),
                "buffer_sd": round(buf, 2) if buf is not None else None,
                "pop_pct": round(_ncdf(buf) * 100, 1) if buf is not None else None,
                "status": status,
            })
        grp["legs"] = leg_objs
        worst = min(leg_objs, key=lambda x: x["buffer_sd"] if x["buffer_sd"] is not None else 9)
        grp["status"] = worst["status"]

        # Build an adjustment for a threatened leg.
        if sd and worst["status"] in ("watch", "tested", "breached"):
            t_side = worst["type"]
            u_side = "PE" if t_side == "CE" else "CE"
            t_leg  = worst
            u_leg  = next((l for l in leg_objs if l["type"] == u_side), None)
            lot    = cfg["lot"]
            actions = []

            # (1) Recenter: roll the UNTESTED side in toward price for extra credit.
            if u_leg:
                new_u = _pick(u_side)
                u_inst = next((i for i in chain if i["instrument_type"] == u_side and abs(i["strike"] - new_u) < 1e-6), None)
                px = _ltp_for(kite, [u_inst]) if u_inst else {}
                new_prem = px.get(u_inst["tradingsymbol"]) if u_inst else None
                if new_prem is not None and new_u != u_leg["strike"]:
                    credit = (new_prem - u_leg["ltp"])
                    actions.append({
                        "kind": "recenter",
                        "title": f"Roll {u_side} in to {int(new_u)} (recenter)",
                        "detail": f"Buy back {int(u_leg['strike'])} {u_side}, sell {int(new_u)} {u_side}",
                        "net": round(credit * lot, 2), "net_type": "credit" if credit >= 0 else "debit",
                        "new_range": f"{int(new_u) if t_side=='PE' else int(t_leg['strike'])}–{int(t_leg['strike']) if t_side=='CE' else int(new_u)}",
                        "why": "Keeps the trade a credit and shifts your range toward price, cutting directional risk.",
                    })

            # (2) Roll the TESTED side out & away (same expiry; further OTM at target POP).
            new_t = _pick(t_side)
            t_inst = next((i for i in chain if i["instrument_type"] == t_side and abs(i["strike"] - new_t) < 1e-6), None)
            px2 = _ltp_for(kite, [t_inst]) if t_inst else {}
            new_t_prem = px2.get(t_inst["tradingsymbol"]) if t_inst else None
            if new_t_prem is not None and new_t != t_leg["strike"]:
                net = (new_t_prem - t_leg["ltp"])
                actions.append({
                    "kind": "roll_away",
                    "title": f"Roll {t_side} away to {int(new_t)}",
                    "detail": f"Buy back {int(t_leg['strike'])} {t_side}, sell {int(new_t)} {t_side}",
                    "net": round(net * lot, 2), "net_type": "credit" if net >= 0 else "debit",
                    "why": "Moves the threatened strike further from price (usually for a small debit).",
                })

            # (3) Always offer: close the tested leg to cap risk.
            actions.append({
                "kind": "close",
                "title": f"Close the {t_side} (cap the risk)",
                "detail": f"Buy back {int(t_leg['strike'])} {t_side}",
                "net": round(-t_leg["ltp"] * cfg["lot"], 2), "net_type": "debit",
                "why": "Simplest defense — take the loss on the tested leg and keep the safe one.",
            })

            recommended = "recenter" if worst["status"] in ("watch", "tested") and u_leg else \
                          ("roll_away" if worst["status"] != "breached" else "close")
            grp["adjustment"] = {"tested_side": t_side, "recommended": recommended, "actions": actions}

        out.append(grp)

    return {"positions": out}


# ── Backtest (approximate) ──────────────────────────────────────────────────────

def backtest(underlying: str = "NIFTY", risk: str = "balanced", weeks: int = 5,
             stop_x: float = 0.0) -> dict:
    """Approximate weekly short-strangle backtest over the last `weeks` weeks.

    For each ~1-week block we sell strikes z·σ away (σ = realized vol from the
    20 days BEFORE entry — no look-ahead), estimate the credit with Black-Scholes,
    then settle against the actual index close at week's end. Premiums are modelled
    (real expired-weekly prices aren't available), so treat P&L as indicative —
    real selling usually collects a bit more (IV > realized).

    If stop_x > 0, a fixed defensive rule is applied: the strangle is marked to a
    Black-Scholes model each day and CLOSED the moment its value reaches stop_x ×
    the credit received (e.g. stop_x=2 ⇒ exit when down ~1× credit)."""
    cfg = UNDERLYINGS.get(underlying, UNDERLYINGS["NIFTY"])
    z   = RISK_Z.get(risk, 1.0)
    step, lot = cfg["step"], cfg["lot"]

    import yfinance as yf, numpy as np, pandas as pd
    df = yf.download(cfg["yf"], period="9mo", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    c = df["Close"].dropna()
    closes, dates = c.values, c.index
    rets = np.log(c / c.shift(1)).values
    n = len(closes)

    W = 5                     # trading days ≈ 1 week
    rows = []
    i_end = n - 1
    while len(rows) < weeks and i_end - W >= 21:
        i0 = i_end - W
        S0, S1 = float(closes[i0]), float(closes[i_end])
        sd_d = float(np.nanstd(rets[i0 - 20:i0]))
        if sd_d <= 0:
            i_end -= W; continue
        sig_ann = sd_d * math.sqrt(252)
        sig_p   = sd_d * math.sqrt(W)
        t       = W / 252.0
        CE = math.ceil(S0 * math.exp(z * sig_p) / step) * step
        PE = math.floor(S0 * math.exp(-z * sig_p) / step) * step
        prem = _bs(S0, CE, sig_ann, t, True) + _bs(S0, PE, sig_ann, t, False)

        # Fixed defensive rule: close early if modelled value hits stop_x × credit.
        stopped, exit_i, exit_spot = False, i_end, S1
        if stop_x and stop_x > 0:
            for j in range(i0 + 1, i_end):
                sp = float(closes[j])
                tr = (i_end - j) / 252.0
                val = _bs(sp, CE, sig_ann, tr, True) + _bs(sp, PE, sig_ann, tr, False)
                if val >= stop_x * prem:
                    stopped, exit_i, exit_spot = True, j, sp
                    pnl = prem - val
                    break
        if not stopped:
            payoff = max(S1 - CE, 0) + max(PE - S1, 0)
            pnl = prem - payoff

        rows.append({
            "entry_date": str(dates[i0].date()), "exit_date": str(dates[exit_i].date()),
            "spot_start": round(S0, 1), "spot_end": round(exit_spot, 1),
            "call": CE, "put": PE,
            "premium": round(prem, 1),
            "in_range": (not stopped) and max(S1 - CE, 0) + max(PE - S1, 0) <= 1e-6,
            "stopped": stopped,
            "pnl_pts": round(pnl, 1), "pnl": round(pnl * lot), "win": pnl > 0,
        })
        i_end -= W
    rows.reverse()

    return _summarize(rows, underlying, risk, lot, stop_x, source="model")


def _summarize(rows, underlying, risk, lot, stop_x, source):
    wins  = sum(1 for r in rows if r["win"])
    total = sum(r["pnl"] for r in rows)
    worst = min((r["pnl"] for r in rows), default=0)
    return {
        "underlying": underlying, "risk": risk, "weeks": len(rows), "lot_size": lot,
        "stop_x": stop_x, "source": source,
        "win_rate": round(wins / len(rows) * 100, 1) if rows else 0,
        "total_pnl": round(total), "avg_pnl": round(total / len(rows)) if rows else 0,
        "worst_pnl": round(worst), "stops": sum(1 for r in rows if r.get("stopped")),
        "rows": rows,
        "note": ("Real NSE settlement premiums — entry & exit prices from the F&O "
                 "bhavcopy; strikes chosen from the actual ATM straddle each week."
                 if source == "real" else
                 "Approximate — premium modelled with Black-Scholes on realized vol; "
                 "settled against actual index closes. Indicative, not exact."),
    }


def backtest_real(underlying: str = "NIFTY", risk: str = "balanced",
                  weeks: int = 5, stop_x: float = 0.0) -> dict:
    """Weekly short-strangle backtest on REAL NSE bhavcopy data.

    Each week: sell the front weekly's strikes chosen at the target POP (σ from the
    actual ATM straddle), using real settlement premiums; settle at the actual
    expiry-day settlement. If stop_x>0, exit early when the position's real daily
    value reaches stop_x × the credit."""
    import bhavcopy as bc

    cfg = UNDERLYINGS.get(underlying, UNDERLYINGS["NIFTY"])
    lot, target_pop = cfg["lot"], RISK_POP.get(risk, 0.84)

    df = bc.load(underlying)
    if df.empty:
        return {"error": "No bhavcopy data — run the downloader first.", "rows": []}
    df = df.copy()
    df["px"] = df["settle"].where(df["settle"] > 0, df["close"])
    s = df.set_index(["date", "expiry", "opt_type", "strike"])["px"].sort_index()
    tdays = sorted(df["date"].unique())

    # On expiry day NSE writes the UNDERLYING final settlement into every option's
    # `settle` field — so capture it here and value the legs by intrinsic at exit.
    fin = df[df["date"] == df["expiry"]]
    exp_settle = {E: float(g["settle"].iloc[0]) for E, g in fin.groupby("expiry")}

    # Front weekly expiry for each trading day → one trade per expiry (enter the
    # day a new front weekly appears, hold to that expiry).
    front = {}
    for d, exps in df.groupby("date")["expiry"]:
        fe = min((e for e in exps.unique() if e >= d), default=None)
        if fe is not None:
            front[d] = fe
    trades, prev = [], None
    for d in tdays:
        fe = front.get(d)
        if fe and fe != prev:
            trades.append((d, fe)); prev = fe
    trades = trades[-(weeks + 1):]              # newest weeks (skip the still-open one)

    rows = []
    for entry, E in trades:
        if E >= tdays[-1]:                      # still-open week → skip
            continue
        try:
            sub = s.loc[(entry, E)]
            ce, pe = sub.loc["CE"], sub.loc["PE"]
        except KeyError:
            continue
        common = ce.index.intersection(pe.index)
        if len(common) < 3:
            continue
        atm = (ce[common] - pe[common]).abs().idxmin()
        spot = float(atm + ce[atm] - pe[atm])      # put-call parity
        straddle = float(ce[atm] + pe[atm])
        if straddle <= 0:
            continue
        sd = straddle / _STRADDLE_TO_SD
        ce_k = min([k for k in ce.index if k > spot and _ncdf((k - spot) / sd) >= target_pop],
                   default=float(ce.index.max()))
        pe_k = max([k for k in pe.index if k < spot and _ncdf((spot - k) / sd) >= target_pop],
                   default=float(pe.index.min()))
        prem = float(ce.get(ce_k, 0) + pe.get(pe_k, 0))
        if prem <= 0:
            continue

        # Defensive stop on the real daily path.
        stopped, exit_d, exit_val = False, E, None
        week_days = [t for t in tdays if entry < t < E]
        if stop_x and stop_x > 0:
            for t in week_days:
                try:
                    val = float(s.loc[(t, E, "CE", ce_k)] + s.loc[(t, E, "PE", pe_k)])
                except KeyError:
                    continue
                if val >= stop_x * prem:
                    stopped, exit_d, exit_val = True, t, val
                    break
        spot_end = None
        if not stopped:
            # Settle at expiry = intrinsic vs the underlying's final settlement.
            spot_end = exp_settle.get(E)
            if spot_end is None:
                continue
            exit_val = max(spot_end - ce_k, 0.0) + max(pe_k - spot_end, 0.0)
        else:
            # Stopped mid-week: spot via put-call parity on that day's chain.
            try:
                ex = s.loc[(exit_d, E)]
                xce, xpe = ex.loc["CE"], ex.loc["PE"]
                xc = xce.index.intersection(xpe.index)
                if len(xc):
                    a2 = (xce[xc] - xpe[xc]).abs().idxmin()
                    spot_end = round(float(a2 + xce[a2] - xpe[a2]), 1)
            except Exception:
                pass
        pnl = prem - exit_val
        rows.append({
            "entry_date": str(entry), "exit_date": str(exit_d),
            "spot_start": round(spot, 1), "spot_end": spot_end,
            "call": int(ce_k), "put": int(pe_k),
            "premium": round(prem, 1),
            "in_range": (not stopped) and exit_val <= 1e-6,
            "stopped": stopped,
            "pnl_pts": round(pnl, 1), "pnl": round(pnl * lot), "win": pnl > 0,
        })

    return _summarize(rows, underlying, risk, lot, stop_x, source="real")
