"""
blackscholes.py — Black-Scholes pricing and Greeks for Nifty options.

All functions work with annualised inputs:
  S     = spot price
  K     = strike price
  T     = time to expiry in years (e.g. 30 days = 30/365)
  r     = risk-free rate (annualised, e.g. 0.065 for 6.5%)
  sigma = implied volatility (annualised, e.g. 0.15 for 15%)
  opt   = 'C' for call, 'P' for put
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

_LOT_SIZE = 75   # Nifty standard lot size


def _d1d2(S: float, K: float, T: float, r: float, sigma: float):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str = "C") -> float:
    if T <= 0:
        return max(0.0, S - K) if opt == "C" else max(0.0, K - S)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if opt == "C":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt: str = "C") -> float:
    if T <= 0:
        if opt == "C": return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(norm.cdf(d1) if opt == "C" else norm.cdf(d1) - 1.0)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, opt: str = "C") -> float:
    """Daily theta (per calendar day)."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if opt == "C":
        return float((term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
    return float((term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega per 1% change in IV."""
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * np.sqrt(T) / 100.0)


def implied_vol(price: float, S: float, K: float, T: float, r: float, opt: str = "C") -> float | None:
    """Implied volatility via Brent's method. Returns None if unsolvable."""
    if T <= 0:
        return None
    intrinsic = max(0.0, S - K) if opt == "C" else max(0.0, K - S)
    if price <= intrinsic + 1e-6:
        return None
    try:
        return float(brentq(
            lambda sig: bs_price(S, K, T, r, sig, opt) - price,
            1e-4, 10.0, xtol=1e-7, maxiter=200
        ))
    except Exception:
        return None


def all_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    opt: str = "C", lots: int = 1, lot_size: int = _LOT_SIZE
) -> dict:
    """Full Greeks dict for `lots` short contracts."""
    mult   = lots * lot_size
    delta  = bs_delta(S, K, T, r, sigma, opt)
    gamma  = bs_gamma(S, K, T, r, sigma)
    theta  = bs_theta(S, K, T, r, sigma, opt)
    vega   = bs_vega(S, K, T, r, sigma)
    price  = bs_price(S, K, T, r, sigma, opt)
    return {
        "price":           round(price,  2),
        "delta":           round(delta,  4),
        "gamma":           round(gamma,  6),
        "theta":           round(theta,  2),
        "vega":            round(vega,   4),
        "delta_exposure":  round(delta * mult, 2),   # total delta in index points
        "gamma_exposure":  round(gamma * mult, 4),
        "theta_per_day":   round(theta * mult, 2),
        "vega_per_pct":    round(vega  * mult, 2),
    }
