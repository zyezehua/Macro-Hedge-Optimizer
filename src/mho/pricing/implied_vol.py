"""Two-way conversion between option price (per share) and implied volatility.

Premiums are quoted sometimes as price/share and sometimes as an IV; this module powers
both directions so the rest of the system can work in a single canonical form (sigma).
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from .black_scholes import OptionType, bs_price

_SIG_LOW = 1e-4
_SIG_HIGH = 5.0


def price_to_iv(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    kind: OptionType,
) -> float:
    """Invert Black-Scholes for implied volatility via Brent, bisection fallback.

    Raises ValueError if the price is outside no-arbitrage bounds.
    """
    kind = kind.lower()
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    fwd = S * math.exp((r - q) * T)
    intrinsic = df_r * (max(fwd - K, 0.0) if kind == "call" else max(K - fwd, 0.0))
    upper = S * df_q if kind == "call" else K * df_r  # max option value

    if price < intrinsic - 1e-8:
        raise ValueError(f"Price {price} below intrinsic {intrinsic:.6f}; no valid IV.")
    if price > upper + 1e-8:
        raise ValueError(f"Price {price} above upper bound {upper:.6f}; no valid IV.")

    def objective(sigma: float) -> float:
        return bs_price(S, K, r, q, sigma, T, kind) - price

    f_low, f_high = objective(_SIG_LOW), objective(_SIG_HIGH)
    if f_low * f_high > 0:
        # Solution at a boundary (price ~ intrinsic or ~ upper bound).
        return _SIG_LOW if abs(f_low) < abs(f_high) else _SIG_HIGH
    return brentq(objective, _SIG_LOW, _SIG_HIGH, xtol=1e-8, maxiter=200)


def iv_to_price(
    sigma: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    kind: OptionType,
) -> float:
    """Forward direction: IV -> price/share. Thin wrapper for API symmetry."""
    return bs_price(S, K, r, q, sigma, T, kind)
