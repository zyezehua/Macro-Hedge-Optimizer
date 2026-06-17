"""Black-Scholes-Merton pricing for European options with a continuous dividend yield.

Used as a PoC pricing engine. SPX (cash-settled index) options are European and priced
exactly; ETF options (SPY/QQQ/HYG) are American, so BS is a documented approximation that
ignores the early-exercise premium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

OptionType = str  # "call" or "put"

_SQRT_EPS = 1e-12


def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> tuple[float, float]:
    sqrtT = math.sqrt(max(T, _SQRT_EPS))
    vol = max(sigma, _SQRT_EPS) * sqrtT
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol
    return d1, d2


def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, kind: OptionType) -> float:
    """Black-Scholes-Merton price (per 1 share). Handles T->0 as intrinsic value."""
    kind = kind.lower()
    if T <= _SQRT_EPS or sigma <= _SQRT_EPS:
        # Degenerate: return discounted intrinsic on the forward.
        fwd = S * math.exp((r - q) * T)
        intrinsic = max(fwd - K, 0.0) if kind == "call" else max(K - fwd, 0.0)
        return math.exp(-r * T) * intrinsic
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if kind == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif kind == "put":
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    raise ValueError(f"Unknown option kind: {kind!r}")


@dataclass(frozen=True)
class Greeks:
    """Per-share greeks. theta is per calendar year; vega per 1.00 (=100 vol pts) of sigma."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    @property
    def vega_per_volpt(self) -> float:
        """Vega expressed per 1 vol point (0.01 of sigma)."""
        return self.vega / 100.0


def bs_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, kind: OptionType) -> Greeks:
    """Analytic BSM greeks (per 1 share)."""
    kind = kind.lower()
    if T <= _SQRT_EPS or sigma <= _SQRT_EPS:
        return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    sqrtT = math.sqrt(T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    pdf = norm.pdf(d1)

    gamma = df_q * pdf / (S * sigma * sqrtT)
    vega = S * df_q * pdf * sqrtT  # per 1.00 change in sigma
    if kind == "call":
        delta = df_q * norm.cdf(d1)
        theta = (
            -S * df_q * pdf * sigma / (2 * sqrtT)
            - r * K * df_r * norm.cdf(d2)
            + q * S * df_q * norm.cdf(d1)
        )
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = (
            -S * df_q * pdf * sigma / (2 * sqrtT)
            + r * K * df_r * norm.cdf(-d2)
            - q * S * df_q * norm.cdf(-d1)
        )
        rho = -K * T * df_r * norm.cdf(-d2)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
