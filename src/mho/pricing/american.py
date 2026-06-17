"""American option pricing via the Barone-Adesi-Whaley (1987) quadratic approximation.

ETF options (SPY/QQQ/HYG/IWM) are physically settled and American, so they carry an
early-exercise premium over the European Black-Scholes value. BS understates the value of an
American put (and of an American call on a dividend payer). BAW is a fast closed-form
approximation that adds this premium analytically; it is exact in the European limit and
typically within a few cents of a binomial tree for the maturities/moneyness used here.

Convention matches `black_scholes.bs_price`: cost of carry b = r - q, premium per 1 share.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from .black_scholes import OptionType, bs_price

_EPS = 1e-12
_MAX_ITER = 60


def _gbs(S: float, K: float, r: float, b: float, sigma: float, T: float, kind: str) -> float:
    """Generalized Black-Scholes with cost of carry b (= r - q)."""
    return bs_price(S, K, r, r - b, sigma, T, kind)


def _seed_call(S0_unused: float, K: float, r: float, b: float, sigma: float, T: float) -> float:
    n = 2 * b / sigma**2
    m = 2 * r / sigma**2
    q2u = (-(n - 1) + math.sqrt((n - 1) ** 2 + 4 * m)) / 2
    su = K / (1 - 1 / q2u)
    h2 = -(b * T + 2 * sigma * math.sqrt(T)) * K / (su - K)
    return K + (su - K) * (1 - math.exp(h2))


def _critical_call(K: float, r: float, b: float, sigma: float, T: float) -> float:
    """Solve for the critical asset price above which an American call is exercised."""
    sqrtT = math.sqrt(T)
    n = 2 * b / sigma**2
    K2 = 2 * r / (sigma**2 * (1 - math.exp(-r * T)))
    Q2 = (-(n - 1) + math.sqrt((n - 1) ** 2 + 4 * K2)) / 2
    Si = _seed_call(0.0, K, r, b, sigma, T)
    for _ in range(_MAX_ITER):
        d1 = (math.log(Si / K) + (b + sigma**2 / 2) * T) / (sigma * sqrtT)
        edrt = math.exp((b - r) * T)
        lhs = Si - K
        rhs = _gbs(Si, K, r, b, sigma, T, "call") + (1 - edrt * norm.cdf(d1)) * Si / Q2
        bi = edrt * norm.cdf(d1) * (1 - 1 / Q2) + (1 - edrt * norm.pdf(d1) / (sigma * sqrtT)) / Q2
        if abs(lhs - rhs) / K <= 1e-6 or abs(1 - bi) < _EPS:
            break
        Si = (K + rhs - bi * Si) / (1 - bi)
        if Si <= 0:
            Si = K
    return Si


def _critical_put(K: float, r: float, b: float, sigma: float, T: float) -> float:
    """Solve for the critical asset price below which an American put is exercised."""
    sqrtT = math.sqrt(T)
    n = 2 * b / sigma**2
    m = 2 * r / sigma**2
    q1u = (-(n - 1) - math.sqrt((n - 1) ** 2 + 4 * m)) / 2
    su = K / (1 - 1 / q1u)
    h1 = (b * T - 2 * sigma * sqrtT) * K / (K - su)
    Si = su + (K - su) * math.exp(h1)
    K2 = 2 * r / (sigma**2 * (1 - math.exp(-r * T)))
    Q1 = (-(n - 1) - math.sqrt((n - 1) ** 2 + 4 * K2)) / 2
    for _ in range(_MAX_ITER):
        d1 = (math.log(Si / K) + (b + sigma**2 / 2) * T) / (sigma * sqrtT)
        edrt = math.exp((b - r) * T)
        lhs = K - Si
        rhs = _gbs(Si, K, r, b, sigma, T, "put") - (1 - edrt * norm.cdf(-d1)) * Si / Q1
        bi = -edrt * norm.cdf(-d1) * (1 - 1 / Q1) - (1 + edrt * norm.pdf(-d1) / (sigma * sqrtT)) / Q1
        if abs(lhs - rhs) / K <= 1e-6 or abs(1 + bi) < _EPS:
            break
        Si = (K - rhs + bi * Si) / (1 + bi)
        if Si <= 0:
            Si = K / 2
    return Si


def american_price(S: float, K: float, r: float, q: float, sigma: float, T: float, kind: OptionType) -> float:
    """Barone-Adesi-Whaley American price (per 1 share). Falls back to BS in degenerate cases."""
    kind = kind.lower()
    if T <= _EPS or sigma <= _EPS:
        return bs_price(S, K, r, q, sigma, T, kind)
    b = r - q
    euro = bs_price(S, K, r, q, sigma, T, kind)
    sqrtT = math.sqrt(T)

    if kind == "call":
        if b >= r:  # no dividends ⇒ never optimal to exercise an American call early.
            return euro
        Sk = _critical_call(K, r, b, sigma, T)
        if S >= Sk:
            return S - K
        n = 2 * b / sigma**2
        K2 = 2 * r / (sigma**2 * (1 - math.exp(-r * T)))
        Q2 = (-(n - 1) + math.sqrt((n - 1) ** 2 + 4 * K2)) / 2
        d1 = (math.log(Sk / K) + (b + sigma**2 / 2) * T) / (sigma * sqrtT)
        a2 = (Sk / Q2) * (1 - math.exp((b - r) * T) * norm.cdf(d1))
        return max(euro, euro + a2 * (S / Sk) ** Q2)

    if kind == "put":
        Sk = _critical_put(K, r, b, sigma, T)
        if S <= Sk:
            return K - S
        n = 2 * b / sigma**2
        K2 = 2 * r / (sigma**2 * (1 - math.exp(-r * T)))
        Q1 = (-(n - 1) - math.sqrt((n - 1) ** 2 + 4 * K2)) / 2
        d1 = (math.log(Sk / K) + (b + sigma**2 / 2) * T) / (sigma * sqrtT)
        a1 = -(Sk / Q1) * (1 - math.exp((b - r) * T) * norm.cdf(-d1))
        return max(euro, euro + a1 * (S / Sk) ** Q1)

    raise ValueError(f"Unknown option kind: {kind!r}")
