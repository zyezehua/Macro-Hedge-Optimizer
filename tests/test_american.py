"""Barone-Adesi-Whaley American pricing and its wiring through the strategy layer."""

import numpy as np
import pytest

from mho.instruments.option import MarketContext, OptionLeg
from mho.instruments.strategy import Strategy
from mho.pricing.american import american_price
from mho.pricing.black_scholes import bs_price
from mho.pricing.surface import VolSurface


def _flat_surface(vol=0.20):
    return VolSurface(np.array([0.70, 1.00, 1.30]), np.array([0.25, 0.5, 1.0]),
                      np.full((3, 3), vol))


def test_american_call_no_dividend_equals_european():
    # With q=0 (b=r) an American call is never exercised early ⇒ equals BS.
    a = american_price(100, 100, 0.05, 0.0, 0.25, 1.0, "call")
    e = bs_price(100, 100, 0.05, 0.0, 0.25, 1.0, "call")
    assert a == pytest.approx(e, abs=1e-9)


def test_american_premium_is_nonnegative_and_positive_for_put():
    # American value must be >= European; an ITM put with positive rates carries a strict premium.
    for S in (80, 90, 100, 110):
        a = american_price(S, 100, 0.05, 0.0, 0.20, 0.5, "put")
        e = bs_price(S, 100, 0.05, 0.0, 0.20, 0.5, "put")
        assert a >= e - 1e-9
    deep = american_price(80, 100, 0.06, 0.0, 0.20, 1.0, "put")
    deep_e = bs_price(80, 100, 0.06, 0.0, 0.20, 1.0, "put")
    assert deep > deep_e


def test_american_put_never_below_intrinsic():
    a = american_price(70, 100, 0.05, 0.0, 0.25, 0.5, "put")
    assert a >= (100 - 70) - 1e-9


def _crr(S, K, r, q, sig, T, kind, N=600):
    """Cox-Ross-Rubinstein American binomial tree (independent reference)."""
    import math
    dt = T / N
    u = math.exp(sig * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    payoff = lambda ST: max(ST - K, 0.0) if kind == "call" else max(K - ST, 0.0)
    vals = [payoff(S * u ** (N - i) * d ** i) for i in range(N + 1)]
    for step in range(N - 1, -1, -1):
        for i in range(step + 1):
            cont = disc * (p * vals[i] + (1 - p) * vals[i + 1])
            vals[i] = max(cont, payoff(S * u ** (step - i) * d ** i))
    return vals[0]


@pytest.mark.parametrize("S,K,r,q,sig,T,kind", [
    (100, 100, 0.08, 0.12, 0.20, 0.25, "put"),
    (100, 100, 0.05, 0.00, 0.20, 0.50, "put"),
    (100, 100, 0.05, 0.04, 0.25, 0.50, "call"),
])
def test_american_matches_binomial_tree(S, K, r, q, sig, T, kind):
    baw = american_price(S, K, r, q, sig, T, kind)
    tree = _crr(S, K, r, q, sig, T, kind)
    assert baw == pytest.approx(tree, abs=0.05)


def test_strategy_uses_american_when_flagged():
    surf = _flat_surface(0.25)
    strat = Strategy("Naked Put", [OptionLeg("put", 95.0, 0.5, +1)])
    euro_mkt = MarketContext(spot=100, r=0.05, q=0.0, multiplier=100.0, american=False)
    amer_mkt = MarketContext(spot=100, r=0.05, q=0.0, multiplier=100.0, american=True)
    assert strat.premium_per_unit(amer_mkt, surf) >= strat.premium_per_unit(euro_mkt, surf)
    # reshock preserves the american flag.
    assert amer_mkt.reshock(-0.1).american is True
