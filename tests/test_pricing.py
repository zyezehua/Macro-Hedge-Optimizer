import math

import numpy as np
import pytest

from mho.pricing.black_scholes import bs_greeks, bs_price
from mho.pricing.implied_vol import iv_to_price, price_to_iv
from mho.pricing.surface import VolSurface


def test_bs_known_value():
    # Hull-style benchmark: S=K=100, r=5%, q=0, sigma=20%, T=1.
    call = bs_price(100, 100, 0.05, 0.0, 0.20, 1.0, "call")
    put = bs_price(100, 100, 0.05, 0.0, 0.20, 1.0, "put")
    assert call == pytest.approx(10.4506, abs=1e-3)
    assert put == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    S, K, r, q, sig, T = 100, 95, 0.04, 0.015, 0.25, 0.75
    call = bs_price(S, K, r, q, sig, T, "call")
    put = bs_price(S, K, r, q, sig, T, "put")
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_iv_roundtrip():
    S, K, r, q, T = 420.0, 400.0, 0.043, 0.013, 0.5
    for sig in (0.10, 0.22, 0.45, 0.85):
        for kind in ("call", "put"):
            px = iv_to_price(sig, S, K, r, q, T, kind)
            recovered = price_to_iv(px, S, K, r, q, T, kind)
            assert recovered == pytest.approx(sig, abs=1e-5)


def test_iv_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        price_to_iv(-1.0, 100, 100, 0.04, 0.0, 1.0, "put")


def test_greeks_signs():
    g_call = bs_greeks(100, 100, 0.04, 0.0, 0.2, 1.0, "call")
    g_put = bs_greeks(100, 100, 0.04, 0.0, 0.2, 1.0, "put")
    assert 0 < g_call.delta < 1
    assert -1 < g_put.delta < 0
    assert g_call.gamma > 0 and g_put.gamma > 0
    assert g_call.vega > 0 and g_put.vega > 0
    # vega is independent of call/put
    assert g_call.vega == pytest.approx(g_put.vega, abs=1e-9)


def _demo_surface():
    moneyness = np.array([0.80, 0.90, 1.00, 1.10])
    maturities = np.array([0.0833, 0.25, 0.50, 1.0])
    vols = np.array(
        [
            [0.32, 0.30, 0.28, 0.26],  # 80% moneyness (downside skew rich)
            [0.26, 0.25, 0.24, 0.23],
            [0.20, 0.20, 0.20, 0.20],  # ATM
            [0.18, 0.19, 0.20, 0.21],
        ]
    )
    return VolSurface(moneyness, maturities, vols)


def test_surface_interpolation_on_node():
    surf = _demo_surface()
    assert surf.iv(1.00, 0.50) == pytest.approx(0.20, abs=1e-12)
    assert surf.iv(0.80, 0.0833) == pytest.approx(0.32, abs=1e-12)


def test_surface_bilinear_midpoint():
    surf = _demo_surface()
    # midpoint between (0.90,0.25)=0.25 and (1.00,0.25)=0.20 at moneyness 0.95
    assert surf.iv(0.95, 0.25) == pytest.approx(0.225, abs=1e-9)


def test_surface_flat_extrapolation():
    surf = _demo_surface()
    assert surf.iv(0.50, 0.50) == pytest.approx(surf.iv(0.80, 0.50))
    assert surf.iv(1.00, 5.0) == pytest.approx(surf.iv(1.00, 1.0))


def test_surface_parallel_shock():
    surf = _demo_surface()
    shocked = surf.shocked(0.05, mode="parallel")
    assert shocked.iv(1.00, 0.50) == pytest.approx(0.25, abs=1e-12)


def test_surface_skew_twist_raises_downside_more():
    surf = _demo_surface()
    sh = surf.shocked(0.05, mode="skew_twist", twist_per_moneyness=0.10)
    down = sh.iv(0.80, 0.50) - surf.iv(0.80, 0.50)
    atm = sh.iv(1.00, 0.50) - surf.iv(1.00, 0.50)
    assert down > atm
