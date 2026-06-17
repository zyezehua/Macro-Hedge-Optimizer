import numpy as np
import pytest

from mho.instruments.option import MarketContext
from mho.pricing.surface import VolSurface


@pytest.fixture
def market():
    return MarketContext(spot=100.0, r=0.043, q=0.013, multiplier=100.0)


@pytest.fixture
def surface():
    moneyness = np.array([0.70, 0.80, 0.90, 1.00, 1.10, 1.20])
    maturities = np.array([0.0833, 0.25, 0.50, 1.0])
    # Downside skew: lower moneyness carries higher IV, mild term structure.
    base = np.array([0.20, 0.21, 0.22, 0.23])
    rows = []
    for m in moneyness:
        skew = 0.30 * (1.0 - m)  # +3 vol pts per 10% below ATM
        rows.append(base + skew)
    vols = np.clip(np.array(rows), 0.05, None)
    return VolSurface(moneyness, maturities, vols)
