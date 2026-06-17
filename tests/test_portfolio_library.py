"""Historical stress library + combined cross-asset (portfolio) hedge optimizer."""

import numpy as np
import pytest

from mho.instruments.option import MarketContext
from mho.optimize.portfolio import HedgeInstrument, optimize_portfolio
from mho.pricing.surface import VolSurface
from mho.scenarios.library import (
    HISTORICAL_STRESSES,
    asset_class,
    build_macro_scenario,
)
from mho.scenarios.macro import InstrumentShock, MacroScenario


def _surface(base=0.20):
    m = np.array([0.70, 0.85, 1.00, 1.15])
    t = np.array([0.25, 0.5, 1.0])
    skew = np.array([base + 0.10, base + 0.04, base, base - 0.01])[:, None] * np.ones((1, 3))
    return VolSurface(m, t, skew)


def _spx():
    return HedgeInstrument("SPX", MarketContext(5400, 0.043, 0.013, 100.0), _surface(0.20), "put_spread")


def _hyg():
    return HedgeInstrument("HYG", MarketContext(78, 0.043, 0.055, 100.0, american=True),
                           _surface(0.18), "naked_put")


# --- library -----------------------------------------------------------------

def test_asset_class_routing():
    assert asset_class("HYG") == "credit"
    assert asset_class("JNK") == "credit"
    assert asset_class("SPX") == "equity"
    assert asset_class("QQQ") == "equity"


def test_build_macro_scenario_assigns_shocks_by_class():
    tpl = HISTORICAL_STRESSES["gfc_2008"]
    m = build_macro_scenario(tpl, 25_000_000, ["SPX", "HYG"], probability=0.04)
    assert m.shocks["SPX"] == tpl.equity
    assert m.shocks["HYG"] == tpl.credit
    assert m.shocks["SPX"].spot_shock < m.shocks["HYG"].spot_shock < 0  # equity falls more in '08
    assert m.target_payoff == 25_000_000
    assert m.probability == 0.04


def test_macro_for_instrument_derives_single_asset_scenario():
    m = MacroScenario("x", 10_000_000, {"SPX": InstrumentShock(-0.20, 0.08, "skew_twist", 0.1)},
                      timing_years=0.25, probability=0.1)
    s = m.for_instrument("SPX")
    assert (s.spot_shock, s.vol_shock, s.vol_mode, s.twist) == (-0.20, 0.08, "skew_twist", 0.1)
    assert s.timing_years == 0.25 and s.probability == 0.1
    assert s.target_payoff == 0.0  # target lives at the portfolio level
    # An unshocked instrument is treated as flat.
    assert m.for_instrument("HYG").spot_shock == 0.0


# --- portfolio optimizer -----------------------------------------------------

def _scenarios():
    return [
        build_macro_scenario(HISTORICAL_STRESSES["gfc_2008"], 30_000_000, ["SPX", "HYG"], probability=0.04),
        build_macro_scenario(HISTORICAL_STRESSES["rates_2022"], 12_000_000, ["SPX", "HYG"], probability=0.15),
    ]


def test_portfolio_feasible_meets_every_target():
    res = optimize_portfolio([_spx(), _hyg()], _scenarios(), maturity=0.5)
    assert res.feasible
    assert np.isfinite(res.total_cost) and res.total_cost > 0
    for m in _scenarios():
        assert res.payoff_in(m.name) >= m.target_payoff - 1e-6


def test_adding_an_instrument_does_not_increase_cost():
    # The LP can always set an instrument's allocation to zero, so a richer instrument set is
    # never more expensive than a single-instrument hedge.
    scns = _scenarios()
    spx_only = optimize_portfolio([_spx()], scns, maturity=0.5)
    both = optimize_portfolio([_spx(), _hyg()], scns, maturity=0.5)
    assert spx_only.feasible and both.feasible
    assert both.total_cost <= spx_only.total_cost + 1.0


def test_portfolio_infeasible_when_target_unreachable():
    # A melt-UP that lands exactly at expiry: the put legs settle worthless, so no number of
    # contracts can pay a positive target ⇒ the LP is infeasible.
    melt_up = [MacroScenario("melt_up", 5_000_000, {"SPX": InstrumentShock(0.20)}, timing_years=0.5)]
    res = optimize_portfolio([_spx()], melt_up, maturity=0.5)
    assert not res.feasible
