import math

import pytest

from mho.crossasset.beta import CrossAssetMap, translate_scenarios
from mho.rolling.roller import (
    compare_roll_strategies,
    forward_iv_from_curve,
    remaining_maturity_at,
    roll_scenario_payoffs,
)
from mho.scenarios.scenario import Scenario


def test_forward_iv_curve_interp_and_flat_ends():
    f = forward_iv_from_curve([0.25, 0.5, 1.0], [0.20, 0.22, 0.25])
    assert f(0.5) == 0.22
    assert abs(f(0.375) - 0.21) < 1e-12  # midpoint
    assert f(0.1) == 0.20    # flat left
    assert f(5.0) == 0.25    # flat right


def test_roll_quotes_basic(surface, market):
    fwd = forward_iv_from_curve([0.0833, 0.25, 0.5, 1.0], [0.21, 0.22, 0.23, 0.24])
    quotes = compare_roll_strategies(
        spot=market.spot, r=market.r, q=market.q, horizon_years=1.0,
        moneyness=0.95, kind="put", option_tenor=0.25, forward_iv=fwd, surface=surface,
        multiplier=market.multiplier,
    )
    styles = {q.style for q in quotes}
    assert styles == {"long_dated", "front_month", "constant_maturity"}
    for q in quotes:
        assert q.total_cost > 0
        assert math.isfinite(q.total_cost)
    fm = next(q for q in quotes if q.style == "front_month")
    assert fm.num_rolls == 4  # 1y / 3M


def test_front_month_cost_scales_with_horizon(surface, market):
    fwd = forward_iv_from_curve([0.25], [0.22])
    kw = dict(spot=market.spot, r=market.r, q=market.q, moneyness=0.95, kind="put",
              option_tenor=0.25, forward_iv=fwd, surface=surface, multiplier=market.multiplier)
    q1 = next(q for q in compare_roll_strategies(horizon_years=0.5, **kw) if q.style == "front_month")
    q2 = next(q for q in compare_roll_strategies(horizon_years=1.0, **kw) if q.style == "front_month")
    assert q2.total_cost > q1.total_cost


def test_remaining_maturity_live_leg():
    # Long-dated ages linearly toward the horizon.
    assert remaining_maturity_at("long_dated", 0.2, 0.25, 0.5) == math.fabs(0.5 - 0.2)
    # Front-month just before a 0.25y expiry has almost no maturity left (gap risk).
    assert remaining_maturity_at("front_month", 0.24, 0.25, 0.5) < 0.02
    # ...and resets to a full tenor right after the roll boundary.
    assert remaining_maturity_at("front_month", 0.26, 0.25, 0.5) > 0.23
    # Constant-maturity keeps a longer leg alive than a near-expiry front-month at the same t.
    assert (remaining_maturity_at("constant_maturity", 0.24, 0.25, 0.5, 0.5)
            > remaining_maturity_at("front_month", 0.24, 0.25, 0.5))


def test_roll_scenario_payoff_immediate_shock_rolling_styles_coincide(surface, market):
    # t_shock = 0 ⇒ both rolling styles hold a freshly struck `option_tenor` option, so their
    # gross payoffs coincide (the long-dated leg is a longer option and legitimately differs).
    fwd = forward_iv_from_curve([0.0833, 0.25, 0.5, 1.0], [0.21, 0.22, 0.23, 0.24])
    scs = [Scenario("now", spot_shock=-0.20, vol_shock=0.08, target_payoff=0, timing_years=0.0)]
    res = {x.style: x for x in roll_scenario_payoffs(
        spot=market.spot, r=market.r, q=market.q, horizon_years=0.5, moneyness=0.95, kind="put",
        option_tenor=0.25, forward_iv=fwd, surface=surface, scenarios=scs, multiplier=market.multiplier)}
    assert res["front_month"].gross_payoff == pytest.approx(res["constant_maturity"].gross_payoff)
    # Payoff/cost still differs because the structural costs differ across styles.
    assert res["front_month"].payoff_to_cost != pytest.approx(res["constant_maturity"].payoff_to_cost)


def test_roll_scenario_payoff_frontmonth_gap_risk(surface, market):
    # A stress landing just before a front-month expiry leaves it with near-zero maturity, so it
    # holds far less protection than a constant-maturity leg of the same program.
    fwd = forward_iv_from_curve([0.0833, 0.25, 0.5, 1.0], [0.21, 0.22, 0.23, 0.24])
    scs = [Scenario("late selloff", spot_shock=-0.08, vol_shock=0.05, target_payoff=0, timing_years=0.24)]
    res = {x.style: x for x in roll_scenario_payoffs(
        spot=market.spot, r=market.r, q=market.q, horizon_years=0.5, moneyness=0.95, kind="put",
        option_tenor=0.25, forward_iv=fwd, surface=surface, scenarios=scs, multiplier=market.multiplier)}
    assert res["front_month"].remaining_maturity < res["constant_maturity"].remaining_maturity
    assert res["front_month"].gross_payoff < res["constant_maturity"].gross_payoff


def test_crossasset_shock_translation():
    xmap = CrossAssetMap("SPX", beta=1.5, correlation=0.9)
    # Exposure expected to fall 15% -> implied SPX move is smaller in magnitude (beta 1.5).
    assert abs(xmap.hedge_shock(-0.15) - (-0.10)) < 1e-12
    assert abs(xmap.basis_risk - (1 - 0.81)) < 1e-12


def test_translate_scenarios_preserves_targets():
    xmap = CrossAssetMap("SPX", beta=2.0, correlation=0.8)
    scs = [Scenario("s", spot_shock=-0.20, vol_shock=0.05, target_payoff=1_000_000)]
    out = translate_scenarios(scs, xmap)
    assert abs(out[0].spot_shock - (-0.10)) < 1e-12
    assert out[0].target_payoff == 1_000_000
    assert out[0].vol_shock == 0.05
