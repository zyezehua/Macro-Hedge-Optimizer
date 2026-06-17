import math

from mho.instruments.catalog import FAMILIES
from mho.scenarios.engine import evaluate_all, evaluate_unit
from mho.scenarios.scenario import Scenario


def test_naked_put_premium_positive(market, surface):
    strat = FAMILIES["naked_put"].build([0.90], market.spot, 0.5)
    prem = strat.premium_per_unit(market, surface)
    assert prem > 0


def test_put_spread_cheaper_than_naked_put(market, surface):
    naked = FAMILIES["naked_put"].build([0.95], market.spot, 0.5)
    spread = FAMILIES["put_spread"].build([0.95, 0.10], market.spot, 0.5)
    assert spread.premium_per_unit(market, surface) < naked.premium_per_unit(market, surface)


def test_down_shock_put_pays(market, surface):
    strat = FAMILIES["naked_put"].build([0.95], market.spot, 0.5)
    sc = Scenario("crash", spot_shock=-0.25, vol_shock=0.10, target_payoff=0.0)
    res = evaluate_unit(strat, market, surface, sc)
    # A 95% put after a 25% down move should be deep ITM and worth well over premium.
    assert res.gross_payoff_per_unit > res.premium_per_unit
    assert res.net_payoff_per_unit > 0
    assert res.payoff_ratio > 1.0


def test_deeper_put_pays_more_on_crash(market, surface):
    near = FAMILIES["naked_put"].build([0.95], market.spot, 0.5)
    far = FAMILIES["naked_put"].build([0.85], market.spot, 0.5)
    sc = Scenario("crash", spot_shock=-0.30, vol_shock=0.0, target_payoff=0.0)
    p_near = evaluate_unit(near, market, surface, sc).gross_payoff_per_unit
    p_far = evaluate_unit(far, market, surface, sc).gross_payoff_per_unit
    # Higher-strike put has more intrinsic after the same crash.
    assert p_near > p_far


def test_collar_low_or_negative_premium(market, surface):
    collar = FAMILIES["collar"].build([0.90, 1.10], market.spot, 0.5)
    naked = FAMILIES["naked_put"].build([0.90], market.spot, 0.5)
    assert collar.premium_per_unit(market, surface) < naked.premium_per_unit(market, surface)


def test_evaluate_all_uses_shared_premium(market, surface):
    strat = FAMILIES["put_spread"].build([0.95, 0.10], market.spot, 0.5)
    scs = [
        Scenario("mild", -0.10, 0.03, 0.0),
        Scenario("severe", -0.25, 0.10, 0.0),
    ]
    results = evaluate_all(strat, market, surface, scs)
    assert len(results) == 2
    assert results[0].premium_per_unit == results[1].premium_per_unit
    # Severe scenario should pay at least as much as the mild one for a put spread.
    assert results[1].gross_payoff_per_unit >= results[0].gross_payoff_per_unit


def test_greeks_directions(market, surface):
    put = FAMILIES["naked_put"].build([0.95], market.spot, 0.5)
    g = put.greeks_per_unit(market, surface)
    assert g.delta < 0  # long put = negative delta
    assert g.vega > 0
    assert g.gamma > 0
