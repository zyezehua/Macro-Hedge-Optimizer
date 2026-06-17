import math

from mho.analytics.compare import build_comparison
from mho.analytics.metrics import annualized_cost_bps
from mho.instruments.catalog import FAMILIES
from mho.optimize.optimizer import optimize_all, optimize_family
from mho.optimize.sizer import size_to_targets
from mho.scenarios.scenario import Scenario


def _scenarios():
    return [
        Scenario("selloff", spot_shock=-0.15, vol_shock=0.05, target_payoff=1_000_000),
        Scenario("crash", spot_shock=-0.30, vol_shock=0.15, target_payoff=2_000_000),
    ]


def test_sizer_meets_every_target(market, surface):
    strat = FAMILIES["naked_put"].build([0.95], market.spot, 0.5)
    scs = _scenarios()
    res = size_to_targets(strat, market, surface, scs)
    assert res.feasible
    for p in res.payoffs:
        assert p.gross_payoff_per_unit * res.units >= p.target_payoff - 1e-6


def test_sizer_infeasible_when_no_payoff(market, surface):
    # An OTM call that EXPIRES at the shock (timing == maturity) has zero intrinsic on a
    # down-move, so no number of contracts can meet a positive target -> infeasible.
    strat = FAMILIES["naked_call"].build([1.05], market.spot, 0.5)
    scs = [Scenario("crash", spot_shock=-0.30, vol_shock=0.0, target_payoff=1_000_000,
                    timing_years=0.5)]
    res = size_to_targets(strat, market, surface, scs)
    assert not res.feasible


def test_optimizer_feasible_and_meets_target(market, surface):
    res = optimize_family("put_spread", market, surface, _scenarios(), maturity=0.5)
    assert res.feasible
    s = res.sizing
    for p in s.payoffs:
        assert p.gross_payoff_per_unit * s.units >= p.target_payoff - 1e-3


def test_optimizer_naked_put_costlier_than_spread_is_not_assumed(market, surface):
    # Both should be feasible; spread is typically cheaper but we only assert feasibility + cost>0.
    keys = ["naked_put", "put_spread", "collar"]
    results = optimize_all(keys, market, surface, _scenarios(), maturity=0.5)
    feasible = [r for r in results if r.feasible]
    assert feasible
    for r in feasible:
        assert math.isfinite(r.total_cost)


def test_comparison_ranks_by_cost(market, surface):
    keys = ["naked_put", "put_spread", "collar", "put_ratio"]
    results = optimize_all(keys, market, surface, _scenarios(), maturity=0.5)
    comp = build_comparison(results, hedged_notional=50_000_000, horizon_years=0.5,
                            probabilities={"selloff": 0.2, "crash": 0.05})
    assert comp.recommended_key is not None
    feas = comp.table[comp.table["Total Cost"] < math.inf]["Total Cost"].tolist()
    assert feas == sorted(feas)  # cheapest first


def test_multistart_no_worse_than_single_start(market, surface):
    # More restarts can only find an equal-or-cheaper feasible structure.
    scs = _scenarios()
    one = optimize_family("put_spread", market, surface, scs, maturity=0.5, n_starts=1)
    many = optimize_family("put_spread", market, surface, scs, maturity=0.5, n_starts=5)
    assert one.feasible and many.feasible
    assert many.total_cost <= one.total_cost + 1.0


def test_annualized_bps():
    assert annualized_cost_bps(500_000, 50_000_000, 0.5) == 200.0
