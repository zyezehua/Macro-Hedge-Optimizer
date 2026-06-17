"""Per-family structure optimization.

Objective: minimise total premium/cost subject to payoff >= target across ALL scenarios.
Decision variables are the family's strike-moneyness parameters. We search with a coarse
grid (using smooth fractional sizing for a well-behaved objective), refine locally with a
derivative-free method, then report the final structure sized in whole contracts.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..instruments.catalog import FAMILIES, StrategyFamily
from ..instruments.option import MarketContext
from ..instruments.strategy import Strategy
from ..pricing.black_scholes import Greeks
from ..pricing.surface import VolSurface
from ..scenarios.scenario import Scenario
from .sizer import SizingResult, size_to_targets

_INFEASIBLE = 1e18


@dataclass
class OptimizationResult:
    family_key: str
    family_name: str
    params: list[float]
    strategy: Strategy
    sizing: SizingResult
    greeks: Greeks
    feasible: bool

    @property
    def total_cost(self) -> float:
        return self.sizing.total_cost


def _cost_at(
    family: StrategyFamily,
    params: list[float],
    market: MarketContext,
    surface: VolSurface,
    scenarios: list[Scenario],
    maturity: float,
    smooth: bool,
    allow_net_credit: bool,
) -> float:
    """Total cost for a parameter vector; large penalty if infeasible (for the search).

    Net-credit structures are rejected by default: minimizing cost without bounding short
    optionality otherwise degenerates into selling unlimited premium (e.g. ATM calls in a
    collar), whose unbounded tail risk the cost metric does not capture. A genuine hedge is
    a net debit — you pay to be protected.
    """
    lo = [b[0] for b in family.bounds]
    hi = [b[1] for b in family.bounds]
    p = [float(np.clip(v, lo[i], hi[i])) for i, v in enumerate(params)]
    strat = family.build(p, market.spot, maturity)
    res = size_to_targets(strat, market, surface, scenarios, allow_fractional=smooth)
    if not res.feasible:
        return _INFEASIBLE
    if not allow_net_credit and res.premium_per_unit < 0:
        return _INFEASIBLE
    return res.total_cost


def optimize_family(
    family_key: str,
    market: MarketContext,
    surface: VolSurface,
    scenarios: list[Scenario],
    maturity: float,
    grid_points: int = 9,
    refine: bool = True,
    allow_net_credit: bool = False,
    n_starts: int = 3,
) -> OptimizationResult:
    family = FAMILIES[family_key]
    lo = [b[0] for b in family.bounds]
    hi = [b[1] for b in family.bounds]

    # 1) Coarse grid over the box-bounded parameter space; keep all feasible points so we can
    #    seed the local search from several basins (the cost surface is piecewise / multimodal,
    #    so a single restart from the global grid minimum can miss a better nearby structure).
    axes = [np.linspace(lo[i], hi[i], grid_points) for i in range(len(family.bounds))]
    scored = []
    for combo in itertools.product(*axes):
        c = _cost_at(family, list(combo), market, surface, scenarios, maturity, True, allow_net_credit)
        if c < _INFEASIBLE:
            scored.append((c, list(combo)))
    scored.sort(key=lambda x: x[0])

    best_params = list(scored[0][1]) if scored else None
    best_cost = scored[0][0] if scored else math.inf

    # 2) Multi-start local refine (derivative-free; cost is piecewise in the params).
    if refine and scored:
        obj = lambda p: _cost_at(family, list(p), market, surface, scenarios, maturity, True, allow_net_credit)
        for _, seed in scored[: max(1, n_starts)]:
            out = minimize(obj, np.array(seed), method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 400})
            if out.fun < best_cost:
                best_cost = out.fun
                best_params = [float(np.clip(v, lo[i], hi[i])) for i, v in enumerate(out.x)]

    if best_params is None or best_cost >= _INFEASIBLE:
        # Fall back to the cheapest-by-build structure so we can report infeasibility.
        params = [b[0] for b in family.bounds]
        strat = family.build(params, market.spot, maturity)
        sizing = size_to_targets(strat, market, surface, scenarios)
        return OptimizationResult(family_key, family.name, params, strat, sizing,
                                  strat.greeks_per_unit(market, surface), feasible=False)

    # 3) Final structure sized in whole contracts.
    strat = family.build(best_params, market.spot, maturity)
    sizing = size_to_targets(strat, market, surface, scenarios, allow_fractional=False)
    greeks = strat.greeks_per_unit(market, surface)
    return OptimizationResult(family_key, family.name, best_params, strat, sizing, greeks,
                              feasible=sizing.feasible)


def optimize_all(
    family_keys: list[str],
    market: MarketContext,
    surface: VolSurface,
    scenarios: list[Scenario],
    maturity: float,
    grid_points: int = 9,
    refine: bool = True,
    allow_net_credit: bool = False,
    n_starts: int = 3,
) -> list[OptimizationResult]:
    return [
        optimize_family(k, market, surface, scenarios, maturity, grid_points, refine,
                        allow_net_credit, n_starts)
        for k in family_keys
    ]
