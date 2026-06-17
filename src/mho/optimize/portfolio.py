"""Combined cross-asset hedge: size a portfolio of instruments to meet one target per scenario.

Single-instrument sizing is a closed form (one binding scenario). Once the hedge spans several
instruments — e.g. an SPX put for the equity leg and an HYG put for the credit leg — the cheapest
way to meet *every* scenario's target is a **linear program**: choose a contract count per
instrument to minimize total premium subject to, in each scenario, the summed gross payoff ≥ the
portfolio target. The LP naturally splits protection toward whichever instrument is cheapest per
dollar of payoff in the *binding* scenarios, capturing cross-asset diversification.

Pipeline per instrument:
  1. pick a cost-efficient structure shape (strikes) by optimizing that instrument's family
     against its own slice of the macro shocks;
  2. take its per-unit premium and per-scenario per-unit payoff as one LP column.
Then solve the LP across columns and report the joint allocation, costed in whole contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linprog

from ..instruments.option import MarketContext
from ..pricing.surface import VolSurface
from ..scenarios.engine import evaluate_all
from ..scenarios.macro import MacroScenario
from .optimizer import optimize_family


@dataclass
class HedgeInstrument:
    """A hedge underlier with its own market state, surface and chosen strategy family."""

    symbol: str
    market: MarketContext
    surface: VolSurface
    family_key: str


@dataclass
class PortfolioLeg:
    symbol: str
    family_name: str
    strikes: dict[str, float]
    premium_per_unit: float
    units: int
    payoff_per_unit: dict[str, float]   # scenario name -> gross payoff per unit

    @property
    def total_cost(self) -> float:
        return self.premium_per_unit * self.units


@dataclass
class PortfolioResult:
    feasible: bool
    legs: list[PortfolioLeg] = field(default_factory=list)
    total_cost: float = math.inf
    reason: str = ""

    def payoff_in(self, scenario_name: str) -> float:
        return sum(leg.payoff_per_unit.get(scenario_name, 0.0) * leg.units for leg in self.legs)


def _build_column(inst: HedgeInstrument, macro: list[MacroScenario], maturity: float, **opt_kw):
    """Optimize one instrument's structure shape, return its premium + per-scenario unit payoffs."""
    scns = [m.for_instrument(inst.symbol) for m in macro]
    # Guide strike selection with the portfolio targets on this instrument alone (a stand-in;
    # the LP re-sizes across instruments afterwards).
    guided = [replace(s, target_payoff=m.target_payoff) for s, m in zip(scns, macro)]
    res = optimize_family(inst.family_key, inst.market, inst.surface, guided, maturity, **opt_kw)
    strat = res.strategy
    payoffs = evaluate_all(strat, inst.market, inst.surface, scns)
    premium = payoffs[0].premium_per_unit if payoffs else 0.0
    payoff_map = {p.scenario: p.gross_payoff_per_unit for p in payoffs}
    return strat, premium, payoff_map


def optimize_portfolio(
    instruments: list[HedgeInstrument],
    macro_scenarios: list[MacroScenario],
    maturity: float,
    *,
    grid_points: int = 9,
    refine: bool = True,
    allow_net_credit: bool = False,
    n_starts: int = 3,
) -> PortfolioResult:
    """Jointly size the instruments to meet every scenario's portfolio target at minimum cost."""
    opt_kw = dict(grid_points=grid_points, refine=refine, allow_net_credit=allow_net_credit,
                  n_starts=n_starts)
    cols = []
    for inst in instruments:
        strat, premium, payoff_map = _build_column(inst, macro_scenarios, maturity, **opt_kw)
        cols.append((inst, strat, premium, payoff_map))

    targeted = [m for m in macro_scenarios if m.target_payoff > 0]
    n = len(cols)

    if not targeted:
        # No binding targets: report 1 unit of each for economics, no LP needed.
        legs = [PortfolioLeg(inst.symbol, strat.name, strat.meta.get("strikes", {}), prem, 1, pmap)
                for (inst, strat, prem, pmap) in cols]
        return PortfolioResult(True, legs, sum(l.total_cost for l in legs))

    c = [prem for (_, _, prem, _) in cols]
    # A_ub x <= b_ub  encodes  sum_j payoff[s,j] * x_j >= target_s.
    A_ub, b_ub = [], []
    for m in targeted:
        row = [-cols[j][3].get(m.name, 0.0) for j in range(n)]
        A_ub.append(row)
        b_ub.append(-m.target_payoff)

    out = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, None)] * n, method="highs")
    if not out.success:
        return PortfolioResult(False, reason="No combination of the instruments can meet every "
                               "scenario target (LP infeasible). Relax targets or add an instrument.")

    # Round each allocation up to whole contracts (conservative: keeps every target met).
    units = [int(math.ceil(max(0.0, u) - 1e-9)) for u in out.x]
    legs = []
    for (inst, strat, prem, pmap), u in zip(cols, units):
        legs.append(PortfolioLeg(inst.symbol, strat.name, strat.meta.get("strikes", {}),
                                 prem, u, pmap))
    total = sum(l.total_cost for l in legs)

    # Verify the rounded (integer) allocation still clears every target.
    feasible = all(
        sum(l.payoff_per_unit.get(m.name, 0.0) * l.units for l in legs) >= m.target_payoff - 1e-6
        for m in targeted
    )
    return PortfolioResult(feasible, legs, total,
                           "" if feasible else "Integer rounding left a target marginally unmet.")
