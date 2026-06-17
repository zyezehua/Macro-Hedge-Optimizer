"""Analytic sizing: how many contracts to meet every scenario's target payoff.

Payoff scales linearly in the number of units, so sizing is closed-form. We pick the unit
count so the *binding* (worst) scenario meets its target; all other scenarios then clear
too. A scenario with a positive target but non-positive per-unit payoff makes the structure
infeasible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..instruments.option import MarketContext
from ..instruments.strategy import Strategy
from ..pricing.surface import VolSurface
from ..scenarios.engine import ScenarioPayoff, evaluate_all
from ..scenarios.scenario import Scenario


@dataclass
class SizingResult:
    feasible: bool
    units: int
    premium_per_unit: float
    total_cost: float
    binding_scenario: str | None
    payoffs: list[ScenarioPayoff]
    reason: str = ""

    def total_gross_payoff(self, scenario_name: str) -> float:
        for p in self.payoffs:
            if p.scenario == scenario_name:
                return p.gross_payoff_per_unit * self.units
        raise KeyError(scenario_name)


def size_to_targets(
    strategy: Strategy,
    market: MarketContext,
    surface: VolSurface,
    scenarios: list[Scenario],
    allow_fractional: bool = False,
) -> SizingResult:
    payoffs = evaluate_all(strategy, market, surface, scenarios)
    premium = payoffs[0].premium_per_unit if payoffs else 0.0

    required_units = 0.0
    binding = None
    for p in payoffs:
        if p.target_payoff <= 0:
            continue
        if p.gross_payoff_per_unit <= 1e-9:
            return SizingResult(
                feasible=False, units=0, premium_per_unit=premium, total_cost=math.inf,
                binding_scenario=p.scenario, payoffs=payoffs,
                reason=f"Structure pays ~0 under scenario '{p.scenario}' but a positive "
                       f"target is required; cannot reach target with this structure.",
            )
        need = p.target_payoff / p.gross_payoff_per_unit
        if need > required_units:
            required_units, binding = need, p.scenario

    if required_units <= 0:
        # No positive targets: minimal position (1 unit) just to evaluate economics.
        units = 1
    else:
        units = required_units if allow_fractional else math.ceil(required_units)

    units_int = units if allow_fractional else int(units)
    return SizingResult(
        feasible=True,
        units=units_int,
        premium_per_unit=premium,
        total_cost=units_int * premium,
        binding_scenario=binding,
        payoffs=payoffs,
    )
