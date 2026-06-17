"""Apply a scenario's shocks, reprice the hedge, and report its per-unit payoff.

Payoff uses a mark-to-market framing: when the stress lands the hedge is monetized, so the
'gross' payoff is the repriced value of one strategy unit under the shocked spot/surface at
the (reduced) remaining maturity. 'net' subtracts the upfront premium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..instruments.option import MarketContext
from ..instruments.strategy import Strategy
from ..pricing.surface import VolSurface
from .scenario import Scenario


@dataclass(frozen=True)
class ScenarioPayoff:
    scenario: str
    premium_per_unit: float
    gross_payoff_per_unit: float
    net_payoff_per_unit: float
    target_payoff: float

    @property
    def payoff_ratio(self) -> float:
        """Gross payoff / premium. inf for an effectively zero/credit-premium structure."""
        if self.premium_per_unit <= 1e-6:  # < a fraction of a cent per contract = zero-cost
            return math.inf
        return self.gross_payoff_per_unit / self.premium_per_unit


def evaluate_unit(
    strategy: Strategy,
    market: MarketContext,
    surface: VolSurface,
    scenario: Scenario,
    premium_per_unit: float | None = None,
) -> ScenarioPayoff:
    """Per-unit payoff of `strategy` under `scenario`.

    `premium_per_unit` may be passed in to avoid recomputing it across many scenarios.
    """
    if premium_per_unit is None:
        premium_per_unit = strategy.premium_per_unit(market, surface)

    shocked_market = market.reshock(scenario.spot_shock)
    shocked_surface = surface.shocked(scenario.vol_shock, scenario.vol_mode, scenario.twist)
    gross = strategy.value_per_unit(shocked_market, shocked_surface, t_elapsed=scenario.timing_years)

    return ScenarioPayoff(
        scenario=scenario.name,
        premium_per_unit=premium_per_unit,
        gross_payoff_per_unit=gross,
        net_payoff_per_unit=gross - premium_per_unit,
        target_payoff=scenario.target_payoff,
    )


def evaluate_all(
    strategy: Strategy,
    market: MarketContext,
    surface: VolSurface,
    scenarios: list[Scenario],
) -> list[ScenarioPayoff]:
    premium = strategy.premium_per_unit(market, surface)
    return [evaluate_unit(strategy, market, surface, s, premium) for s in scenarios]
