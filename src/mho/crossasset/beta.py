"""Cross-asset mapping: hedge an exposure with a different (more liquid) instrument.

When the originated risk behaves like `beta` times the hedge instrument (e.g. a sector book
hedged with SPX, or a credit warehouse hedged with HYG), a stress defined on the *exposure*
must be translated onto the *hedge instrument* before pricing options on it. The residual
correlation gap is surfaced explicitly as basis risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scenarios.scenario import Scenario


@dataclass(frozen=True)
class CrossAssetMap:
    """Linear exposure ~ beta * hedge_instrument relationship.

    beta        : sensitivity of the exposure return to the hedge instrument's return.
    correlation : return correlation between exposure and hedge instrument (for basis risk).
    """

    hedge_symbol: str
    beta: float
    correlation: float = 1.0

    def hedge_shock(self, exposure_shock: float) -> float:
        """Hedge-instrument move implied by an exposure move: idx = exposure / beta."""
        if self.beta == 0:
            raise ValueError("beta must be non-zero to map an exposure shock.")
        return exposure_shock / self.beta

    @property
    def r_squared(self) -> float:
        """Fraction of exposure variance explained by the hedge instrument."""
        return self.correlation ** 2

    @property
    def basis_risk(self) -> float:
        """Unexplained fraction (1 - R^2): the part of the exposure the hedge cannot track."""
        return 1.0 - self.correlation ** 2


def translate_scenarios(scenarios: list[Scenario], xmap: CrossAssetMap) -> list[Scenario]:
    """Re-express exposure-defined scenarios as shocks on the hedge instrument.

    The spot shock is divided by beta; vol shock and target payoff are preserved (the target
    is the currency protection still required on the exposure).
    """
    out = []
    for s in scenarios:
        out.append(
            Scenario(
                name=s.name,
                spot_shock=xmap.hedge_shock(s.spot_shock),
                vol_shock=s.vol_shock,
                target_payoff=s.target_payoff,
                timing_years=s.timing_years,
                vol_mode=s.vol_mode,
                twist=s.twist,
                probability=s.probability,
            )
        )
    return out
