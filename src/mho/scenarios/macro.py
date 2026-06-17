"""Multi-factor (cross-asset) stress scenarios.

A risk-origination warehouse loses value through *several* macro channels at once: a broad
equity sell-off marks down the equity-sensitive piece, while credit-spread widening marks down
the financing/credit piece. Different hedge instruments respond to different channels — SPX/QQQ
puts to the equity move, HYG puts to the credit move — so a scenario must carry a separate shock
for each instrument, and the combined hedge is judged on the *summed* payoff versus one target.

`MacroScenario` owns the portfolio-level target and a per-instrument shock map. A single-asset
`Scenario` (used by the existing engine/sizer) is derived per instrument via `for_instrument`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenario import Scenario


@dataclass(frozen=True)
class InstrumentShock:
    """How one hedge instrument is shocked under a macro scenario."""

    spot_shock: float            # fractional move of that instrument (e.g. -0.20)
    vol_shock: float = 0.0       # additive IV shock (decimals)
    vol_mode: str = "parallel"   # "parallel" | "skew_twist"
    twist: float = 0.0           # extra downside-vs-ATM tilt for skew_twist


@dataclass(frozen=True)
class MacroScenario:
    """A cross-asset stress the combined hedge must protect against.

    target_payoff is the gross MtM payoff required from the *whole* hedge portfolio (summed across
    instruments) when this stress lands. `shocks` maps an instrument symbol to its own shock.
    """

    name: str
    target_payoff: float
    shocks: dict[str, InstrumentShock]
    timing_years: float = 0.0
    probability: float | None = None

    def for_instrument(self, symbol: str) -> Scenario:
        """Per-instrument single-asset Scenario (target is handled at the portfolio level → 0).

        A symbol with no shock entry is treated as unaffected (zero shock).
        """
        sh = self.shocks.get(symbol, InstrumentShock(0.0))
        return Scenario(
            name=self.name,
            spot_shock=sh.spot_shock,
            vol_shock=sh.vol_shock,
            target_payoff=0.0,
            timing_years=self.timing_years,
            vol_mode=sh.vol_mode,
            twist=sh.twist,
            probability=self.probability,
        )
