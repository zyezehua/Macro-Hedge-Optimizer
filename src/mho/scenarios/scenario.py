"""User-defined stress scenario."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """A stress scenario the hedge must protect against.

    spot_shock    : fractional move of the underlying (e.g. -0.20 = -20%).
    vol_shock     : additive IV shock in decimals (e.g. 0.05 = +5 vol points).
    target_payoff : required gross MtM payoff (currency) of the hedge under this scenario.
    timing_years  : t_shock, when the stress lands relative to inception (default immediate).
    vol_mode      : "parallel" or "skew_twist".
    twist         : extra downside-vs-ATM tilt for skew_twist mode.
    probability   : optional weight for probability-weighted expected metrics.
    """

    name: str
    spot_shock: float
    vol_shock: float = 0.0
    target_payoff: float = 0.0
    timing_years: float = 0.0
    vol_mode: str = "parallel"
    twist: float = 0.0
    probability: float | None = None
