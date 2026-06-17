"""Single option leg and the market context it is priced in."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketContext:
    """Underlying state shared by all legs of a strategy."""

    spot: float
    r: float
    q: float
    multiplier: float = 100.0

    def reshock(self, spot_shock: float) -> "MarketContext":
        """Return a context with the spot moved by `spot_shock` (e.g. -0.20 = -20%)."""
        return MarketContext(self.spot * (1.0 + spot_shock), self.r, self.q, self.multiplier)


@dataclass(frozen=True)
class OptionLeg:
    """One option position.

    kind     : "call" or "put".
    strike   : absolute strike price.
    maturity : time to expiry in years at inception.
    qty      : signed units per 1 strategy unit (+ long, - short).
    """

    kind: str
    strike: float
    maturity: float
    qty: float = 1.0
