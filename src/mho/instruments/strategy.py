"""A multi-leg option strategy: valuation, premium and aggregate greeks.

Each leg's implied vol is read from the supplied VolSurface at the leg's current moneyness
(strike / spot) and remaining maturity, so a strategy naturally walks along the surface as
spot moves under a stress scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..pricing.black_scholes import Greeks, bs_greeks, bs_price
from ..pricing.surface import VolSurface
from .option import MarketContext, OptionLeg


@dataclass
class Strategy:
    name: str
    legs: list[OptionLeg]
    family: str = ""
    meta: dict = field(default_factory=dict)

    def value_per_unit(
        self,
        market: MarketContext,
        surface: VolSurface,
        t_elapsed: float = 0.0,
    ) -> float:
        """Mark-to-market value of one strategy unit (in currency, multiplier applied).

        t_elapsed shrinks each leg's remaining maturity (used when a scenario lands at
        t_shock > 0). Legs whose maturity has fully elapsed settle at intrinsic.
        """
        total = 0.0
        for leg in self.legs:
            T = leg.maturity - t_elapsed
            if T <= 1e-9:
                fwd = market.spot
                intrinsic = max(fwd - leg.strike, 0.0) if leg.kind == "call" else max(leg.strike - fwd, 0.0)
                total += leg.qty * intrinsic
                continue
            sigma = surface.iv_for_strike(market.spot, leg.strike, T)
            px = bs_price(market.spot, leg.strike, market.r, market.q, sigma, T, leg.kind)
            total += leg.qty * px
        return total * market.multiplier

    def premium_per_unit(self, market: MarketContext, surface: VolSurface) -> float:
        """Upfront cost of one unit at inception (positive = net debit)."""
        return self.value_per_unit(market, surface, t_elapsed=0.0)

    def greeks_per_unit(self, market: MarketContext, surface: VolSurface) -> Greeks:
        """Aggregate per-unit greeks at inception (multiplier applied)."""
        d = g = v = th = rho = 0.0
        for leg in self.legs:
            T = leg.maturity
            sigma = surface.iv_for_strike(market.spot, leg.strike, T)
            lg = bs_greeks(market.spot, leg.strike, market.r, market.q, sigma, T, leg.kind)
            d += leg.qty * lg.delta
            g += leg.qty * lg.gamma
            v += leg.qty * lg.vega
            th += leg.qty * lg.theta
            rho += leg.qty * lg.rho
        m = market.multiplier
        return Greeks(d * m, g * m, v * m, th * m, rho * m)
