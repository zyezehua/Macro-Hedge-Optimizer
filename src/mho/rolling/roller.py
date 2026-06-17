"""Decide between a single long-dated hedge and rolling short-dated options.

Roll cost is modeled on a transparent flat-spot *carry* basis: between rolls the underlying
is assumed to sit at its forward, so the realized cost of each roll is the option's time-decay
(premium paid minus salvage value when sold), priced from the user-provided forward IV term
structure. This isolates the structural cost of each rolling style; path/scenario P&L is
handled separately by the scenario engine.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..pricing.black_scholes import bs_price
from ..pricing.surface import VolSurface


def forward_iv_from_curve(tenors: list[float], ivs: list[float]) -> Callable[[float], float]:
    """Build a forward-IV(tenor) function from a sparse curve (linear interp, flat ends)."""
    t = np.asarray(tenors, float)
    v = np.asarray(ivs, float)
    order = np.argsort(t)
    t, v = t[order], v[order]

    def f(tenor: float) -> float:
        return float(np.interp(tenor, t, v, left=v[0], right=v[-1]))

    return f


@dataclass
class RollQuote:
    style: str
    num_rolls: int
    cost_per_roll: float
    total_cost: float
    note: str


def _price(S, K, r, q, sigma, T, kind, mult):
    return bs_price(S, K, r, q, sigma, T, kind) * mult


def compare_roll_strategies(
    spot: float,
    r: float,
    q: float,
    horizon_years: float,
    moneyness: float,
    kind: str,
    option_tenor: float,
    forward_iv: Callable[[float], float],
    surface: VolSurface,
    *,
    multiplier: float = 100.0,
    cm_roll_fraction: float = 0.5,
    bid_ask_haircut: float = 0.01,
) -> list[RollQuote]:
    """Return cost quotes (per 1 unit) for long-dated, front-month and constant-maturity hedges.

    moneyness/kind/option_tenor describe the recurring hedge leg (e.g. a 0.95 put, 3M tenor).
    """
    K = moneyness * spot
    quotes: list[RollQuote] = []

    # --- Long-dated: a single option covering the whole horizon, priced off the surface. ---
    T_long = max(horizon_years, option_tenor)
    iv_long = surface.iv_for_strike(spot, K, T_long)
    cost_long = _price(spot, K, r, q, iv_long, T_long, kind, multiplier)
    quotes.append(RollQuote(
        "long_dated", 1, cost_long, cost_long,
        f"Single {T_long:.2f}y option @ IV {iv_long:.1%}. No roll friction; highest vega/upfront premium.",
    ))

    # --- Front-month: buy `option_tenor` options, hold to expiry, repeat. Salvage ~ 0 (carry). ---
    n_fm = max(1, math.ceil(horizon_years / option_tenor))
    iv_fm = forward_iv(option_tenor)
    buy_fm = _price(spot, K, r, q, iv_fm, option_tenor, kind, multiplier)
    salvage_fm = 0.0  # flat-spot carry: an OTM option expires worthless
    cost_per_fm = (buy_fm - salvage_fm) * (1 + bid_ask_haircut)
    quotes.append(RollQuote(
        "front_month", n_fm, cost_per_fm, cost_per_fm * n_fm,
        f"{n_fm}x {option_tenor:.2f}y rolls held to expiry @ fwd IV {iv_fm:.1%}. "
        f"Max gamma, but pays full theta each roll.",
    ))

    # --- Constant-maturity: roll early (after cm_roll_fraction of tenor), recover salvage. ---
    roll_every = max(1e-3, cm_roll_fraction * option_tenor)
    n_cm = max(1, math.ceil(horizon_years / roll_every))
    rem = option_tenor - roll_every
    buy_cm = _price(spot, K, r, q, forward_iv(option_tenor), option_tenor, kind, multiplier)
    salvage_cm = _price(spot, K, r, q, forward_iv(rem), rem, kind, multiplier) if rem > 1e-6 else 0.0
    cost_per_cm = (buy_cm - salvage_cm) * (1 + bid_ask_haircut)
    quotes.append(RollQuote(
        "constant_maturity", n_cm, cost_per_cm, cost_per_cm * n_cm,
        f"{n_cm} rolls every {roll_every:.2f}y, keeping ~{option_tenor:.2f}y maturity. "
        f"Stable greeks; sells residual value back each roll.",
    ))

    return quotes
