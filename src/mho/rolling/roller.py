"""Decide between a single long-dated hedge and rolling short-dated options.

Two complementary views of each rolling style:

1. **Structural cost** (`compare_roll_strategies`) — modeled on a transparent flat-spot *carry*
   basis: between rolls the underlying is assumed to sit at its forward, so the realized cost of
   each roll is the option's time-decay (premium paid minus salvage value when sold), priced from
   the user-provided forward IV term structure. This isolates the structural cost of each style.

2. **Scenario payoff** (`roll_scenario_payoffs`) — what the program actually pays when a stress
   lands at `t_shock`. This depends on *which* option is live at that instant and how much
   maturity it has left: a front-month roll about to expire holds almost no time value (gap risk),
   while a constant-maturity roll still carries a longer-dated, vega-rich option. The live leg is
   repriced through the same scenario engine used to size the static structures, so the rolling
   decision now reflects both carry cost and protection delivered.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ..instruments.option import MarketContext, OptionLeg
from ..instruments.strategy import Strategy
from ..pricing.black_scholes import bs_price
from ..pricing.surface import VolSurface
from ..scenarios.engine import evaluate_unit
from ..scenarios.scenario import Scenario


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


def remaining_maturity_at(
    style: str,
    t_shock: float,
    option_tenor: float,
    horizon_years: float,
    cm_roll_fraction: float = 0.5,
) -> float:
    """Remaining maturity of the option that is *live* at `t_shock` for a given roll style.

    Under the flat-spot carry assumption each roll is struck near the prevailing spot, so the
    strike is ~constant across rolls; the styles differ only in how much maturity the live option
    has left when the stress lands — which is what drives the difference in scenario payoff.
    """
    if style == "long_dated":
        return max(0.0, max(horizon_years, option_tenor) - t_shock)
    if style == "front_month":
        # Options bought every `option_tenor`, held to expiry; live one started at the last boundary.
        rem = option_tenor - (t_shock % option_tenor) if option_tenor > 1e-12 else 0.0
        return rem if rem > 1e-9 else option_tenor
    if style == "constant_maturity":
        roll_every = max(1e-3, cm_roll_fraction * option_tenor)
        # Each roll re-establishes an `option_tenor` option, so the live one has aged by the time
        # since the last roll boundary.
        return option_tenor - (t_shock % roll_every)
    raise ValueError(f"Unknown roll style: {style!r}")


@dataclass
class RollScenarioPayoff:
    style: str
    scenario: str
    t_shock: float
    remaining_maturity: float
    gross_payoff: float          # MtM of one unit of the live leg under the shock
    payoff_to_cost: float        # gross_payoff / structural total cost of the style
    note: str


def roll_scenario_payoffs(
    spot: float,
    r: float,
    q: float,
    horizon_years: float,
    moneyness: float,
    kind: str,
    option_tenor: float,
    forward_iv: Callable[[float], float],
    surface: VolSurface,
    scenarios: list[Scenario],
    *,
    multiplier: float = 100.0,
    cm_roll_fraction: float = 0.5,
    bid_ask_haircut: float = 0.01,
    american: bool = False,
) -> list[RollScenarioPayoff]:
    """Payoff of each rolling style when each scenario's stress lands at its `t_shock`.

    For every (style, scenario) pair we build the option that is live at `t_shock` (correct
    remaining maturity, strike ~ moneyness x spot under carry) and reprice it through the scenario
    engine under the shocked spot/surface. The payoff is divided by the style's structural total
    cost to give a directly comparable protection-per-dollar number.

    Note: when a scenario's `t_shock` is 0 (immediate stress) every style holds a freshly struck
    option, so their payoffs coincide; the styles separate only for stresses that land mid-program.
    """
    quotes = compare_roll_strategies(
        spot, r, q, horizon_years, moneyness, kind, option_tenor, forward_iv, surface,
        multiplier=multiplier, cm_roll_fraction=cm_roll_fraction, bid_ask_haircut=bid_ask_haircut,
    )
    cost_by_style = {qd.style: qd.total_cost for qd in quotes}

    K = moneyness * spot
    market = MarketContext(spot=spot, r=r, q=q, multiplier=multiplier, american=american)
    out: list[RollScenarioPayoff] = []
    for style in cost_by_style:
        for sc in scenarios:
            rem_T = remaining_maturity_at(style, sc.timing_years, option_tenor, horizon_years,
                                          cm_roll_fraction)
            leg = Strategy(f"{style} live leg", [OptionLeg(kind, K, rem_T, +1)])
            # Apply the shock at the live leg's own clock (t_elapsed handled by rem_T already).
            shock_now = replace(sc, timing_years=0.0)
            payoff = evaluate_unit(leg, market, surface, shock_now)
            gross = payoff.gross_payoff_per_unit
            cost = cost_by_style[style]
            out.append(RollScenarioPayoff(
                style=style, scenario=sc.name, t_shock=sc.timing_years,
                remaining_maturity=rem_T, gross_payoff=gross,
                payoff_to_cost=(gross / cost if cost > 1e-9 else float("inf")),
                note=f"live {kind} {rem_T:.2f}y remaining @ t_shock {sc.timing_years:.2f}y",
            ))
    return out
