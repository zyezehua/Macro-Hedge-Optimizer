"""Parametric strategy families for the optimizer.

Each family exposes a small set of decision variables (in moneyness / width) with box
bounds, plus a builder that turns those parameters into a concrete `Strategy` at a given
spot and option maturity. Spreads are parametrized by (anchor moneyness, width) so strike
ordering is guaranteed and the optimizer can use simple box constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .option import OptionLeg
from .strategy import Strategy


@dataclass(frozen=True)
class StrategyFamily:
    name: str
    param_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    build: Callable[[list[float], float, float], Strategy]
    description: str = ""


def _k(moneyness: float, spot: float) -> float:
    return moneyness * spot


def _naked_put(p, S, T):
    return Strategy("Naked Put", [OptionLeg("put", _k(p[0], S), T, +1)], family="naked_put",
                    meta={"strikes": {"long_put": _k(p[0], S)}})


def _naked_call(p, S, T):
    return Strategy("Naked Call", [OptionLeg("call", _k(p[0], S), T, +1)], family="naked_call",
                    meta={"strikes": {"long_call": _k(p[0], S)}})


def _put_spread(p, S, T):
    m_long, width = p
    kl, ks = _k(m_long, S), _k(m_long - width, S)
    return Strategy("Put Spread", [OptionLeg("put", kl, T, +1), OptionLeg("put", ks, T, -1)],
                    family="put_spread", meta={"strikes": {"long_put": kl, "short_put": ks}})


def _call_spread(p, S, T):
    m_long, width = p
    kl, ks = _k(m_long, S), _k(m_long + width, S)
    return Strategy("Call Spread", [OptionLeg("call", kl, T, +1), OptionLeg("call", ks, T, -1)],
                    family="call_spread", meta={"strikes": {"long_call": kl, "short_call": ks}})


def _collar(p, S, T):
    # Long downside put financed by a short upside call (protective collar).
    m_put, m_call = p
    kp, kc = _k(m_put, S), _k(m_call, S)
    return Strategy("Collar", [OptionLeg("put", kp, T, +1), OptionLeg("call", kc, T, -1)],
                    family="collar", meta={"strikes": {"long_put": kp, "short_call": kc}})


def _put_ratio(p, S, T):
    # 1x2 put ratio: long 1 put, short 2 lower-strike puts (cheap deep protection, tail short).
    m_long, width = p
    kl, ks = _k(m_long, S), _k(m_long - width, S)
    return Strategy("Put Ratio 1x2", [OptionLeg("put", kl, T, +1), OptionLeg("put", ks, T, -2)],
                    family="put_ratio", meta={"strikes": {"long_put": kl, "short_put_x2": ks}})


FAMILIES: dict[str, StrategyFamily] = {
    "naked_put": StrategyFamily(
        "Naked Put", ("put_moneyness",), ((0.70, 1.05),), _naked_put,
        "Buy a put. Simplest convex downside hedge; highest premium per unit of protection.",
    ),
    "naked_call": StrategyFamily(
        "Naked Call", ("call_moneyness",), ((0.95, 1.30),), _naked_call,
        "Buy a call. Upside hedge (e.g. short-squeeze / re-leverage risk).",
    ),
    "put_spread": StrategyFamily(
        "Put Spread", ("long_put_moneyness", "width"), ((0.80, 1.05), (0.03, 0.30)), _put_spread,
        "Buy a put, sell a lower put. Caps protection at the lower strike but much cheaper.",
    ),
    "call_spread": StrategyFamily(
        "Call Spread", ("long_call_moneyness", "width"), ((0.95, 1.25), (0.03, 0.30)), _call_spread,
        "Buy a call, sell a higher call. Bounded upside hedge, cheaper than a naked call.",
    ),
    "collar": StrategyFamily(
        "Collar", ("put_moneyness", "call_moneyness"), ((0.80, 1.00), (1.00, 1.25)), _collar,
        "Long put financed by a short call. Very low / zero net cost; gives up upside.",
    ),
    "put_ratio": StrategyFamily(
        "Put Ratio 1x2", ("long_put_moneyness", "width"), ((0.85, 1.05), (0.05, 0.30)), _put_ratio,
        "Long 1 put, short 2 lower puts. Cheap mid-range protection; re-exposed in the deep tail.",
    ),
}
