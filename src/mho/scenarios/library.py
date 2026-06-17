"""Preset historical stress scenarios.

Each template captures the *joint* equity / credit move of a well-known crisis window, so a user
can stress a hedge against a realistic episode without inventing shock numbers. The values are
round, illustrative peak-to-trough figures for the acute window (editable in the UI) — they are
calibration starting points, not exact historical prints. Equity = broad index move; credit =
high-yield ETF (HYG-style) price move; vol shocks are additive IV points; steep episodes use a
downside skew twist.
"""

from __future__ import annotations

from dataclasses import dataclass

from .macro import InstrumentShock, MacroScenario

# Asset-class routing for the liquid presets. Anything unlisted defaults to "equity".
_CREDIT_SYMBOLS = {"HYG", "JNK", "LQD"}


def asset_class(symbol: str) -> str:
    return "credit" if symbol.upper() in _CREDIT_SYMBOLS else "equity"


@dataclass(frozen=True)
class StressTemplate:
    key: str
    name: str
    equity: InstrumentShock
    credit: InstrumentShock
    note: str = ""


# Illustrative acute-window moves. Skew-twist used where the sell-off was sharp/convex.
HISTORICAL_STRESSES: dict[str, StressTemplate] = {
    "gfc_2008": StressTemplate(
        "gfc_2008", "2008 GFC (Sep–Nov)",
        equity=InstrumentShock(-0.45, vol_shock=0.40, vol_mode="skew_twist", twist=0.12),
        credit=InstrumentShock(-0.30, vol_shock=0.25, vol_mode="skew_twist", twist=0.10),
        note="Lehman → systemic credit crunch; equities ~-45%, HY spreads blew out, VIX→80.",
    ),
    "covid_2020": StressTemplate(
        "covid_2020", "2020 COVID (Feb–Mar)",
        equity=InstrumentShock(-0.34, vol_shock=0.50, vol_mode="skew_twist", twist=0.15),
        credit=InstrumentShock(-0.22, vol_shock=0.30, vol_mode="skew_twist", twist=0.12),
        note="Fastest-ever bear market; VIX 14→82 in five weeks; HY gapped before the backstop.",
    ),
    "rates_2022": StressTemplate(
        "rates_2022", "2022 Rate Shock",
        equity=InstrumentShock(-0.25, vol_shock=0.15, vol_mode="parallel"),
        credit=InstrumentShock(-0.18, vol_shock=0.10, vol_mode="parallel"),
        note="Hiking cycle; grinding de-rating rather than a vol spike; duration + spread on HY.",
    ),
    "q4_2018": StressTemplate(
        "q4_2018", "2018 Q4 Selloff",
        equity=InstrumentShock(-0.20, vol_shock=0.15, vol_mode="skew_twist", twist=0.08),
        credit=InstrumentShock(-0.06, vol_shock=0.08, vol_mode="parallel"),
        note="Growth-scare / QT drawdown into year-end; equity-led, modest HY contagion.",
    ),
}


def build_macro_scenario(
    template: StressTemplate,
    target_payoff: float,
    symbols: list[str],
    *,
    timing_years: float = 0.0,
    probability: float | None = None,
) -> MacroScenario:
    """Instantiate a `MacroScenario` for the given hedge symbols from a historical template."""
    shocks = {
        sym: (template.credit if asset_class(sym) == "credit" else template.equity)
        for sym in symbols
    }
    return MacroScenario(
        name=template.name, target_payoff=target_payoff, shocks=shocks,
        timing_years=timing_years, probability=probability,
    )
