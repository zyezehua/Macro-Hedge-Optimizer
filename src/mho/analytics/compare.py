"""Rank optimized strategies and build a comparison table with a recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..optimize.optimizer import OptimizationResult
from .metrics import (
    annualized_cost_bps,
    best_case_payoff_ratio,
    expected_cost_efficiency,
    worst_case_payoff_ratio,
)

# Families that carry naked short optionality (assignment / unbounded tail risk).
_SHORT_RISK = {"collar", "put_ratio"}


@dataclass
class Comparison:
    table: pd.DataFrame
    recommended_key: str | None
    recommendation_note: str


def _strike_str(res: OptimizationResult) -> str:
    strikes = res.strategy.meta.get("strikes", {})
    return ", ".join(f"{k}={v:.1f}" for k, v in strikes.items())


def build_comparison(
    results: list[OptimizationResult],
    hedged_notional: float,
    horizon_years: float,
    probabilities: dict[str, float] | None = None,
) -> Comparison:
    rows = []
    for r in results:
        s = r.sizing
        flags = []
        if not r.feasible:
            flags.append("INFEASIBLE")
        if r.family_key in _SHORT_RISK:
            flags.append("short-gamma/tail risk")
        if s.premium_per_unit < 0:
            flags.append("net credit")
        rows.append(
            {
                "key": r.family_key,
                "Strategy": r.family_name,
                "Strikes": _strike_str(r),
                "Contracts": s.units if r.feasible else None,
                "Total Cost": s.total_cost if r.feasible else float("inf"),
                "Cost (bps/yr)": annualized_cost_bps(s.total_cost, hedged_notional, horizon_years)
                if r.feasible else float("nan"),
                "Worst Payoff Ratio": worst_case_payoff_ratio(s),
                "Best Payoff Ratio": best_case_payoff_ratio(s),
                "Exp. Cost Eff.": expected_cost_efficiency(s, probabilities),
                "Binding Scenario": s.binding_scenario,
                "Net Delta": round(r.greeks.delta, 2),
                "Net Vega": round(r.greeks.vega_per_volpt, 2),
                "Flags": ", ".join(flags),
                "Feasible": r.feasible,
            }
        )

    df = pd.DataFrame(rows)
    feasible = df[df["Feasible"]].sort_values("Total Cost").reset_index(drop=True)

    if feasible.empty:
        return Comparison(df, None, "No strategy can meet the targets within the surface range. "
                                    "Relax targets, widen scenarios, or extend the surface.")

    best = feasible.iloc[0]
    zero_cost = best["Total Cost"] <= max(1.0, 1e-9 * hedged_notional)
    if zero_cost:
        note = (
            f"Cheapest feasible hedge is a near-zero-cost {best['Strategy']} "
            f"({best['Strikes']}), {int(best['Contracts'])} contracts (~$0 premium): the "
            f"downside protection is financed by the short leg."
        )
    else:
        note = (
            f"Cheapest feasible hedge: {best['Strategy']} ({best['Strikes']}), "
            f"{int(best['Contracts'])} contracts, total cost {best['Total Cost']:,.0f} "
            f"({best['Cost (bps/yr)']:.0f} bps/yr), worst-case payoff ratio "
            f"{best['Worst Payoff Ratio']:.2f}x."
        )
    if best["Flags"]:
        note += (
            f" NOTE: carries {best['Flags']} — it caps upside / is short gamma; confirm this is "
            f"acceptable vs a slightly costlier long-only structure (e.g. the put spread)."
        )

    # Order: feasible by cost, then infeasible appended.
    ordered = pd.concat([feasible, df[~df["Feasible"]]], ignore_index=True)
    display_cols = [c for c in ordered.columns if c not in ("key", "Feasible")]
    return Comparison(ordered[["key"] + display_cols], best["key"], note)
