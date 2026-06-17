"""Cost-efficiency metrics for sized hedges."""

from __future__ import annotations

from ..optimize.sizer import SizingResult


def annualized_cost_bps(total_cost: float, hedged_notional: float, horizon_years: float) -> float:
    """Premium expressed as an annualized running cost in bps of hedged notional."""
    if hedged_notional <= 0 or horizon_years <= 0:
        return float("nan")
    return total_cost / hedged_notional / horizon_years * 1e4


def worst_case_payoff_ratio(sizing: SizingResult) -> float:
    """Smallest gross payoff / premium across scenarios that carry a positive target."""
    ratios = [p.payoff_ratio for p in sizing.payoffs if p.target_payoff > 0]
    return min(ratios) if ratios else float("nan")


def best_case_payoff_ratio(sizing: SizingResult) -> float:
    ratios = [p.payoff_ratio for p in sizing.payoffs]
    return max(ratios) if ratios else float("nan")


def expected_net_payoff(sizing: SizingResult, probabilities: dict[str, float] | None) -> float | None:
    """Probability-weighted expected net payoff (total position). None if no probabilities.

    Includes a (1 - sum P) 'no-stress' branch whose hedge payoff is taken as 0 gross,
    i.e. the premium is lost.
    """
    if not probabilities:
        return None
    total = 0.0
    p_sum = 0.0
    for p in sizing.payoffs:
        prob = probabilities.get(p.scenario, 0.0)
        p_sum += prob
        total += prob * p.net_payoff_per_unit * sizing.units
    residual = max(0.0, 1.0 - p_sum)
    total += residual * (-sizing.premium_per_unit * sizing.units)
    return total


def expected_cost_efficiency(sizing: SizingResult, probabilities: dict[str, float] | None) -> float | None:
    """Probability-weighted expected gross payoff divided by total premium."""
    if not probabilities or sizing.total_cost <= 1.0:  # zero-cost: efficiency undefined
        return None
    exp_gross = sum(
        probabilities.get(p.scenario, 0.0) * p.gross_payoff_per_unit * sizing.units
        for p in sizing.payoffs
    )
    return exp_gross / sizing.total_cost
