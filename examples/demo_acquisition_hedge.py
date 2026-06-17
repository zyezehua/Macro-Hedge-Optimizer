"""End-to-end demo: hedging an acquisition-financing warehouse with SPX options.

Scenario: a bank has underwritten a $500mm bridge to an acquisition and warehouses the risk
for ~6 months until syndication. A broad equity sell-off would widen spreads and force a
markdown. We size and compare macro overlay hedges built from SPX options.

Run:  python examples/demo_acquisition_hedge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make 'mho' importable when run directly from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mho.analytics.compare import build_comparison
from mho.crossasset.beta import CrossAssetMap, translate_scenarios
from mho.instruments.option import MarketContext
from mho.io.surface_paste import parse_surface
from mho.optimize.optimizer import optimize_all
from mho.rolling.roller import compare_roll_strategies, forward_iv_from_curve
from mho.scenarios.scenario import Scenario

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def main() -> None:
    surface = parse_surface((Path(__file__).parent / "sample_surface.csv").read_text())

    spot = 5400.0  # SPX
    market = MarketContext(spot=spot, r=0.043, q=0.013, multiplier=100.0)
    hedged_notional = 500_000_000
    horizon = 0.5

    scenarios = [
        Scenario("Selloff -10%", spot_shock=-0.10, vol_shock=0.05,
                 target_payoff=15_000_000, probability=0.20),
        Scenario("Crash -25%", spot_shock=-0.25, vol_shock=0.12,
                 target_payoff=35_000_000, vol_mode="skew_twist", twist=0.10, probability=0.05),
    ]

    families = ["naked_put", "put_spread", "collar", "put_ratio"]
    results = optimize_all(families, market, surface, scenarios, maturity=horizon)
    comp = build_comparison(
        results, hedged_notional=hedged_notional, horizon_years=horizon,
        probabilities={s.name: s.probability for s in scenarios},
    )

    print("=" * 110)
    print("HEDGE STRATEGY COMPARISON  (SPX spot 5,400 | $500mm warehouse | 6M horizon)")
    print("=" * 110)
    def fmt(v, nd=2):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        if v == float("inf"):
            return "∞"
        return f"{v:,.{nd}f}"

    show = comp.table.drop(columns=["key"]).copy()
    show["Total Cost"] = show["Total Cost"].map(lambda v: fmt(v, 0))
    for col in ("Worst Payoff Ratio", "Best Payoff Ratio", "Exp. Cost Eff.", "Cost (bps/yr)"):
        show[col] = show[col].map(lambda v: fmt(v, 2))
    print(show.to_string(index=False))
    print("\nRECOMMENDATION:\n  " + comp.recommendation_note)

    print("\n" + "=" * 110)
    print("ROLLING vs LONG-DATED  (recurring 0.95 put, 3M tenor)")
    print("=" * 110)
    fwd_iv = forward_iv_from_curve([1 / 12, 0.25, 0.5, 1.0], [0.21, 0.215, 0.22, 0.225])
    quotes = compare_roll_strategies(
        spot=spot, r=market.r, q=market.q, horizon_years=horizon, moneyness=0.95,
        kind="put", option_tenor=0.25, forward_iv=fwd_iv, surface=surface,
        multiplier=market.multiplier,
    )
    for q in quotes:
        print(f"  {q.style:18s} | rolls={q.num_rolls:2d} | cost/roll={q.cost_per_roll:9,.0f} "
              f"| total={q.total_cost:11,.0f}\n      {q.note}")

    print("\n" + "=" * 110)
    print("CROSS-ASSET EXAMPLE  (deal risk ~ 1.4x SPX; map exposure shocks onto SPX)")
    print("=" * 110)
    xmap = CrossAssetMap("SPX", beta=1.4, correlation=0.85)
    translated = translate_scenarios(scenarios, xmap)
    for s0, s1 in zip(scenarios, translated):
        print(f"  {s0.name:14s}: exposure {s0.spot_shock:+.0%}  ->  SPX {s1.spot_shock:+.1%}")
    print(f"  R^2 = {xmap.r_squared:.2f} | basis risk (unhedgeable) = {xmap.basis_risk:.0%}")


if __name__ == "__main__":
    main()
