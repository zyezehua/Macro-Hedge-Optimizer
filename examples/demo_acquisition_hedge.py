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
from mho.optimize.portfolio import HedgeInstrument, optimize_portfolio
from mho.scenarios.library import HISTORICAL_STRESSES, build_macro_scenario
from mho.scenarios.macro import InstrumentShock, MacroScenario
from mho.rolling.roller import (
    compare_roll_strategies,
    forward_iv_from_curve,
    roll_scenario_payoffs,
)
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
    print("ROLL PAYOFF WHEN THE STRESS LANDS MID-PROGRAM  (t_shock = 0.24y, just before a FM expiry)")
    print("=" * 110)
    print("  Same stresses, but priced on the option that is LIVE at t_shock for each roll style:")
    mid_scenarios = [
        Scenario("Selloff -10%", spot_shock=-0.10, vol_shock=0.05, timing_years=0.24),
        Scenario("Crash -25%", spot_shock=-0.25, vol_shock=0.12, timing_years=0.24,
                 vol_mode="skew_twist", twist=0.10),
    ]
    payoffs = roll_scenario_payoffs(
        spot=spot, r=market.r, q=market.q, horizon_years=horizon, moneyness=0.95, kind="put",
        option_tenor=0.25, forward_iv=fwd_iv, surface=surface, scenarios=mid_scenarios,
        multiplier=market.multiplier,
    )
    pay_df = pd.DataFrame([{
        "Style": p.style, "Scenario": p.scenario, "Live maturity": f"{p.remaining_maturity:.2f}y",
        "Gross payoff": f"{p.gross_payoff:,.0f}", "Payoff/cost": f"{p.payoff_to_cost:.2f}",
    } for p in payoffs])
    print(pay_df.to_string(index=False))
    print("  -> front_month is ~at expiry here, so it holds only intrinsic (gap risk); the "
          "constant-maturity\n     leg keeps a longer, vega-rich option. Read alongside the cost "
          "table above to trade off carry vs protection.")

    print("\n" + "=" * 110)
    print("CROSS-ASSET EXAMPLE  (deal risk ~ 1.4x SPX; map exposure shocks onto SPX)")
    print("=" * 110)
    xmap = CrossAssetMap("SPX", beta=1.4, correlation=0.85)
    translated = translate_scenarios(scenarios, xmap)
    for s0, s1 in zip(scenarios, translated):
        print(f"  {s0.name:14s}: exposure {s0.spot_shock:+.0%}  ->  SPX {s1.spot_shock:+.1%}")
    print(f"  R^2 = {xmap.r_squared:.2f} | basis risk (unhedgeable) = {xmap.basis_risk:.0%}")

    print("\n" + "=" * 110)
    print("COMBINED CROSS-ASSET HEDGE  (equity SPX put-spread + credit HYG put, historical stresses)")
    print("=" * 110)
    # The warehouse carries BOTH equity and credit-spread risk; hedge with two instruments and let
    # the optimizer split protection toward the cheaper-per-payoff leg in each crisis.
    hyg_surface = parse_surface(
        "Moneyness,0.25,0.5,1.0\n"
        "0.80,0.24,0.23,0.22\n0.90,0.20,0.20,0.20\n1.00,0.17,0.17,0.18\n1.10,0.16,0.17,0.18\n")
    spx_inst = HedgeInstrument("SPX", market, surface, "put_spread")
    hyg_inst = HedgeInstrument("HYG", MarketContext(78.0, market.r, 0.055, 100.0, american=True),
                               hyg_surface, "naked_put")
    # Stress lands at expiry (terminal intrinsic) so an out-of-the-money leg pays nothing — this
    # is what makes the two channels separable and forces the optimizer to use both instruments.
    # The key point: equity puts CANNOT hedge an idiosyncratic HY credit event, so the credit leg
    # earns its place even though equity puts alone would cover the equity-led drawdown.
    macro = [
        # Equity-led drawdown (HYG barely moves) ⇒ only the SPX leg pays.
        build_macro_scenario(HISTORICAL_STRESSES["q4_2018"], 20_000_000, ["SPX", "HYG"],
                             timing_years=horizon, probability=0.12),
        # HY credit event: spreads blow out while equities grind higher (risk-on in stocks, stress
        # isolated to credit, à la 2015–16 HY energy) ⇒ every SPX put expires worthless, so only the
        # HYG leg can pay and the credit hedge becomes indispensable.
        MacroScenario("HY credit event", 18_000_000,
                      {"SPX": InstrumentShock(0.06, 0.0),
                       "HYG": InstrumentShock(-0.22, 0.20, "skew_twist", 0.10)},
                      timing_years=horizon, probability=0.06),
    ]
    print("  Stresses (equity / credit shocks per instrument):")
    for m in macro:
        eq, cr = m.shocks["SPX"], m.shocks["HYG"]
        print(f"    {m.name:22s} target {m.target_payoff/1e6:4.0f}mm | "
              f"SPX {eq.spot_shock:+.0%}/{eq.vol_shock:+.0%}vol  HYG {cr.spot_shock:+.0%}/{cr.vol_shock:+.0%}vol")
    equity_only = optimize_portfolio([spx_inst], macro, maturity=horizon)
    combined = optimize_portfolio([spx_inst, hyg_inst], macro, maturity=horizon)

    def _show(title, res):
        status = "FEASIBLE" if res.feasible else "INFEASIBLE"
        cost = f"{res.total_cost:,.0f}" if res.feasible else "—"
        print(f"\n  {title}: {status} | total cost = {cost}")
        for leg in res.legs:
            sk = ", ".join(f"{k}={v:.1f}" for k, v in leg.strikes.items())
            print(f"      {leg.symbol:4s} {leg.family_name:12s} {leg.units:7,d} contracts | "
                  f"cost {leg.total_cost:12,.0f} | {sk}")
        if not res.feasible:
            print(f"      -> {res.reason}")

    _show("Equity-only (SPX put-spread)", equity_only)
    _show("Combined (SPX put-spread + HYG put)", combined)
    if combined.feasible:
        eff = combined.expected_cost_efficiency(macro)
        net = combined.expected_net_payoff(macro)
        if eff is not None:
            print(f"\n  Probability-weighted (combined): expected cost efficiency = {eff:.2f}x "
                  f"| expected net P&L = {net:,.0f}")
    print("\n  Equity puts expire worthless in the credit event (equities rise), so an equity-only "
          "overlay\n  CANNOT hedge it; the LP-sized credit leg closes the gap. When both legs can "
          "pay, the LP instead\n  splits toward whichever is cheapest per dollar of payoff in the "
          "binding scenario.")


if __name__ == "__main__":
    main()
