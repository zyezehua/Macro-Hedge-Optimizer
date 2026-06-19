"""Macro Hedge Optimizer — Streamlit front end.

Workflow (top to bottom):
  1. Transaction & horizon   2. Vol surface (paste)   3. Stress scenarios
  4. Strategies & rolling     5. Optimize & compare    6. Export
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from mho.analytics.compare import build_comparison
from mho.analytics.metrics import annualized_cost_bps
from mho.crossasset.beta import CrossAssetMap, translate_scenarios
from mho.instruments.catalog import FAMILIES
from mho.instruments.option import MarketContext
from mho.io.surface_paste import parse_surface
from mho.optimize.optimizer import optimize_all
from mho.pricing.implied_vol import iv_to_price, price_to_iv
from mho.optimize.portfolio import HedgeInstrument, optimize_portfolio
from mho.rolling.roller import (
    compare_roll_strategies,
    forward_iv_from_curve,
    roll_scenario_payoffs,
)
from mho.scenarios.library import (
    HISTORICAL_STRESSES,
    StressTemplate,
    asset_class,
    build_macro_scenario,
)
from mho.scenarios.library_io import dump_templates, load_templates
from mho.scenarios.macro import InstrumentShock
from mho.scenarios.scenario import Scenario

CFG = yaml.safe_load((Path(__file__).resolve().parent / "config" / "defaults.yaml").read_text())
SAMPLE = (Path(__file__).resolve().parent / "examples" / "sample_surface.csv").read_text()

st.set_page_config(page_title="Macro Hedge Optimizer", layout="wide")
st.title("Macro Hedge Optimizer")
st.caption("Cost-efficiency analysis of macro overlay hedges for risk-origination transactions.")
st.warning(
    "**Proof-of-concept — for analysis only, not investment advice.** All quotes are user-supplied "
    "and processed in-session (nothing is stored). Pricing/sizing use documented simplifications "
    "(BS / Barone-Adesi-Whaley, flat-spot roll carry). Do not enter confidential data you are not "
    "comfortable sending to a public app.",
    icon="⚠️",
)


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if v == float("inf"):
        return "∞"
    return f"{v:,.{nd}f}"


# ----------------------------------------------------------------------------
# 1. Transaction & horizon
# ----------------------------------------------------------------------------
st.header("1 · Transaction & horizon")
c1, c2, c3, c4 = st.columns(4)
presets = list(CFG["instruments"].keys())
symbol = c1.selectbox("Hedge instrument", presets, index=0)
inst = CFG["instruments"][symbol]
spot = c1.number_input("Spot", value=5400.0, min_value=0.01, step=1.0)
notional = c2.number_input("Hedged notional ($)", value=500_000_000.0, step=1e6, format="%.0f")
horizon = c2.number_input("Hedge horizon (years)", value=0.5, min_value=0.02, step=0.25)
r = c3.number_input("Risk-free rate r", value=float(CFG["market"]["risk_free_rate"]), step=0.005, format="%.3f")
q = c3.number_input("Dividend yield q", value=float(inst["div_yield"]), step=0.005, format="%.3f")
mult = c4.number_input("Contract multiplier", value=float(CFG["market"]["contract_multiplier"]), step=1.0)
is_american = not inst.get("european", True)
market = MarketContext(spot=spot, r=r, q=q, multiplier=mult, american=is_american)
if is_american:
    c4.caption("⚠️ ETF options are American — priced with the Barone-Adesi-Whaley "
               "early-exercise approximation (greeks still BS).")


# ----------------------------------------------------------------------------
# 2. Vol surface
# ----------------------------------------------------------------------------
st.header("2 · Implied-vol surface")
st.caption("Paste a Moneyness(%) × Maturity grid from Excel (tab or comma separated). "
           "Rows = K/S, columns = tenors (e.g. 1M/3M/6M/1Y), cells = IV.")
surf_text = st.text_area("Surface grid", value=SAMPLE, height=200)
try:
    surface = parse_surface(surf_text)
    grid = pd.DataFrame(surface.vols,
                        index=[f"{m:.0%}" for m in surface.moneyness],
                        columns=[f"{t:.2f}y" for t in surface.maturities])
    st.dataframe(grid.style.format("{:.1%}"), width="stretch")
except Exception as e:  # noqa: BLE001
    st.error(f"Could not parse surface: {e}")
    st.stop()

with st.expander("Premium ⇄ implied-vol converter"):
    cc = st.columns(5)
    conv_K = cc[0].number_input("Strike", value=round(spot * 0.95, 1))
    conv_T = cc[1].number_input("Maturity (yrs)", value=0.5, min_value=0.01)
    conv_kind = cc[2].selectbox("Type", ["put", "call"])
    mode = cc[3].radio("Input", ["IV → price", "price → IV"])
    if mode == "IV → price":
        iv_in = cc[4].number_input("IV", value=0.20, step=0.01, format="%.4f")
        px = iv_to_price(iv_in, spot, conv_K, r, q, conv_T, conv_kind)
        st.write(f"**Price/share = {px:,.4f}**  ·  per contract = {px * mult:,.2f}")
    else:
        px_in = cc[4].number_input("Price/share", value=120.0, step=1.0)
        try:
            st.write(f"**Implied vol = {price_to_iv(px_in, spot, conv_K, r, q, conv_T, conv_kind):.4%}**")
        except ValueError as e:
            st.warning(str(e))


# ----------------------------------------------------------------------------
# 3. Stress scenarios
# ----------------------------------------------------------------------------
st.header("3 · Stress scenarios")
st.caption("Spot/vol shocks plus the gross payoff the hedge must deliver in each. "
           "vol_shock and twist are in vol points (decimals). probability is optional.")

# Custom stress library: define your own equity+credit presets, save them as JSON (nothing is
# stored server-side), and re-upload to restore. Custom presets merge with the historical ones.
st.session_state.setdefault("custom_stresses", {})
with st.expander("Custom stress library (define / save / load your own presets)"):
    up = st.file_uploader("Load a saved stress library (JSON)", type="json", key="stress_upload")
    if up is not None:
        try:
            loaded = load_templates(up.getvalue().decode("utf-8"))
            st.session_state["custom_stresses"].update(loaded)
            st.success(f"Loaded {len(loaded)} custom stress(es): {', '.join(loaded)}.")
        except ValueError as e:
            st.error(str(e))

    st.markdown("**Define a new stress** (equity = index leg, credit = HY ETF leg):")
    dc = st.columns(4)
    new_key = dc[0].text_input("Key (id)", value="my_stress")
    new_name = dc[1].text_input("Display name", value="My Stress")
    new_note = dc[2].text_input("Note", value="")
    modes = ["parallel", "skew_twist"]
    eq = st.columns(4)
    eq_spot = eq[0].number_input("Equity spot shock", value=-0.30, step=0.01, format="%.2f")
    eq_vol = eq[1].number_input("Equity vol shock", value=0.20, step=0.01, format="%.2f")
    eq_mode = eq[2].selectbox("Equity vol mode", modes, key="eqmode")
    eq_twist = eq[3].number_input("Equity twist", value=0.10, step=0.01, format="%.2f")
    cr = st.columns(4)
    cr_spot_s = cr[0].number_input("Credit spot shock", value=-0.18, step=0.01, format="%.2f")
    cr_vol_s = cr[1].number_input("Credit vol shock", value=0.12, step=0.01, format="%.2f")
    cr_mode = cr[2].selectbox("Credit vol mode", modes, key="crmode")
    cr_twist = cr[3].number_input("Credit twist", value=0.05, step=0.01, format="%.2f")
    if st.button("💾 Save stress to session library"):
        if not new_key.strip() or not new_name.strip():
            st.warning("Key and display name are required.")
        else:
            st.session_state["custom_stresses"][new_key.strip()] = StressTemplate(
                new_key.strip(), new_name.strip(),
                equity=InstrumentShock(eq_spot, eq_vol, eq_mode, eq_twist),
                credit=InstrumentShock(cr_spot_s, cr_vol_s, cr_mode, cr_twist),
                note=new_note.strip())
            st.success(f"Saved '{new_key.strip()}' to the session library.")
    if st.session_state["custom_stresses"]:
        st.download_button(
            "⬇ Download custom stress library (JSON)",
            dump_templates(st.session_state["custom_stresses"]).encode(),
            file_name="custom_stresses.json", mime="application/json")
        st.caption("In session: " + ", ".join(st.session_state["custom_stresses"]))

# Historical presets + any custom stresses saved this session.
STRESS_LIB = {**HISTORICAL_STRESSES, **st.session_state["custom_stresses"]}
default_scen = pd.DataFrame([
    {"name": "Selloff -10%", "spot_shock": -0.10, "vol_shock": 0.05, "target_payoff": 15_000_000,
     "timing_years": 0.0, "vol_mode": "parallel", "twist": 0.0, "probability": 0.20},
    {"name": "Crash -25%", "spot_shock": -0.25, "vol_shock": 0.12, "target_payoff": 35_000_000,
     "timing_years": 0.0, "vol_mode": "skew_twist", "twist": 0.10, "probability": 0.05},
])
if "scen_seed" not in st.session_state:
    st.session_state["scen_seed"] = default_scen

# Historical preset loader — append a real crisis as a scenario row, using the shock that matches
# the selected instrument's asset class (equity index vs HY credit ETF).
pc = st.columns([3, 2, 2])
preset_key = pc[0].selectbox("Stress preset (historical + custom)", list(STRESS_LIB),
                             format_func=lambda k: STRESS_LIB[k].name)
preset_target = pc[1].number_input("Preset target payoff ($)", value=35_000_000.0, step=1e6,
                                   format="%.0f")
if pc[2].button("➕ Add preset row"):
    tpl = STRESS_LIB[preset_key]
    sh = tpl.credit if asset_class(symbol) == "credit" else tpl.equity
    new_row = pd.DataFrame([{
        "name": f"{tpl.name} [{symbol}]", "spot_shock": sh.spot_shock, "vol_shock": sh.vol_shock,
        "target_payoff": preset_target, "timing_years": 0.0, "vol_mode": sh.vol_mode,
        "twist": sh.twist, "probability": None}])
    st.session_state["scen_seed"] = pd.concat([st.session_state["scen_seed"], new_row],
                                              ignore_index=True)
st.caption(f"_{STRESS_LIB[preset_key].note}_")

scen_df = st.data_editor(st.session_state["scen_seed"], num_rows="dynamic", width="stretch",
                         key="scen_editor",
                         column_config={"vol_mode": st.column_config.SelectboxColumn(
                             options=["parallel", "skew_twist"])})

# Optional cross-asset mapping.
with st.expander("Cross-asset mapping (exposure hedged with a different instrument)"):
    use_beta = st.checkbox("Map exposure shocks onto the hedge instrument via beta")
    bcol = st.columns(3)
    beta = bcol[0].number_input("Beta (exposure vs hedge)", value=1.4, step=0.1)
    corr = bcol[1].number_input("Correlation", value=0.85, min_value=0.0, max_value=1.0, step=0.05)

scenarios = []
for _, row in scen_df.iterrows():
    if not row.get("name"):
        continue
    prob = row.get("probability")
    scenarios.append(Scenario(
        name=str(row["name"]), spot_shock=float(row["spot_shock"]),
        vol_shock=float(row["vol_shock"]), target_payoff=float(row["target_payoff"]),
        timing_years=float(row["timing_years"]), vol_mode=str(row["vol_mode"]),
        twist=float(row["twist"]),
        probability=None if prob is None or (isinstance(prob, float) and np.isnan(prob)) else float(prob),
    ))
if use_beta:
    xmap = CrossAssetMap(symbol, beta=beta, correlation=corr)
    scenarios = translate_scenarios(scenarios, xmap)
    st.info(f"Exposure shocks mapped onto {symbol} (÷ beta {beta}). "
            f"Basis risk (unhedgeable) ≈ {xmap.basis_risk:.0%}.")


# ----------------------------------------------------------------------------
# 4. Strategies & rolling
# ----------------------------------------------------------------------------
st.header("4 · Strategies & rolling")
sc1, sc2 = st.columns([2, 1])
chosen = sc1.multiselect("Strategy families to optimize", list(FAMILIES.keys()),
                         default=["naked_put", "put_spread", "collar", "put_ratio"],
                         format_func=lambda k: FAMILIES[k].name)
allow_credit = sc1.checkbox("Allow net-credit structures (else hedge must be a net debit)", value=False)
opt_maturity = sc2.number_input("Option maturity for structures (yrs)", value=float(horizon),
                                min_value=0.02, step=0.25)
roll_moneyness = sc2.number_input("Rolling leg moneyness", value=0.95, step=0.01)
roll_tenor = sc2.number_input("Rolling option tenor (yrs)", value=0.25, min_value=0.02, step=0.08)


# ----------------------------------------------------------------------------
# 5. Optimize & compare
# ----------------------------------------------------------------------------
st.header("5 · Optimize & compare")
if st.button("▶ Run optimization", type="primary"):
    if not chosen or not scenarios:
        st.warning("Pick at least one strategy family and define at least one scenario.")
        st.stop()

    results = optimize_all(chosen, market, surface, scenarios, maturity=opt_maturity,
                           grid_points=int(CFG["optimizer"]["grid_points"]),
                           refine=bool(CFG["optimizer"]["refine"]), allow_net_credit=allow_credit,
                           n_starts=int(CFG["optimizer"].get("n_starts", 3)))
    comp = build_comparison(results, hedged_notional=notional, horizon_years=horizon,
                            probabilities={s.name: s.probability for s in scenarios
                                           if s.probability is not None} or None)

    st.subheader("Ranked comparison")
    show = comp.table.drop(columns=["key"]).copy()
    show["Total Cost"] = show["Total Cost"].map(lambda v: fmt(v, 0))
    for col in ("Cost (bps/yr)", "Worst Payoff Ratio", "Best Payoff Ratio", "Exp. Cost Eff."):
        show[col] = show[col].map(lambda v: fmt(v, 2))
    st.dataframe(show, width="stretch")
    st.success(comp.recommendation_note)
    st.session_state["export_df"] = comp.table.drop(columns=["key"])

    # Payoff-profile overlay: P&L of each sized hedge vs terminal spot (at t_shock=0 MtM).
    st.subheader("Payoff profile (hedge MtM P&L vs spot move)")
    shocks = np.linspace(-0.40, 0.20, 61)
    fig = go.Figure()
    for r_ in results:
        if not r_.feasible:
            continue
        prem = r_.sizing.premium_per_unit * r_.sizing.units
        pnl = []
        for sh in shocks:
            mk = market.reshock(sh)
            val = r_.strategy.value_per_unit(mk, surface, t_elapsed=0.0) * r_.sizing.units
            pnl.append(val - prem)
        fig.add_trace(go.Scatter(x=shocks * 100, y=pnl, mode="lines", name=r_.family_name))
    for s in scenarios:
        fig.add_vline(x=s.spot_shock * 100, line_dash="dot", line_color="grey",
                      annotation_text=s.name, annotation_position="top")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(xaxis_title="Spot move (%)", yaxis_title="Hedge net P&L ($)",
                      height=460, legend_title="Strategy")
    st.plotly_chart(fig, width="stretch")

    # Rolling vs long-dated.
    st.subheader("Rolling vs long-dated (recurring downside leg)")
    fwd_curve = list(surface.maturities)
    fwd_iv = forward_iv_from_curve(
        fwd_curve, [surface.iv(roll_moneyness, t) for t in fwd_curve])
    quotes = compare_roll_strategies(
        spot=spot, r=r, q=q, horizon_years=horizon, moneyness=roll_moneyness, kind="put",
        option_tenor=roll_tenor, forward_iv=fwd_iv, surface=surface, multiplier=mult,
        cm_roll_fraction=0.5, bid_ask_haircut=float(CFG["costs"]["bid_ask_haircut"]))
    rolldf = pd.DataFrame([{
        "Style": qd.style, "Rolls": qd.num_rolls, "Cost/roll": round(qd.cost_per_roll, 0),
        "Total cost": round(qd.total_cost, 0), "Note": qd.note} for qd in quotes])
    st.dataframe(rolldf, width="stretch")
    cheapest = min(quotes, key=lambda qd: qd.total_cost)
    st.caption(f"Lowest-cost roll path under flat-spot carry: **{cheapest.style}** "
               f"(${cheapest.total_cost:,.0f}). Forward IV term structure taken from the surface "
               f"at moneyness {roll_moneyness:.0%}.")

    # Roll payoff when the stress lands mid-program (path P&L through the scenario engine).
    st.subheader("Roll payoff when the stress lands mid-program")
    st.caption("Each scenario's shock is repriced on the option that is *live* at its t_shock for "
               "every roll style — so a front-month roll near expiry shows its gap risk, while a "
               "constant-maturity roll keeps a longer, vega-rich leg. Set a scenario's timing_years "
               "> 0 to separate the styles (an immediate t_shock=0 leaves both rolling styles equal).")
    roll_pay = roll_scenario_payoffs(
        spot=spot, r=r, q=q, horizon_years=horizon, moneyness=roll_moneyness, kind="put",
        option_tenor=roll_tenor, forward_iv=fwd_iv, surface=surface, scenarios=scenarios,
        multiplier=mult, cm_roll_fraction=0.5, bid_ask_haircut=float(CFG["costs"]["bid_ask_haircut"]),
        american=is_american)
    if roll_pay:
        paydf = pd.DataFrame([{
            "Style": p.style, "Scenario": p.scenario, "t_shock (y)": round(p.t_shock, 2),
            "Live maturity (y)": round(p.remaining_maturity, 3),
            "Gross payoff": round(p.gross_payoff, 0),
            "Payoff / cost": round(p.payoff_to_cost, 2)} for p in roll_pay])
        st.dataframe(paydf, width="stretch")


# ----------------------------------------------------------------------------
# 5b. Combined cross-asset hedge (equity + credit leg)
# ----------------------------------------------------------------------------
st.header("5b · Combined cross-asset hedge")
st.caption("A risk-origination warehouse carries **both** equity and credit-spread risk. Equity "
           "puts can't hedge an idiosyncratic HY credit event. Add a credit leg (e.g. HYG) and let "
           "a linear program size the two instruments jointly to meet each historical stress at "
           "minimum cost.")
HYG_SURFACE = ("Moneyness,0.25,0.5,1.0\n0.80,0.24,0.23,0.22\n0.90,0.20,0.20,0.20\n"
               "1.00,0.17,0.17,0.18\n1.10,0.16,0.17,0.18\n")
with st.expander("Configure & run combined hedge"):
    enable_combined = st.checkbox("Enable combined cross-asset hedge")
    gc = st.columns(4)
    eq_family = gc[0].selectbox("Equity leg family", ["put_spread", "naked_put", "put_ratio"],
                                format_func=lambda k: FAMILIES[k].name)
    cr_symbol = gc[1].selectbox("Credit instrument", ["HYG", "JNK", "LQD"], index=0)
    cr_spot = gc[2].number_input("Credit spot", value=78.0, min_value=0.01, step=1.0)
    cr_q = gc[3].number_input("Credit div yield", value=0.055, step=0.005, format="%.3f")
    cr_surf_text = st.text_area("Credit instrument vol surface (Moneyness × Maturity)",
                                value=HYG_SURFACE, height=140)
    st.caption("Pick historical stresses and the portfolio-level target each must deliver "
               "(equity + credit shocks are taken from the preset by asset class).")
    preset_targets = st.data_editor(
        pd.DataFrame([
            {"preset": "q4_2018", "target_payoff": 20_000_000, "probability": 0.12},
            {"preset": "gfc_2008", "target_payoff": 30_000_000, "probability": 0.04},
        ]),
        num_rows="dynamic", width="stretch",
        column_config={"preset": st.column_config.SelectboxColumn(
            options=list(STRESS_LIB))})

    if enable_combined and st.button("▶ Run combined hedge"):
        try:
            cr_surface = parse_surface(cr_surf_text)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not parse credit surface: {e}")
            st.stop()
        macro = []
        for _, row in preset_targets.iterrows():
            key = row.get("preset")
            if key not in STRESS_LIB:
                continue
            prob = row.get("probability")
            macro.append(build_macro_scenario(
                STRESS_LIB[key], float(row["target_payoff"]), [symbol, cr_symbol],
                probability=None if prob is None or (isinstance(prob, float) and np.isnan(prob))
                else float(prob)))
        if not macro:
            st.warning("Add at least one preset stress with a target.")
        else:
            eq_inst = HedgeInstrument(symbol, market, surface, eq_family)
            cr_inst = HedgeInstrument(cr_symbol,
                                      MarketContext(cr_spot, r, cr_q, mult,
                                                    american=not CFG["instruments"].get(
                                                        cr_symbol, {}).get("european", False)),
                                      cr_surface, "naked_put")
            eq_only = optimize_portfolio([eq_inst], macro, maturity=opt_maturity)
            both = optimize_portfolio([eq_inst, cr_inst], macro, maturity=opt_maturity)

            def _leg_rows(res):
                return pd.DataFrame([{
                    "Symbol": lg.symbol, "Structure": lg.family_name,
                    "Strikes": ", ".join(f"{k}={v:.1f}" for k, v in lg.strikes.items()),
                    "Contracts": lg.units, "Cost": round(lg.total_cost, 0)} for lg in res.legs])

            m1, m2 = st.columns(2)
            m1.metric(f"Equity-only ({symbol})",
                      f"${eq_only.total_cost:,.0f}" if eq_only.feasible else "INFEASIBLE")
            m2.metric(f"Combined ({symbol} + {cr_symbol})",
                      f"${both.total_cost:,.0f}" if both.feasible else "INFEASIBLE")
            if both.feasible:
                st.dataframe(_leg_rows(both), width="stretch")

                # Probability-weighted expected metrics (if any preset carries a probability).
                exp_eff = both.expected_cost_efficiency(macro)
                exp_net = both.expected_net_payoff(macro)
                if exp_eff is not None:
                    e1, e2 = st.columns(2)
                    e1.metric("Exp. cost efficiency (prob-weighted)", f"{exp_eff:.2f}x")
                    e2.metric("Exp. net P&L (prob-weighted)", f"${exp_net:,.0f}")

                # Per-leg payoff contribution by scenario, with the target marked.
                fig = go.Figure()
                for lg in both.legs:
                    if lg.units == 0:
                        continue
                    fig.add_trace(go.Bar(
                        name=f"{lg.symbol} {lg.family_name}", x=[m.name for m in macro],
                        y=[lg.payoff_per_unit.get(m.name, 0.0) * lg.units for m in macro]))
                fig.add_trace(go.Scatter(
                    name="Target", mode="markers", x=[m.name for m in macro],
                    y=[m.target_payoff for m in macro],
                    marker=dict(symbol="line-ew", size=40, line=dict(width=3, color="black"))))
                fig.update_layout(barmode="stack", height=420, yaxis_title="Gross payoff ($)",
                                  xaxis_title="Stress scenario", legend_title="Hedge leg",
                                  title="Combined hedge payoff by scenario (stacked legs vs target)")
                st.plotly_chart(fig, width="stretch")
            if not eq_only.feasible and both.feasible:
                st.success(f"The equity-only overlay can't hedge every stress, but adding the "
                           f"{cr_symbol} credit leg makes the program feasible.")
            elif eq_only.feasible and both.feasible and both.total_cost < eq_only.total_cost - 1:
                st.success(f"Adding the {cr_symbol} credit leg lowers total cost by "
                           f"${eq_only.total_cost - both.total_cost:,.0f}.")


# ----------------------------------------------------------------------------
# 6. Export
# ----------------------------------------------------------------------------
st.header("6 · Export")
if "export_df" in st.session_state:
    st.download_button("⬇ Download comparison (CSV)",
                       st.session_state["export_df"].to_csv(index=False).encode(),
                       file_name="hedge_comparison.csv", mime="text/csv")
else:
    st.caption("Run the optimization to enable CSV export.")
