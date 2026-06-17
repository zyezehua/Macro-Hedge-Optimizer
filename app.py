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
from mho.rolling.roller import compare_roll_strategies, forward_iv_from_curve
from mho.scenarios.scenario import Scenario

CFG = yaml.safe_load((Path(__file__).resolve().parent / "config" / "defaults.yaml").read_text())
SAMPLE = (Path(__file__).resolve().parent / "examples" / "sample_surface.csv").read_text()

st.set_page_config(page_title="Macro Hedge Optimizer", layout="wide")
st.title("Macro Hedge Optimizer")
st.caption("Cost-efficiency analysis of macro overlay hedges for risk-origination transactions.")


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
default_scen = pd.DataFrame([
    {"name": "Selloff -10%", "spot_shock": -0.10, "vol_shock": 0.05, "target_payoff": 15_000_000,
     "timing_years": 0.0, "vol_mode": "parallel", "twist": 0.0, "probability": 0.20},
    {"name": "Crash -25%", "spot_shock": -0.25, "vol_shock": 0.12, "target_payoff": 35_000_000,
     "timing_years": 0.0, "vol_mode": "skew_twist", "twist": 0.10, "probability": 0.05},
])
scen_df = st.data_editor(default_scen, num_rows="dynamic", width="stretch",
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
