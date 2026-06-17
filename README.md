# Macro Hedge Optimizer

Cost-efficiency analysis and optimization of **macro overlay hedges** for risk-origination
transactions (acquisition / leveraged financing, underwriting warehouses, etc.).

A bank that originates risk carries market exposure between commitment and distribution: a broad
sell-off, spread widening or vol spike can force a markdown on the warehoused risk. This tool
sizes and ranks hedges built from liquid equity index / ETF options (SPX, QQQ, HYG, …) against
**user-defined stress scenarios and target payoffs**, and tells you the cheapest way to get the
protection you need.

## What it does

- **Price ⇄ implied-vol conversion** (Black-Scholes, Brent solver) — quote premiums either way.
- **Full vol surface** input: paste a *Moneyness(%) × Maturity* grid straight from Excel.
- **Scenario engine**: apply spot + vol shocks (parallel or skew-twist), reprice on a
  mark-to-market basis, and measure the hedge's payoff when the stress lands.
- **Automatic sizing**: solve the number of contracts so every scenario meets its target payoff.
- **Per-family optimization**: for each strategy family (naked put/call, put/call spread, collar,
  put ratio) find the strikes that **minimize total cost subject to payoff ≥ target across all
  scenarios**, then rank families by cost / efficiency.
- **Rolling decision**: compare a single long-dated option vs front-month and constant-maturity
  rolls, costed from a user-provided forward IV term structure.
- **Cross-asset hedge ratio**: map an exposure onto a different hedge instrument via beta, with
  explicit **basis risk**.
- **Outputs**: ranked comparison table, payoff-ratio / annualized cost (bps), greeks, Plotly
  payoff-profile overlay, CSV export.

## Quick start

```bash
pip install -r requirements.txt

# Run the test suite
pytest -q

# Scripted end-to-end example (acquisition warehouse hedged with SPX)
python examples/demo_acquisition_hedge.py

# Launch the interactive app
streamlit run app.py
```

## Key modeling choices

- **MtM payoff framing.** A hedge is monetized when the stress hits, so a scenario's payoff is the
  repriced (mark-to-market) value of the position under the shocked spot/surface, not terminal
  intrinsic. The shock timing `t_shock` is configurable (default immediate).
- **Objective = minimize cost s.t. target.** Sizing is analytic (payoff scales linearly in
  contracts); the binding worst-case scenario sets the contract count.
- **Net-debit constraint (default).** A genuine hedge is a net debit — you pay for protection.
  Net-credit "hedges" are excluded by default because minimizing cost without bounding short
  optionality degenerates into selling unlimited premium (a tail-risk trade, not a hedge). A
  zero-/low-cost collar is still allowed; toggle `allow_net_credit` to relax further.
- **Roll cost on a flat-spot carry basis.** Roll quotes isolate the structural time-decay cost of
  each rolling style using the forward IV curve; path/scenario P&L is handled by the scenario engine.
- **Pricing approximation.** SPX (cash-settled, European) is priced exactly; ETF options
  (SPY/QQQ/HYG) are American — BS ignores the early-exercise premium (documented PoC simplification).

## Layout

```
app.py                 Streamlit UI (transaction → surface → scenarios → optimize → export)
config/defaults.yaml   Rates, multipliers, instrument presets, optimizer settings
src/mho/
  pricing/             Black-Scholes, implied-vol solver, vol surface
  instruments/         Option legs, multi-leg strategies, parametric family catalog
  scenarios/           Scenario definition + shock/repricing engine
  optimize/            Analytic sizer + per-family optimizer
  rolling/             Long-dated vs front-month vs constant-maturity roll costing
  crossasset/          Beta mapping + basis risk
  analytics/           Cost-efficiency metrics + comparison/ranking
  io/                  Excel-paste vol-surface parser
examples/              sample_surface.csv + demo_acquisition_hedge.py
tests/                 Unit tests
```

> PoC for analysis only — not investment advice. All quotes are user-supplied.
