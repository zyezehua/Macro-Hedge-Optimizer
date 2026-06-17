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
  rolls on two axes — (i) structural carry cost from a user-provided forward IV term structure, and
  (ii) the payoff each style actually delivers when a stress lands *mid-program*, repriced through
  the scenario engine on whichever option is live at `t_shock` (so a front-month roll near expiry
  shows its gap risk).
- **Cross-asset hedge ratio**: map an exposure onto a different hedge instrument via beta, with
  explicit **basis risk**.
- **Historical stress library**: one-click presets for real crises (2008 GFC, 2020 COVID, 2022
  rate shock, 2018 Q4) with *joint* equity + HY-credit shock numbers.
- **Combined cross-asset hedge**: a warehouse carries equity *and* credit-spread risk, and equity
  puts can't hedge an idiosyncratic HY credit event. Size an equity leg (SPX) and a credit leg
  (HYG) **jointly via a linear program** — minimize total premium s.t. the *summed* payoff meets
  each scenario's target — so the optimizer splits protection toward the cheapest leg per scenario
  and flags when an equity-only overlay is simply infeasible.
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

## Deploy to Streamlit Community Cloud

The app is deploy-ready (`requirements.txt` pinned, `runtime.txt` fixes Python 3.13):

1. Push to GitHub (this repo is already public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → pick this repo, branch `main`, main file `app.py` → **Deploy**.
4. Every `git push` to `main` auto-redeploys.

Notes: the free tier is **public** — anyone with the link can use it. Quotes are processed
in-session and never stored, and the app shows an in-page "not investment advice" disclaimer. To
restrict access, set a viewer allow-list in the app's Streamlit Cloud settings, or self-host.

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
  each rolling style using the forward IV curve. The *scenario* payoff of a rolling program is
  computed separately by repricing the option that is live at `t_shock` (correct remaining
  maturity; strike ~constant under carry) through the scenario engine — so carry cost and
  protection-when-it-hits are both on the table for the rolling decision.
- **Pricing.** SPX (cash-settled, European) is priced with Black-Scholes; ETF options
  (SPY/QQQ/HYG/IWM) are American and priced with the Barone-Adesi-Whaley quadratic
  approximation, which adds the early-exercise premium analytically (validated against a CRR
  binomial tree to ~1c). Greeks remain BS analytic (documented approximation).
- **Optimizer robustness.** The per-family search is multi-start: it scores a coarse parameter
  grid and runs a derivative-free (Nelder-Mead) refine from the best *N* seeds, so a piecewise /
  multimodal cost surface doesn't trap the result in a single basin.

## Layout

```
app.py                 Streamlit UI (transaction → surface → scenarios → optimize → export)
config/defaults.yaml   Rates, multipliers, instrument presets, optimizer settings
src/mho/
  pricing/             Black-Scholes, implied-vol solver, vol surface
  instruments/         Option legs, multi-leg strategies, parametric family catalog
  scenarios/           Scenario definition + shock/repricing engine; multi-factor macro
                       scenarios + historical stress library
  optimize/            Analytic sizer, per-family optimizer, LP cross-asset portfolio optimizer
  rolling/             Long-dated vs front-month vs constant-maturity roll costing
  crossasset/          Beta mapping + basis risk
  analytics/           Cost-efficiency metrics + comparison/ranking
  io/                  Excel-paste vol-surface parser
examples/              sample_surface.csv + demo_acquisition_hedge.py
tests/                 Unit tests
```

> PoC for analysis only — not investment advice. All quotes are user-supplied.
