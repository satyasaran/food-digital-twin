# Digital twin with GUI: joint pricing and lot sizing of a deteriorating food item

A runnable digital twin built around the manuscript's optimisation model. A simulated fresh-food store (the "physical" side) streams operational data. The twin estimates the model inputs, an autonomous agent decides when to re-plan, and Algorithm 1 computes the price and replenishment decisions that go back to the store. A Streamlit GUI shows the inputs, outputs, plots and checks.

## Install and run

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
python check_model.py         # optional: verifies the engine (about 1.5 min)
```

## Files

| File | Content |
|---|---|
| `model.py` | **Optimisation engine.** Closed forms (Eqs. 3–7, holding cost via M(x)), Proposition 1 price, Lemmas 1–2 price intervals and closed brackets, Algorithm 1 (enumerate n and j, grid + bounded Brent in t1), Theorem 1 certificate, Proposition 2 Φ(L), Proposition 4 break-even discount, and the storage cap imposed exactly as a price bound |
| `simulator.py` | **Physical side.** A simulated store emitting POS, ERP receipts, supplier price list, stocktake, waste, back-orders, chiller sensors and calendar; finance accounting; per-day paired randomness |
| `twin.py` | **Virtual side.** Estimation (censoring-corrected demand intercept, 7-day median cost, temperature-aware λ), CUSUM agent with demand, cost, horizon and cold-chain rules, benchmark policies, session runner, policy comparison |
| `app.py` | **GUI** |
| `assets.py` | Original vegetable illustrations (inline SVG), food-themed styling, shelf-label card |
| `check_model.py` | Self-checks: closed forms vs ODE/quadrature (max rel. error ~1e-11); Algorithm 1 vs brute force (agreement to 1e-4); Φ vs finite differences |

## GUI

**Sidebar (inputs)**
- *Run:* policy, seed, warm-up and test length; Reset / apply, Step 1 day, Run 7 days, Run to end; live auto-play with adjustable speed.
- *Twin model:* b, k, β, λ, backlogging (η or δ), T, storage cap, whole-day cycles, g, f, K, C2, C4.
- *Supplier discount schedule:* an editable table.
- *Autonomous agent:* CUSUM settings, cost threshold, re-calibration window, cold-chain alarm, the rules to enable, censoring correction, Q10.
- *Simulated store (ground truth):* true parameters (set them different from the twin's to test misspecification) and an editable table of injected events (demand shifts, cost shocks, chiller faults).

**Tabs**
1. **Overview:** data-flow diagram, latest decision, status flags (Theorem 1, break-even discount, storage, cold chain).
2. **Inputs: data streams.** The latest day's hourly POS with stock-out hours, 24 h chiller trace, price list, delivery and stocktake, waste write-offs; the estimated model inputs and their sources; all raw tables with CSV download.
3. **Outputs: decisions & plan.** p*, L*, n*, S*, Q*, t1*, ν*, bracket, TAP*, expected waste share; the planned inventory path I(t); profit components per cycle; structural checks (Theorem 1, Proposition 2, Proposition 4, capacity); plan history.
4. **Plots:** demand intercept (implied, true, used by the plan, alarms); selling vs wholesale price; stock and back-orders; rolling waste share; chiller temperature; cumulative profit; CUSUM statistics; sales vs true demand.
5. **Optimiser explorer:** solve Algorithm 1 for any a, c1, λ, K; TAP*(n) by bracket; price-optimised profiles G_j(ν).
6. **Policy comparison:** agent vs calendar vs daily re-planning vs static plan vs store practice vs hindsight, on identical simulated conditions; KPI table and cumulative profit relative to the static plan.
7. **Agent log:** every re-plan, with its reason, and every cold-chain alert.
8. **About the model:** equations, an input/output table, agent rules, limitations.

## Result with the default settings (seed 7, 365 warm-up + 180 test days)

| Policy | Profit vs static plan |
|---|---|
| Twin + agent (11 solver calls) | +8.1% |
| Daily re-planning (180 calls) | +5.6% |
| Calendar re-planning (every 28 days) | −6.4% |
| Hindsight (not implementable) | +14.8% |

The pattern is the same as in the manuscript's replay. However, the store is simulated with illustrative parameters, so these numbers demonstrate the software. They are not evidence for the paper.

## Notes and limits

- **Units:** days, kg and CNY; all rates are per day. Parameter values are illustrative. Replace them with the calibrated values (Table `tab:calib`).
- **Cycles:** with "whole-day cycles" on, n is restricted to the divisors of T, so that deliveries fall on whole days.
- **Demand estimate:** the implied demand intercept is (estimated demand) / ρ + b·p, where ρ = U(1)/1 is the model's sales per unit of D(p) for a daily cycle. Demand on stock-out days is extrapolated from the in-stock hours using the hourly sales profile. Without this correction, censored sales bias the estimate down and the twin under-orders.
- **What a run tests:** the simulated store follows the model's structure, so a run tests the data flow, estimation and re-planning logic, not the model's validity for a real store.
- **Connecting a real store:** replace `FreshStore.step()` with readers for the real POS, ERP, WMS and sensor feeds that return the same record structure, and send `Action` to the pricing and ordering systems.
