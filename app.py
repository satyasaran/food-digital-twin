"""
app.py - GUI of the digital twin for joint pricing and lot sizing of a
deteriorating food item (Streamlit).

    streamlit run app.py

Layout
  sidebar : all inputs (run control, twin model, supplier schedule, agent, simulated store)
  header  : status and KPIs
  tabs    : Overview | Inputs (data streams) | Outputs (decisions & plan) | Plots |
            Optimiser explorer | Policy comparison | Agent log | About the model
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import assets
from model import ModelParams, inventory_path, profiles, schedule_from_discounts, solve
from simulator import DEFAULT_EVENTS, HOUR_W, StoreConfig
from twin import MODES, AgentConfig, Session, compare_policies, kpis

st.set_page_config(page_title="FreshTwin - digital twin for perishable food", page_icon="🥦", layout="wide")
st.markdown(assets.CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- palette (validated categorical order)
# food-themed categorical order (validated for colour-vision deficiency, adjacent pairs):
# broccoli green, blueberry blue, carrot orange, aubergine violet, corn yellow, radish pink
C = dict(s1="#2e8b3d", s2="#2a78d6", s3="#eb6834", s4="#6250d6", s5="#eda100", s6="#d6457a",
         grey="#8a8985", alert="#c62828")
BRACKET_COLORS = [C["s1"], C["s2"], C["s3"], C["s4"], C["s5"]]


def fig_base(title, ytitle, height=300):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(size=15, color="#1f3d1f")), height=height,
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
                      yaxis_title=ytitle, font=dict(size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eceee4")
    return fig


# ---------------------------------------------------------------- sidebar: inputs
sb = st.sidebar
sb.markdown(assets.veg_strip(), unsafe_allow_html=True)
sb.title("Inputs")

sb.subheader("Run")
mode = sb.selectbox("Decision policy in the test period", list(MODES), format_func=lambda m: MODES[m], index=0)
c1_, c2_ = sb.columns(2)
seed = c1_.number_input("Random seed", 1, 10_000, 7)
warmup_days = c2_.number_input("Warm-up days", 60, 730, 365, step=5,
                               help="Store practice runs first; the twin calibrates on these data.")
test_days = sb.number_input("Test days (twin in control)", 14, 365, 180, step=7)

with sb.expander("Twin model (assumed parameters)", expanded=False):
    b = st.number_input("Price slope b (kg/day per CNY)", 0.1, 10.0, 1.35, 0.05)
    k = st.number_input("Lot-age coefficient k (per day)", -1.0, 1.0, -0.10, 0.01)
    beta = st.number_input("Display coefficient beta (per day)", 0.0, 0.5, 0.02, 0.005, format="%.3f")
    lam = st.number_input("Deterioration rate lambda at 4 C (per day)", 0.01, 1.0, 0.20, 0.01)
    use_eta = st.checkbox("Waiting-time-dependent backlogging", True)
    eta = st.number_input("Impatience eta (per day)", 0.0, 50.0, 4.0, 0.5) if use_eta else None
    delta = st.number_input("Constant backlogged fraction delta", 0.0, 1.0, 0.5, 0.05) if not use_eta else 0.5
    T = st.number_input("Planning horizon T (days)", 7, 84, 28, 7)
    cap = st.number_input("Storage capacity for initial stock (kg)", 5.0, 1000.0, 100.0, 5.0)
    integer_cycles = st.checkbox("Whole-day cycles only (L = T/n integer)", True)
    g = st.number_input("Holding g (per day, x unit cost)", 0.0, 0.2, 0.01, 0.005, format="%.3f")
    f = st.number_input("Holding f (per day^2, x unit cost)", 0.0, 0.1, 0.005, 0.001, format="%.3f")
    K = st.number_input("Ordering cost K (CNY/order)", 0.0, 500.0, 5.0, 1.0)
    C2 = st.number_input("Backorder cost C2 (CNY/kg/day)", 0.0, 20.0, 0.5, 0.1)
    C4 = st.number_input("Lost-sale cost C4 (CNY/kg)", 0.0, 50.0, 2.0, 0.5)

with sb.expander("Supplier all-units discount schedule", expanded=False):
    sched = st.data_editor(pd.DataFrame({"min_qty_kg": [0.0, 25.0, 50.0, 100.0], "discount_%": [0.0, 2.0, 4.0, 6.0]}),
                           num_rows="dynamic", key="sched", width="stretch")
    sched = sched.dropna().sort_values("min_qty_kg").reset_index(drop=True)
    st.caption("First row must start at 0 kg. Discounts apply to the day's wholesale price.")

with sb.expander("Autonomous agent", expanded=False):
    cusum_k = st.number_input("CUSUM reference value", 0.0, 3.0, 0.5, 0.1)
    cusum_h = st.number_input("CUSUM decision interval h (std. dev.)", 1.0, 20.0, 5.0, 0.5)
    cost_thr = st.slider("Cost rule: re-plan if 7-day wholesale median moves by more than (%)", 1, 50, 15)
    wmin, wmax = st.slider("Re-calibration window after a demand alarm (days)", 3, 120, (7, 56))
    temp_alarm = st.number_input("Chiller alarm temperature (C)", 5.0, 15.0, 8.0, 0.5)
    temp_hours = st.number_input("Alarm if hours above it (per day) >=", 1, 24, 3)
    lam_thr = st.slider("Cold-chain rule: re-plan if lambda changes by more than (%)", 1, 100, 10)
    st.markdown("**Rules enabled**")
    r_dem = st.checkbox("Demand (CUSUM)", True)
    r_cost = st.checkbox("Wholesale cost", True)
    r_hor = st.checkbox("End of horizon", True)
    r_cold = st.checkbox("Cold chain (temperature-dependent lambda)", True)
    temp_lam = st.checkbox("Temperature-aware lambda (Q10)", True)
    censor = st.checkbox("Correct demand for stock-outs (censoring)", True)
    q10 = st.number_input("Q10 assumed by the twin", 1.0, 5.0, 2.5, 0.1)

with sb.expander("Simulated store (ground truth)", expanded=False):
    st.caption("The 'physical' store. Change these to test the twin under misspecification.")
    a_base = st.number_input("Market size a (kg/day)", 5.0, 500.0, 40.0, 1.0)
    b_true = st.number_input("True price slope b", 0.1, 10.0, 1.35, 0.05)
    k_true = st.number_input("True lot-age k", -1.0, 1.0, -0.10, 0.01)
    beta_true = st.number_input("True display beta", 0.0, 0.5, 0.02, 0.005, format="%.3f")
    lam_true = st.number_input("True lambda at 4 C", 0.01, 1.0, 0.20, 0.01)
    q10_true = st.number_input("True Q10", 1.0, 5.0, 2.5, 0.1)
    eta_true = st.number_input("True impatience eta", 0.0, 50.0, 4.0, 0.5)
    c_mean = st.number_input("Mean wholesale price (CNY/kg)", 1.0, 100.0, 7.0, 0.5)
    sigma_d = st.number_input("Daily demand noise (log sd)", 0.0, 1.0, 0.15, 0.01)
    excursion = st.number_input("Random chiller excursions per 100 days", 0.0, 20.0, 2.2, 0.2)
    markdown = st.checkbox("Store marks down 30% in last 2 hours when overstocked", False)
    st.markdown("**Injected events** (day index from start)")
    ev_df = st.data_editor(pd.DataFrame(DEFAULT_EVENTS, columns=["start", "end", "kind", "size", "note"]),
                           num_rows="dynamic", key="events", width="stretch",
                           column_config={"kind": st.column_config.SelectboxColumn(options=["demand", "cost", "coldfault"])})


def settings():
    bps = tuple(float(x) for x in sched["min_qty_kg"])
    dis = tuple(float(x) / 100 for x in sched["discount_%"])
    if not bps or bps[0] != 0:
        bps, dis = (0.0,) + bps, (0.0,) + dis
    events = [tuple(r) for r in ev_df.dropna(subset=["start", "end", "kind", "size"]).itertuples(index=False)]
    store = StoreConfig(seed=int(seed), a_base=a_base, b=b_true, k=k_true, beta=beta_true, eta=eta_true,
                        lam_ref=lam_true, Q10=q10_true, c_mean=c_mean, daily_sigma=sigma_d,
                        excursion_prob_per_day=excursion / 100, breakpoints=bps, discounts=dis, K=K, g=g, f=f,
                        C2=C2, C4=C4, storage_cap=cap, markdown=markdown, events=events)
    base = ModelParams(a=a_base, b=b, k=k, beta=beta, lam=lam, eta=eta, delta=delta, g=g, f=f, K=K, C2=C2, C4=C4,
                       T=float(T), breakpoints=bps, costs=schedule_from_discounts(c_mean, bps, dis),
                       storage_cap=cap, integer_cycles=integer_cycles, n_max=int(T))
    agent = AgentConfig(cusum_k=cusum_k, cusum_h=cusum_h, cost_threshold=cost_thr / 100, window_min=wmin,
                        window_max=wmax, window_default=int(T), use_demand_rule=r_dem, use_cost_rule=r_cost,
                        use_horizon_rule=r_hor, use_cold_rule=r_cold, temp_alarm_c=temp_alarm,
                        temp_alarm_hours=int(temp_hours), temperature_aware_lambda=temp_lam,
                        lambda_change_threshold=lam_thr / 100, censoring_correction=censor, Q10=q10)
    return store, base, agent


def new_session():
    store, base, agent = settings()
    s = Session(store, base, agent, mode, int(warmup_days), int(test_days))
    with st.spinner(f"Running {warmup_days} warm-up days under store practice..."):
        s.run_warmup()
    st.session_state.sess = s
    st.session_state.settings_used = (store, base, agent)


sb.divider()
b1, b2 = sb.columns(2)
if b1.button("Reset / apply", type="primary", width="stretch") or "sess" not in st.session_state:
    new_session()
sess: Session = st.session_state.sess
if b2.button("Step 1 day", width="stretch"):
    sess.step()
b3, b4 = sb.columns(2)
if b3.button("Run 7 days", width="stretch"):
    sess.run(7)
if b4.button("Run to end", width="stretch"):
    with st.spinner("Running..."):
        sess.run(sess.warmup_days + sess.test_days)
autoplay = sb.toggle("Live auto-play", False, help="Steps one day per refresh, like a live data feed.")
speed = sb.slider("Seconds per simulated day", 0.0, 3.0, 0.4, 0.1)
sb.caption("Input changes take effect after 'Reset / apply'.")

# ---------------------------------------------------------------- header
df = sess.daily_df()
test = df[df["phase"] == "test"] if len(df) else df
twin = sess.twin
cur_date = sess.store.date_of(sess.day).isoformat() if not sess.done else df["date"].iloc[-1].date().isoformat()
phase = "finished" if sess.done else ("test (twin in control)" if sess.in_test else "warm-up (store practice)")
st.markdown(assets.hero(f"Store {sess.store_cfg.store_id} &middot; SKU {sess.store_cfg.sku} (broccoli) &middot; next decision "
                        f"{cur_date} &middot; day {sess.day} of {sess.warmup_days + sess.test_days} &middot; {phase} "
                        f"&middot; {MODES[sess.mode]}"), unsafe_allow_html=True)
st.progress(min(1.0, max(0.0, (sess.day - sess.warmup_days) / sess.test_days)), text="Test period progress")

K_ = kpis(df, twin.n_replans if sess.mode != "practice" else 0) if len(test) else {}
m = st.columns(7)
m[0].metric("Profit (CNY)", f"{K_.get('Profit (CNY)', 0):,.0f}")
m[1].metric("Waste %", f"{K_.get('Waste (% of purchases)', float('nan')):.1f}")
m[2].metric("Stock-out h %", f"{K_.get('Stock-out hours (% of open hours)', float('nan')):.1f}")
m[3].metric("Re-plans", f"{K_.get('Re-plans (solver calls)', 0)}")
m[4].metric("Price changes", f"{K_.get('Price changes', 0)}")
la = sess.last_action
m[5].metric("Price CNY/kg", f"{la.price:.2f}" if la else "-")
m[6].metric("Order kg", f"{la.order_qty:.1f}" if la else "-")

tabs = st.tabs(["🥦 Overview", "📥 Inputs: data streams", "📤 Outputs: decisions & plan", "📈 Plots",
                "🧮 Optimiser explorer", "⚖️ Policy comparison", "📜 Agent log", "ℹ️ About the model"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader("How the twin works")
    if True:
        st.graphviz_chart("""
digraph G { rankdir=LR; nodesep=0.3; ranksep=0.35; node [shape=box, style="rounded,filled", fillcolor="#f1f8e9", color="#9ccc65", fontname=Helvetica, fontsize=16]; edge [fontname=Helvetica, fontsize=13];
 store [label="Physical store\\n(simulated)", fillcolor="#fff3e0", color="#ffb74d"];
 data [label="Data streams\\nPOS, ERP, supplier,\\nstocktake, waste,\\nback-orders, sensors"];
 est [label="State & parameter\\nestimation\\na, c1, lambda(T), censoring"];
 agent [label="Autonomous agent\\nCUSUM demand, cost,\\nhorizon, cold-chain rules"];
 opt [label="Optimisation engine\\nAlgorithm 1 (exact reduction)\\nProp. 1, Lemmas 1-2"];
 chk [label="Checks\\nTheorem 1, Prop. 2,\\nProp. 4, storage cap"];
 dec [label="Decisions\\nprice p*, order Q*/S*,\\ncycle L*", fillcolor="#c8e6c9", color="#43a047"];
 {rank=same; store; dec} {rank=same; data; chk} {rank=same; est; opt}
 store -> data -> est -> agent; agent -> opt [label="re-plan?"]; opt -> chk -> dec; dec -> store [label="ESL / ERP"];
}""", width="stretch")
    l, r = st.columns(2)
    with l:
        st.subheader("Shelf today")
        if la:
            st_ = pd.DataFrame(sess.ledger.tables["stocktake"])
            age = st_["oldest_lot_age_days"].iloc[-1] if len(st_) else 0
            age = 0 if age is None or age != age else age
            w28 = 100 * df["waste_kg"].tail(28).sum() / max(df["purchased_kg"].tail(28).sum(), 1e-9)
            st.markdown(assets.shelf_label("Broccoli", la.price, la.order_qty, la.next_delivery_in_days,
                                           df["on_hand_kg"].iloc[-1], w28, age, sess.store_cfg.shelf_life_days),
                        unsafe_allow_html=True)
            st.caption(f"Phase: {sess.last_info.get('phase', '-')} | re-planned: "
                       f"{'yes - ' + sess.last_info.get('reasons', '') if sess.last_info.get('replanned') else 'no'}")
    with r:
        st.subheader("Status")
        badges = []
        if twin.plan is not None:
            p = twin.plan
            badges.append(("ok", "✓ Theorem 1 holds") if p["thm1_ok"] else ("warn", "⚠ Theorem 1 fails: review plan"))
            badges.append(("warn", "⚠ Storage cap binds") if p["storage_binding"] else ("ok", "✓ Within storage capacity"))
            be = p["break_even"]
            if be:
                badges.append(("ok" if be["meet"] else "info",
                               f"Break {be['breakpoint']:.0f} kg: offered {100 * be['d_offered']:.1f}% vs "
                               f"break-even {100 * be['d_dagger']:.1f}%"))
            badges.append(("info", f"Plan: cycle {p['L']:.0f} d, in-stock {100 * p['nu']:.0f}% of cycle"))
        if len(df) and df["hours_above_alarm"].iloc[-1] >= sess.agent.temp_alarm_hours:
            badges.append(("warn", f"🌡 Cold-chain alert: {df['hours_above_alarm'].iloc[-1]} h above {sess.agent.temp_alarm_c:.0f} °C"))
        else:
            badges.append(("ok", "✓ Cold chain normal"))
        if twin.n_replans:
            badges.append(("info", f"{twin.n_replans} re-plans so far"))
        st.markdown('<div class="card">' + "".join(f'<span class="badge {k}">{t}</span><br>' for k, t in badges)
                    + "</div>", unsafe_allow_html=True)
        if twin.plan is not None:
            st.caption(f"Theorem 1 check: {twin.plan['thm1_case']}")
        if sess.mode == "practice":
            st.info("Policy is the store's practice: no optimisation plan.")

# ---------------------------------------------------------------- Inputs
with tabs[1]:
    st.subheader("Data streams received (physical -> virtual)")
    if not len(df):
        st.info("No data yet.")
    else:
        L_ = sess.ledger.tables
        last_date = df["date"].iloc[-1].date().isoformat()
        st.caption(f"Most recent trading day: **{last_date}**")
        a, b_ = st.columns(2)
        pos = pd.DataFrame(L_["pos_hourly"])
        pos_d = pos[pos["timestamp"].str.startswith(last_date) & (pos["channel"] == "shelf")]
        fig = fig_base("POS: hourly sales, most recent day", "kg", 280)
        fig.add_bar(x=pos_d["timestamp"].str[11:16], y=pos_d["qty_kg"], name="Sold", marker_color=C["s1"])
        so = pos_d[pos_d["stockout"]]
        fig.add_scatter(x=so["timestamp"].str[11:16], y=[0] * len(so), mode="markers", name="Stock-out hour",
                        marker=dict(symbol="x", size=10, color=C["alert"]))
        fig.update_layout(hovermode="closest")
        a.plotly_chart(fig, width="stretch")
        sen = pd.DataFrame(L_["sensors"])
        sen_d = sen.tail(24)
        fig = fig_base("Chiller temperature, last 24 h", "deg C", 280)
        fig.add_scatter(x=sen_d["timestamp"].str[11:16], y=sen_d["temp_c"], mode="lines+markers", name="Chiller",
                        line=dict(color=C["s1"], width=2), marker=dict(size=6))
        fig.add_hline(y=sess.agent.temp_alarm_c, line=dict(color=C["alert"], dash="dash"),
                      annotation_text="alarm", annotation_position="top left")
        b_.plotly_chart(fig, width="stretch")

        a, b_, c_ = st.columns(3)
        a.markdown("**Supplier price list (today)**")
        a.dataframe(pd.DataFrame(sess.store.pricelist(min(sess.day, sess.store.n_days - 1)))
                    [["tier", "min_qty_kg", "discount", "unit_cost"]], hide_index=True, width="stretch")
        b_.markdown("**Delivery / stocktake (last day)**")
        y = df.iloc[-1]
        b_.dataframe(pd.DataFrame({"field": ["received (kg)", "unit cost paid", "on hand at 22:00 (kg)",
                                             "open back-orders (kg)", "stock-out hours"],
                                   "value": [y["received_kg"], y["unit_cost_paid"], y["on_hand_kg"],
                                             y["open_backorders_kg"], y["stockout_hours"]]}),
                     hide_index=True, width="stretch")
        c_.markdown("**Waste write-offs (last day)**")
        w = pd.DataFrame(L_["waste"])
        c_.dataframe(w[w["date"] == last_date][["reason", "qty_kg"]], hide_index=True, width="stretch")

        st.subheader("Estimated model inputs (virtual state)")
        if twin.plan is not None:
            pi = twin.plan_inputs
            est = pd.DataFrame({
                "input": ["Demand intercept a (kg/day)", "Unit cost c1 (CNY/kg)", "Deterioration lambda (per day)",
                          "Calibration window (days)", "Price slope b (fixed)", "sigma of log demand (warm-up)",
                          "Last implied intercept", "Sales factor rho = U/L (daily cycle)"],
                "value": [pi["a_hat"], pi["c_hat"], pi["lam_hat"], pi["window"], twin.base.b, twin.sigma,
                          df["a_implied"].iloc[-1], twin.rho],
                "source": ["POS, censoring-corrected", "Supplier price list, 7-day median", "Chiller sensors via Q10",
                           "Agent change-point estimate", "Calibration (2SLS)", "Warm-up data", "Yesterday's POS",
                           "Model, Eq. (7)"]})
            st.dataframe(est.style.format({"value": "{:.3f}"}), hide_index=True, width="stretch")
        else:
            st.info("The twin estimates its inputs when it takes control (end of warm-up).")

        st.subheader("Raw tables")
        tname = st.selectbox("Table", ["daily summary"] + [t for t in L_ if t != "ground_truth"] + ["ground_truth (hidden)"])
        if tname == "daily summary":
            tdf = df
        elif tname.startswith("ground_truth"):
            tdf = pd.DataFrame(L_["ground_truth"])
            st.warning("Ground truth is never used by the twin (except the hindsight benchmark).")
        else:
            tdf = pd.DataFrame(L_[tname])
        st.dataframe(tdf.tail(500), width="stretch", height=280)
        st.download_button(f"Download {tname} (CSV)", tdf.to_csv(index=False).encode(), f"{tname.split()[0]}.csv")

# ---------------------------------------------------------------- Outputs
with tabs[2]:
    if twin.plan is None:
        st.info("No optimisation plan yet (warm-up, or store-practice policy). Step into the test period.")
    else:
        p, P = twin.plan, twin.P
        st.subheader("Decisions sent to the store (virtual -> physical)")
        a = st.columns(4)
        a[0].metric("Selling price p* (CNY/kg)", f"{p['p']:.2f}")
        a[1].metric("Cycle length L* (days)", f"{p['L']:.0f}", help=f"n* = {p['n']} cycles in T = {P.T:.0f} days")
        a[2].metric("Order-up-to S* (kg)", f"{p['S']:.1f}")
        a[3].metric("Order quantity Q* (kg)", f"{p['Q']:.1f}", help="Q = S + backlog of the cycle")
        a = st.columns(4)
        a[0].metric("Stock-out time t1* (days)", f"{p['t1']:.3f}", help=f"in-stock fraction nu* = {p['nu']:.3f}")
        a[1].metric("Discount bracket j*", f"{p['j'] + 1} (c = {p['c']:.2f})", help="at a breakpoint" if p["at_breakpoint"] else None)
        a[2].metric("Expected profit TAP* (CNY/day)", f"{p['TAP']:.1f}")
        a[3].metric("Expected waste share D_tau/S", f"{100 * p['waste_share']:.1f}%")

        l, r = st.columns(2)
        with l:
            t, I = inventory_path(P, p)
            fig = fig_base("Planned inventory over one cycle, Eq. (3)", "kg")
            fig.add_scatter(x=t, y=I, mode="lines", name="I(t)", line=dict(color=C["s1"], width=2),
                            fill="tozeroy", fillcolor="rgba(46,139,61,0.15)")
            fig.add_vline(x=p["t1"], line=dict(color=C["grey"], dash="dot"), annotation_text="t1*")
            fig.update_xaxes(title="days since delivery")
            st.plotly_chart(fig, width="stretch")
        with r:
            items = [("Revenue", p["revenue_cycle"]), ("Purchase", -p["purchase_cycle"]), ("Holding", -p["holding_cycle"]),
                     ("Back-order", -p["backorder_cost_cycle"]), ("Lost sales", -p["lost_cost_cycle"]),
                     ("Ordering", -P.K), ("Profit", p["profit_cycle"])]
            fig = fig_base("Expected profit components per cycle", "CNY")
            fig.add_bar(x=[i[0] for i in items], y=[i[1] for i in items],
                        marker_color=[C["s1"] if v >= 0 else C["s3"] for _, v in items],
                        text=[f"{v:,.1f}" for _, v in items], textposition="outside")
            fig.update_layout(hovermode="closest", showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Structural checks")
        be = p["break_even"]
        checks = pd.DataFrame({
            "check": ["Theorem 1 (strict concavity in t1)", "Proposition 2 (planned shortage profitable?)",
                      "Proposition 4 (meet next quantity break?)", "Storage capacity", "Physical quantities per cycle"],
            "result": [("holds " if p["thm1_ok"] else "fails -> review ") + p["thm1_case"],
                       f"Phi(L) = {p['Phi_L']:.3f} -> {'yes, t1* < L' if p['shortage_profitable'] else 'no'}",
                       (f"break {be['breakpoint']:.0f} kg: offered {100 * be['d_offered']:.1f}% vs d-dagger "
                        f"{100 * be['d_dagger']:.1f}% -> {'meet' if be['meet'] else 'do not meet at current cycle'}")
                       if be else "no higher bracket / not applicable",
                       f"S* = {p['S']:.1f} of {P.storage_cap:.0f} kg" + (" (binding)" if p["storage_binding"] else ""),
                       f"sold {p['sold_cycle']:.1f} kg, deteriorated {p['deteriorated_cycle']:.2f} kg, "
                       f"back-ordered {p['backlog_cycle']:.2f} kg, lost {p['lost_cycle']:.2f} kg"]})
        st.dataframe(checks, hide_index=True, width="stretch")

        st.subheader("Plan history")
        ph = pd.DataFrame(twin.plans)
        if len(ph):
            ph.insert(1, "date", [sess.store.date_of(d).isoformat() for d in ph["day"]])
            st.dataframe(ph.round(3), hide_index=True, width="stretch", height=250)
            st.download_button("Download plan history (CSV)", ph.to_csv(index=False).encode(), "plans.csv")

# ---------------------------------------------------------------- Plots
with tabs[3]:
    if not len(df):
        st.info("No data yet.")
    else:
        show_warm = st.checkbox("Include warm-up period", False)
        d = df if show_warm else (test if len(test) else df)
        x = d["date"]
        alarms = d[d["reasons"].fillna("").str.contains("demand alarm")]
        l, r = st.columns(2)

        fig = fig_base("Demand intercept a", "kg/day")
        fig.add_scatter(x=x, y=d["a_implied"], mode="markers", name="Implied by daily sales",
                        marker=dict(size=4, color=C["grey"], opacity=0.6))
        fig.add_scatter(x=x, y=d["gt_a_true"], mode="lines", name="True level (hidden)",
                        line=dict(color=C["s3"], width=2, dash="dash"))
        if d["a_plan"].notna().any():
            fig.add_scatter(x=x, y=d["a_plan"], mode="lines", name="Level used by twin plan",
                            line=dict(color=C["s1"], width=2, shape="hv"))
        fig.add_scatter(x=alarms["date"], y=alarms["a_plan"], mode="markers", name="Demand alarm",
                        marker=dict(symbol="triangle-up", size=11, color=C["alert"]))
        l.plotly_chart(fig, width="stretch")

        fig = fig_base("Prices", "CNY/kg")
        fig.add_scatter(x=x, y=d["price"], mode="lines", name="Selling price", line=dict(color=C["s1"], width=2, shape="hv"))
        fig.add_scatter(x=x, y=d["wholesale"], mode="lines", name="Wholesale price", line=dict(color=C["s2"], width=2))
        if d["c_plan"].notna().any():
            fig.add_scatter(x=x, y=d["c_plan"], mode="lines", name="Cost used by plan",
                            line=dict(color=C["s4"], width=2, dash="dot", shape="hv"))
        r.plotly_chart(fig, width="stretch")

        fig = fig_base("Stock at 22:00 and open back-orders", "kg")
        fig.add_scatter(x=x, y=d["on_hand_kg"], mode="lines", name="On hand", line=dict(color=C["s1"], width=2))
        fig.add_scatter(x=x, y=d["open_backorders_kg"], mode="lines", name="Open back-orders", line=dict(color=C["s2"], width=2))
        fig.add_scatter(x=x, y=d["received_kg"], mode="markers", name="Delivered", marker=dict(size=4, color=C["s3"]))
        l.plotly_chart(fig, width="stretch")

        ws_all = 100 * df["waste_kg"].rolling(28, min_periods=7).sum() / df["purchased_kg"].rolling(28, min_periods=7).sum()
        ws = ws_all.loc[d.index]
        fig = fig_base("Waste share, rolling 28 days", "% of purchases")
        fig.add_scatter(x=x, y=ws, mode="lines", name="Waste share", line=dict(color=C["s3"], width=2))
        r.plotly_chart(fig, width="stretch")

        fig = fig_base("Chiller temperature (daily)", "deg C")
        fig.add_scatter(x=x, y=d["temp_max_c"], mode="lines", name="Daily max", line=dict(color=C["s2"], width=2))
        fig.add_scatter(x=x, y=d["temp_mean_c"], mode="lines", name="Daily mean", line=dict(color=C["s1"], width=2))
        fig.add_hline(y=sess.agent.temp_alarm_c, line=dict(color=C["alert"], dash="dash"), annotation_text="alarm")
        l.plotly_chart(fig, width="stretch")

        fig = fig_base("Cumulative profit (simulator accounting)", "CNY")
        fig.add_scatter(x=x, y=d["fin_profit"].cumsum(), mode="lines", name="Cumulative profit", line=dict(color=C["s1"], width=2))
        r.plotly_chart(fig, width="stretch")

        if d["cusum_pos"].notna().any():
            fig = fig_base("Agent: CUSUM statistics of log demand", "std. dev.")
            fig.add_scatter(x=x, y=d["cusum_pos"], mode="lines", name="Upward", line=dict(color=C["s1"], width=2))
            fig.add_scatter(x=x, y=d["cusum_neg"], mode="lines", name="Downward", line=dict(color=C["s2"], width=2))
            fig.add_hline(y=sess.agent.cusum_h, line=dict(color=C["alert"], dash="dash"), annotation_text="h")
            l.plotly_chart(fig, width="stretch")

        fig = fig_base("Daily sales vs true demand", "kg/day")
        fig.add_scatter(x=x, y=d["gt_true_demand_kg"], mode="lines", name="True demand (hidden)", line=dict(color=C["s3"], width=2))
        fig.add_scatter(x=x, y=d["sales_kg"], mode="lines", name="Sales", line=dict(color=C["s1"], width=2))
        r.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Optimiser explorer
with tabs[4]:
    st.subheader("Solve the model directly (Algorithm 1)")
    st.caption("Starts from the twin's current estimates; change them to see how the optimal policy responds.")
    pi = twin.plan_inputs if twin.plan is not None else {"a_hat": a_base, "c_hat": c_mean, "lam_hat": lam}
    _, base_used, _ = st.session_state.settings_used
    cc = st.columns(4)
    a_x = cc[0].number_input("Demand intercept a", 1.0, 1000.0, float(round(pi["a_hat"], 2)), 1.0)
    c_x = cc[1].number_input("Base unit cost c1", 0.1, 100.0, float(round(pi["c_hat"], 2)), 0.1)
    l_x = cc[2].number_input("Deterioration lambda", 0.01, 2.0, float(round(pi["lam_hat"], 3)), 0.01)
    K_x = cc[3].number_input("Ordering cost K", 0.0, 1000.0, float(base_used.K), 5.0)
    Px = replace(base_used, a=a_x, lam=l_x, K=K_x,
                 costs=schedule_from_discounts(c_x, base_used.breakpoints, sess.store_cfg.discounts))
    t0 = time.perf_counter()
    try:
        px = solve(Px, keep_profiles=True)
        ms = 1000 * (time.perf_counter() - t0)
        st.success(f"Optimum in {ms:.0f} ms: p* = {px['p']:.2f}, n* = {px['n']} (L* = {px['L']:.1f} d), "
                   f"nu* = {px['nu']:.3f}, Q* = {px['Q']:.1f} kg, bracket {px['j'] + 1}, TAP* = {px['TAP']:.2f} CNY/day, "
                   f"Theorem 1 {'holds' if px['thm1_ok'] else 'fails'}")
        l, r = st.columns(2)
        tn = pd.DataFrame(px["TAP_by_n"])
        fig = fig_base("Optimal profit by number of cycles n", "TAP* (CNY/day)")
        fig.add_scatter(x=tn["n"], y=tn["TAP"], mode="lines", line=dict(color=C["grey"], width=1), showlegend=False)
        for j in sorted(tn["j"].unique()):
            s = tn[tn["j"] == j]
            fig.add_scatter(x=s["n"], y=s["TAP"], mode="markers", name=f"bracket {j + 1}",
                            marker=dict(size=10, color=BRACKET_COLORS[j % 5]))
        fig.update_xaxes(title="n (cycles in horizon)", type="category")
        fig.update_layout(hovermode="closest", legend_y=-0.3, height=340)
        l.plotly_chart(fig, width="stretch")
        pr = px["profiles"]
        fig = fig_base(f"Price-optimised profiles G_j at n* = {px['n']}", "TAP (CNY/day)")
        for j in range(Px.J):
            fig.add_scatter(x=pr["nu"], y=pr[j], mode="lines", name=f"bracket {j + 1} (c={Px.costs[j]:.2f})",
                            line=dict(color=BRACKET_COLORS[j % 5], width=2))
        fig.add_scatter(x=[px["nu"]], y=[px["TAP"]], mode="markers", name="optimum",
                        marker=dict(symbol="star", size=14, color="#0b0b0b"))
        fig.update_xaxes(title="in-stock fraction nu = t1/L")
        fig.update_layout(legend_y=-0.3, height=340)
        r.plotly_chart(fig, width="stretch")
        st.caption("Gaps in a profile mark stock-out times at which no price reaches that bracket (Lemma 1) or the storage cap binds.")
    except Exception as e:
        st.error(f"No feasible plan: {e}")

# ---------------------------------------------------------------- Policy comparison
with tabs[5]:
    st.subheader("Compare decision policies on the same simulated store")
    st.caption("Each policy runs on identical demand, prices, temperatures and events (same seed), after the same warm-up.")
    if st.button("Run comparison", type="primary"):
        store_c, base_c, agent_c = settings()
        bar = st.progress(0.0)
        tab, ser = compare_policies(store_c, base_c, agent_c, int(warmup_days), int(test_days),
                                    progress=lambda v, name: bar.progress(v, text=f"finished: {name}"))
        st.session_state.cmp = (tab, ser)
    if "cmp" in st.session_state:
        tab, ser = st.session_state.cmp
        st.dataframe(tab.style.format("{:,.1f}"), width="stretch")
        ref = MODES["static"]
        fig = fig_base("Cumulative profit minus that of the static plan", "CNY", 380)
        cols = [C["s1"], C["s2"], C["s3"], C["grey"], C["s4"], C["s6"]]
        for i, col in enumerate(ser.columns):
            if col == ref:
                continue
            fig.add_scatter(x=ser.index, y=ser[col] - ser[ref], mode="lines", name=col,
                            line=dict(color=cols[i % len(cols)], width=2,
                                      dash="dash" if "Hindsight" in col else "solid"))
        fig.add_hline(y=0, line=dict(color="#52514e", width=1))
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download comparison (CSV)", tab.to_csv().encode(), "policy_comparison.csv")

# ---------------------------------------------------------------- Agent log
with tabs[6]:
    st.subheader("Agent decisions and alerts")
    lg = pd.DataFrame(twin.log)
    if len(lg):
        lg.insert(1, "date", [sess.store.date_of(x).isoformat() for x in lg["day"]])
        kinds = st.multiselect("Event types", sorted(lg["event"].unique()), default=sorted(lg["event"].unique()))
        st.dataframe(lg[lg["event"].isin(kinds)], hide_index=True, width="stretch", height=420)
        st.download_button("Download log (CSV)", lg.to_csv(index=False).encode(), "agent_log.csv")
    else:
        st.info("No agent events yet.")

# ---------------------------------------------------------------- About
with tabs[7]:
    st.subheader("The optimisation model inside the twin")
    st.markdown("Demand while in stock and during stock-outs, inventory dynamics (Eqs. 1-2):")
    st.latex(r"\frac{dI}{dt}+\lambda I=-D(p)e^{kt}-\beta I\ (0\le t\le t_1),\qquad \frac{dI}{dt}=-\delta(L-t)D(p)\ (t_1\le t\le L),\qquad D(p)=a-bp")
    st.markdown("Profit per cycle in bracket $j$ and the problem solved at every re-plan:")
    st.latex(r"\pi_j(p,t_1)=D(p)\{pU(t_1)-W_j(t_1)\}-K,\qquad \max_{n,\;p,\;0<t_1\le T/n}\ \frac{n}{T}\,\pi_{j(Q)}(p,t_1),\quad S=D(p)\tilde S(t_1)\le \bar S")
    st.markdown("Closed-form price (Proposition 1b), projected on the bracket's price interval (Lemma 1):")
    st.latex(r"p_j^\circ(t_1)=\tfrac12\Big(\tfrac ab+\tfrac{W_j(t_1)}{U(t_1)}\Big)")
    st.markdown("""
**Inputs and outputs of the twin**

| Inputs (from the store) | Used for | Outputs (to the store) |
|---|---|---|
| Hourly POS sales, price, stock-out flags | demand intercept $a$ (censoring-corrected) | selling price $p^*$ (pricing system / shelf labels) |
| Supplier price list and discount tiers | $c_j$, $(q_j,c_j)$, $K$ | order quantity $Q^*$, order-up-to $S^*$ (ERP purchase order) |
| Deliveries, stocktake, back-orders | inventory state, backlog | cycle length $L^*$, stock-out time $t_1^*$, bracket $j^*$ |
| Waste write-offs | waste KPIs | Theorem 1 certificate / review flag |
| Chiller temperature | $\\lambda(T)=\\lambda_{ref}Q_{10}^{(T-4)/10}$ | break-even discount $d^\\dagger$ (Prop. 4) |
| Calendar, weather | demand covariates (simulated store) | alerts: demand shift, cost shift, cold chain |

**Agent rules.** Each morning the agent standardises yesterday's implied log-intercept against the plan, updates a
two-sided CUSUM, and re-plans on (i) a demand alarm (re-calibrating on the data since the estimated change point),
(ii) a wholesale-cost deviation, (iii) the end of the horizon, or (iv) a temperature-driven change in $\\lambda$.
Each new plan is checked against Theorem 1 and flagged (not blocked) if the sufficient condition fails.

**Limitations.** The store is simulated and follows the model's structure, so a run tests the data pipeline,
estimation and re-planning logic, not the model's validity for a real store. All parameter values are illustrative;
replace them with the calibrated values. The price slope $b$ is held fixed.
""")

# ---------------------------------------------------------------- live auto-play
if autoplay and not sess.done:
    time.sleep(speed)
    sess.step()
    st.rerun()
