"""
twin.py - the VIRTUAL side: state estimation, the optimisation engine
(model.solve = Algorithm 1) and the autonomous re-planning agent, plus the
benchmark policies and a session runner that couples the twin to the store.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from model import ModelParams, schedule_from_discounts, solve, unit_terms
from simulator import Action, FreshStore, Ledger, StoreConfig

MODES = {
    "agent": "Twin + autonomous agent",
    "calendar": "Twin, re-plan every 28 days",
    "daily": "Twin, re-plan every day",
    "static": "Static plan (calibrated once)",
    "practice": "Store practice (cost-plus, daily orders)",
    "hindsight": "Hindsight plan (true future demand; not implementable)",
}


@dataclass
class AgentConfig:
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    cost_threshold: float = 0.15
    window_default: int = 28
    window_min: int = 7
    window_max: int = 56
    use_demand_rule: bool = True
    use_cost_rule: bool = True
    use_horizon_rule: bool = True
    use_cold_rule: bool = True
    temp_alarm_c: float = 8.0
    temp_alarm_hours: int = 3
    temperature_aware_lambda: bool = True
    lambda_change_threshold: float = 0.10
    censoring_correction: bool = True
    T_ref: float = 4.0
    Q10: float = 2.5
    practice_markup: float = 1.55
    practice_price_test_prob: float = 0.10


class DigitalTwin:
    def __init__(self, base: ModelParams, agent: AgentConfig, discounts, mode: str = "agent", seed: int = 7):
        self.base, self.ag, self.discounts, self.mode, self.seed = base, agent, tuple(discounts), mode, seed
        self.lam_ref = base.lam
        self.rho = float(unit_terms(base, np.array([1.0]), 1.0)["U"][0])   # sales per unit D per day (daily cycle)
        self.plan = None
        self.plan_start = None
        self.plan_inputs = {}
        self.cusum_pos = self.cusum_neg = 0.0
        self.zero_pos = self.zero_neg = 0
        self.sigma = 0.2
        self.n_replans = 0
        self.log: list[dict] = []
        self.plans: list[dict] = []
        self.last_price = None

    # ---------------------------------------------------------------- estimation
    def implied_a(self, row):
        dem = row["demand_est_kg"] if self.ag.censoring_correction else row["shelf_sales_kg"] + row["bo_created_kg"]
        return dem / self.rho + self.base.b * row["price"]

    def build_params(self, a_hat, c_hat, lam_hat):
        return replace(self.base, a=a_hat, lam=lam_hat,
                       costs=schedule_from_discounts(c_hat, self.base.breakpoints, self.discounts))

    def lam_from_temp(self, daily, days=3):
        if not self.ag.temperature_aware_lambda or not daily:
            return self.lam_ref
        T = float(np.mean([r["temp_mean_c"] for r in daily[-days:]]))
        return self.lam_ref * self.ag.Q10 ** ((T - self.ag.T_ref) / 10)

    def warmup(self, daily: list[dict]):
        a = np.array([self.implied_a(r) for r in daily])
        la = np.log(np.maximum(a, 1e-6))
        roll = pd.Series(la).rolling(28, min_periods=7).mean().to_numpy()
        dev = (la - roll)[~np.isnan(roll)]
        self.sigma = float(np.std(dev)) if len(dev) > 10 else 0.2
        if self.mode == "static":
            a_hat, c_hat = float(np.mean(a)), float(np.median([r["wholesale"] for r in daily]))
        else:
            a_hat = float(np.mean(a[-self.ag.window_default:]))
            c_hat = float(np.median([r["wholesale"] for r in daily[-7:]]))
        return a_hat, c_hat

    # ---------------------------------------------------------------- planning
    def replan(self, d, a_hat, c_hat, lam_hat, reason, window):
        P = self.build_params(a_hat, c_hat, lam_hat)
        try:
            plan = solve(P, keep_profiles=True)
        except Exception as e:                                            # keep the old plan
            self.log.append({"day": d, "event": "solver failed", "detail": str(e)})
            return
        self.plan, self.plan_start, self.P = plan, d, P
        self.plan_inputs = dict(a_hat=a_hat, c_hat=c_hat, lam_hat=lam_hat, window=window)
        self.cusum_pos = self.cusum_neg = 0.0
        self.zero_pos = self.zero_neg = d
        self.n_replans += 1
        flag = "" if plan["thm1_ok"] else "  [Theorem 1 condition fails: plan flagged for review]"
        self.log.append({"day": d, "event": "re-plan", "detail": f"{reason}; window {window} d; a={a_hat:.1f}, "
                         f"c1={c_hat:.2f}, lambda={lam_hat:.3f} -> p*={plan['p']:.2f}, L*={plan['L']:.0f} d, "
                         f"S*={plan['S']:.1f} kg{flag}"})
        self.plans.append({"day": d, "reason": reason, **{k: plan[k] for k in
                           ("p", "L", "n", "t1", "nu", "Q", "S", "j", "c", "TAP", "waste_share", "thm1_ok")},
                           **self.plan_inputs})

    # ---------------------------------------------------------------- daily decision
    def decide(self, d: int, daily: list[dict], pricelist, store: FreshStore | None = None):
        info = {"replanned": False, "reasons": "", "cusum_pos": None, "cusum_neg": None, "a_implied": None}
        on_hand = daily[-1]["on_hand_kg"] if daily else 0.0
        bo = daily[-1]["open_backorders_kg"] if daily else 0.0
        med7 = float(np.median([r["wholesale"] for r in daily[-6:]] + [pricelist[0]["base_wholesale"]]))
        if self.mode == "practice":
            return self.practice(d, daily, med7, on_hand, bo), info
        y = daily[-1]
        a_y = self.implied_a(y)
        info["a_implied"] = a_y
        reasons, window = [], self.ag.window_default
        if self.plan is not None and self.mode == "agent":
            z = (math.log(max(a_y, 1e-6)) - math.log(self.plan_inputs["a_hat"])) / self.sigma
            self.cusum_pos = max(0.0, self.cusum_pos + z - self.ag.cusum_k)
            self.cusum_neg = max(0.0, self.cusum_neg - z - self.ag.cusum_k)
            if self.cusum_pos == 0:
                self.zero_pos = d
            if self.cusum_neg == 0:
                self.zero_neg = d
            if self.ag.use_demand_rule and max(self.cusum_pos, self.cusum_neg) > self.ag.cusum_h:
                up = self.cusum_pos > self.cusum_neg
                cp = self.zero_pos if up else self.zero_neg
                window = int(np.clip(d - cp, self.ag.window_min, self.ag.window_max))
                reasons.append(f"demand alarm ({'up' if up else 'down'}, change point {d - cp} d ago)")
            if self.ag.use_cost_rule and abs(med7 - self.plan_inputs["c_hat"]) / self.plan_inputs["c_hat"] > self.ag.cost_threshold:
                reasons.append(f"cost rule (7-day median {med7:.2f} vs plan {self.plan_inputs['c_hat']:.2f})")
            if self.ag.use_horizon_rule and d - self.plan_start >= self.base.T:
                reasons.append("end of 28-day horizon")
            if y["hours_above_alarm"] >= self.ag.temp_alarm_hours:
                lam_now = self.lam_from_temp(daily)
                ch = lam_now / self.plan_inputs["lam_hat"] - 1
                self.log.append({"day": d, "event": "cold-chain alert",
                                 "detail": f"{y['hours_above_alarm']} h above {self.ag.temp_alarm_c:.0f} C yesterday; "
                                           f"estimated lambda {lam_now:.3f} ({100 * ch:+.0f}% vs plan)"})
                if self.ag.use_cold_rule and abs(ch) > self.ag.lambda_change_threshold:
                    reasons.append(f"cold-chain rule (lambda {100 * ch:+.0f}% vs plan)")
            elif self.ag.use_cold_rule and self.plan_inputs["lam_hat"] > self.lam_ref * (1 + self.ag.lambda_change_threshold) \
                    and abs(self.lam_from_temp(daily) / self.plan_inputs["lam_hat"] - 1) > self.ag.lambda_change_threshold:
                reasons.append("cold chain back to normal")
        info.update(cusum_pos=self.cusum_pos, cusum_neg=self.cusum_neg)
        if self.mode == "calendar" and self.plan is not None and d - self.plan_start >= self.base.T:
            reasons.append("calendar (28 days)")
        if self.mode == "daily" and self.plan is not None:
            reasons.append("daily re-plan")
        if self.mode == "hindsight" and self.plan is not None and d - self.plan_start >= self.base.T:
            reasons.append("hindsight (28 days)")
        if reasons or self.plan is None:
            if self.mode == "hindsight" and store is not None:
                a_hat = float(np.mean(store.a_true[d:d + int(self.base.T)]))
                c_hat = float(np.mean(store.wholesale[d:d + int(self.base.T)]))
                lam_hat = self.lam_ref
            else:
                hist = daily[-window:]
                a_hat = float(np.mean([self.implied_a(r) for r in hist]))
                c_hat = med7
                lam_hat = self.lam_from_temp(daily)
            if self.plan is None and self.mode != "hindsight":
                a_hat, c_hat = self.warmup(daily)
            self.replan(d, a_hat, c_hat, lam_hat, "; ".join(reasons) or "initial plan", window)
            info.update(replanned=True, reasons="; ".join(reasons) or "initial plan")
        return self.execute(d, on_hand, bo), info

    def execute(self, d, on_hand, bo):
        p = self.plan
        Ld = max(1, int(round(p["L"])))
        k = (d - self.plan_start) % Ld
        order = max(0.0, p["S"] - on_hand) + bo if k == 0 else 0.0
        return Action(round(p["p"], 2), round(order, 2), Ld - k)

    def practice(self, d, daily, med7, on_hand, bo):
        r = np.random.default_rng([self.seed, 3, d])
        p = self.ag.practice_markup * med7
        if r.random() < self.ag.practice_price_test_prob:
            p *= 1 + r.choice([-0.05, 0.05])
        sales = [x["sales_kg"] for x in daily[-28:]]
        target = 1.2 * (np.mean(sales) if sales else 25.0)
        return Action(round(p, 1), round(max(0.0, target - on_hand) + bo, 1), 1)


# ---------------------------------------------------------------- session runner
class Session:
    """Couples one store (physical) with one twin (virtual)."""

    def __init__(self, store_cfg: StoreConfig, base: ModelParams, agent: AgentConfig, mode="agent",
                 warmup_days=365, test_days=180):
        self.store_cfg, self.base, self.agent, self.mode = store_cfg, base, agent, mode
        self.warmup_days, self.test_days = warmup_days, test_days
        self.store = FreshStore(store_cfg, warmup_days + test_days + 60)
        self.ledger = Ledger(agent.temp_alarm_c)
        self.practice_twin = DigitalTwin(base, agent, store_cfg.discounts, "practice", store_cfg.seed)
        self.twin = DigitalTwin(base, agent, store_cfg.discounts, mode, store_cfg.seed)
        self.last_info = {}
        self.last_action = None

    @property
    def day(self):
        return self.store.d

    @property
    def done(self):
        return self.store.d >= self.warmup_days + self.test_days

    @property
    def in_test(self):
        return self.store.d >= self.warmup_days

    def step(self):
        if self.done:
            return
        d = self.store.d
        pl = self.store.pricelist(d)
        if d < self.warmup_days:
            act, info = self.practice_twin.decide(d, self.ledger.daily, pl)
            info["phase"] = "warm-up"
        else:
            act, info = self.twin.decide(d, self.ledger.daily, pl, self.store)
            info["phase"] = "test"
        rec = self.store.step(act)
        extra = {"phase": info["phase"], "replanned": info["replanned"], "reasons": info["reasons"],
                 "cusum_pos": info.get("cusum_pos"), "cusum_neg": info.get("cusum_neg"),
                 "a_plan": self.twin.plan_inputs.get("a_hat") if info["phase"] == "test" else None,
                 "c_plan": self.twin.plan_inputs.get("c_hat") if info["phase"] == "test" else None,
                 }
        self.ledger.add(rec, act, extra)
        self.ledger.daily[-1]["a_implied"] = self.twin.implied_a(self.ledger.daily[-1])
        self.last_info, self.last_action, self.last_rec = info, act, rec

    def run(self, n_days):
        for _ in range(n_days):
            if self.done:
                break
            self.step()

    def run_warmup(self):
        while self.store.d < self.warmup_days:
            self.step()

    def daily_df(self):
        df = pd.DataFrame(self.ledger.daily)
        if len(df):
            df["day"] = np.arange(len(df))
            df["date"] = pd.to_datetime(df["date"])
        return df


def kpis(df: pd.DataFrame, n_replans: int | None = None) -> dict:
    t = df[df["phase"] == "test"] if "phase" in df else df
    if not len(t):
        return {}
    purchased = t["purchased_kg"].sum()
    dem = t["gt_true_demand_kg"].sum()
    return {
        "Profit (CNY)": t["fin_profit"].sum(),
        "Revenue (CNY)": t["fin_revenue"].sum(),
        "Mean price (CNY/kg)": t["price"].mean(),
        "Sold (kg)": t["sales_kg"].sum(),
        "Waste (% of purchases)": 100 * t["waste_kg"].sum() / purchased if purchased else float("nan"),
        "Lost demand (% of demand)": 100 * t["gt_lost_demand_kg"].sum() / dem if dem else float("nan"),
        "Stock-out hours (% of open hours)": 100 * t["stockout_hours"].sum() / (14 * len(t)),
        "Price changes": int((t["price"].diff().abs() > 1e-9).sum()),
        "Re-plans (solver calls)": n_replans if n_replans is not None else int(t.get("replanned", pd.Series(dtype=bool)).sum()),
    }


def compare_policies(store_cfg, base, agent, warmup_days, test_days, modes=None, progress=None):
    modes = modes or list(MODES)
    out, series = {}, {}
    for i, m in enumerate(modes):
        s = Session(store_cfg, base, agent, m, warmup_days, test_days)
        s.run(warmup_days + test_days)
        df = s.daily_df()
        out[MODES[m]] = kpis(df, s.twin.n_replans if m != "practice" else 0)
        t = df[df["phase"] == "test"]
        series[MODES[m]] = t.set_index("date")["fin_profit"].cumsum()
        if progress:
            progress((i + 1) / len(modes), MODES[m])
    table = pd.DataFrame(out).T
    ref = MODES["static"]
    if ref in table.index:
        table["Profit vs static (%)"] = 100 * (table["Profit (CNY)"] / table.loc[ref, "Profit (CNY)"] - 1)
    return table, pd.DataFrame(series)
