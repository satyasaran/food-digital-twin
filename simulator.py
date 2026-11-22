"""
simulator.py - the PHYSICAL side: a simulated fresh-food store that emits the
data streams of a retailer's systems (POS, ERP receipts, supplier price list,
stocktake, waste write-offs, back-orders, chiller sensors, calendar) and reacts
to the twin's daily decision (price, order quantity).

Randomness is drawn per day from a seed that depends only on (seed, day), so
different policies run on the same seed face the same demand, prices,
temperatures and events (paired comparison).

Values are illustrative (broccoli-like, kg / CNY / days).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np

OPEN_HOURS = list(range(8, 22))
_HW = np.array([4, 6, 8, 9, 7, 5, 5, 6, 8, 10, 11, 9, 7, 5], float)
HOUR_W = _HW / _HW.sum()
HOLIDAYS_MMDD = {"01-01", "05-01", "05-02", "10-01", "10-02", "12-24", "12-25", "12-31"}

DEFAULT_EVENTS = [
    # start_day, end_day, kind, size, note
    (120, 160, "demand", 1.60, "summer demand surge (warm-up)"),
    (250, 275, "cost", 1.35, "wholesale price shock (warm-up)"),
    (330, 99999, "demand", 0.85, "permanent demand drop: new competitor"),
    (420, 455, "demand", 1.50, "demand surge (test period)"),
    (480, 500, "cost", 1.30, "wholesale price shock (test period)"),
    (510, 516, "coldfault", 4.0, "chiller fault, +4 C at night (test period)"),
]


@dataclass
class StoreConfig:
    store_id: str = "S01"
    sku: str = "BROC-001"
    start: str = "2024-01-01"
    seed: int = 7
    # true demand
    a_base: float = 40.0
    b: float = 1.35
    k: float = -0.10
    beta: float = 0.02
    eta: float = 4.0
    seasonal_amp: float = 0.12
    weekend_uplift: float = 0.12
    holiday_effect: float = -0.25
    rain_effect: float = -0.06
    daily_sigma: float = 0.15
    hourly_cv: float = 0.5
    basket_kg: float = 0.45
    # true deterioration
    lam_ref: float = 0.20
    T_ref: float = 4.0
    Q10: float = 2.5
    shelf_life_days: float = 3.0
    handling_damage: float = 0.003
    transit_damage_max: float = 0.02
    excursion_prob_per_day: float = 1 / 45
    # supplier and accounting (known to both store and twin)
    c_mean: float = 7.0
    c_seasonal_amp: float = 0.15
    c_ar_phi: float = 0.9
    c_ar_sigma: float = 0.04
    breakpoints: tuple = (0.0, 25.0, 50.0, 100.0)
    discounts: tuple = (0.0, 0.02, 0.04, 0.06)
    K: float = 5.0
    g: float = 0.01
    f: float = 0.005
    C2: float = 0.5
    C4: float = 2.0
    storage_cap: float = 100.0
    # store practice markdown (end-of-day 30% off when overstocked)
    markdown: bool = False
    events: list = field(default_factory=lambda: list(DEFAULT_EVENTS))


@dataclass
class Action:
    price: float
    order_qty: float
    next_delivery_in_days: int = 1


class FreshStore:
    def __init__(self, cfg: StoreConfig, n_days: int):
        self.cfg, self.n_days = cfg, n_days
        self.start = date.fromisoformat(cfg.start)
        self.d = 0
        self.lots: list[dict] = []
        self.backorders: list[dict] = []
        self.lot_ctr = self.po_ctr = self.bo_ctr = 0
        self.spoiled_since_count = 0.0
        self._exogenous()

    # ------------------------------------------------------------ exogenous paths
    def _exogenous(self):
        c, n = self.cfg, self.n_days
        r = np.random.default_rng([c.seed, 0])
        days = [self.start + timedelta(days=i) for i in range(n)]
        doy = np.array([x.timetuple().tm_yday for x in days])
        self.weekday = np.array([x.weekday() for x in days])
        self.holiday = np.array([x.strftime("%m-%d") in HOLIDAYS_MMDD for x in days])
        self.outdoor_T = 10 + 10 * np.sin(2 * np.pi * (doy - 110) / 365) + r.normal(0, 2.5, n)
        self.rain_mm = np.where(r.random(n) < 0.3, r.gamma(1.5, 4.0, n), 0.0)
        dem, cost, cold = np.ones(n), np.ones(n), np.zeros(n)
        for ev in c.events:
            s, e, kind, size = int(ev[0]), int(ev[1]), ev[2], float(ev[3])
            idx = np.arange(max(s, 0), min(e, n))
            if not len(idx):
                continue
            if kind == "demand":
                ramp = np.minimum(1, np.minimum(idx - s + 1, e - idx) / 7) if e < 99999 else np.minimum(1, (idx - s + 1) / 7)
                dem[idx] *= 1 + (size - 1) * ramp
            elif kind == "cost":
                cost[idx] *= size
            elif kind == "coldfault":
                cold[idx] = size
        self.dem_event, self.cost_event, self.cold_event = dem, cost, cold
        season = 1 + c.seasonal_amp * np.sin(2 * np.pi * (doy - 80) / 365)
        week = np.where(self.weekday >= 5, 1 + c.weekend_uplift, 1.0)
        hol = np.where(self.holiday, 1 + c.holiday_effect, 1.0)
        rain = np.where(self.rain_mm > 5, 1 + c.rain_effect, 1.0)
        self.a_true = c.a_base * season * week * hol * rain * dem
        self.a_noisy = self.a_true * np.exp(r.normal(-c.daily_sigma**2 / 2, c.daily_sigma, n))
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = c.c_ar_phi * x[i - 1] + r.normal(0, c.c_ar_sigma)
        cs = 1 + c.c_seasonal_amp * np.sin(2 * np.pi * (doy - 200) / 365)
        self.wholesale = np.round(c.c_mean * cs * np.exp(x) * cost, 2)
        T = np.empty((n, 24))
        for i in range(n):
            hrs = (np.arange(24) + 4) % 24
            T[i] = c.T_ref + 0.4 * np.sin(2 * np.pi * (hrs - 14) / 24) + 0.03 * (self.outdoor_T[i] - 10) + r.normal(0, 0.25, 24)
            if r.random() < c.excursion_prob_per_day:
                h0, dur = r.integers(0, 18), r.integers(4, 10)
                T[i, h0:h0 + dur] += r.uniform(4, 7)
            if cold[i] > 0:
                T[i, 18:24] += cold[i]
        self.chiller_T = T

    def date_of(self, d):
        return self.start + timedelta(days=d)

    def pricelist(self, d):
        c, w = self.cfg, float(self.wholesale[d])
        return [{"date": str(self.date_of(d)), "tier": j + 1, "min_qty_kg": c.breakpoints[j],
                 "discount": c.discounts[j], "unit_cost": round(w * (1 - c.discounts[j]), 3),
                 "base_wholesale": w, "ordering_cost": c.K} for j in range(len(c.breakpoints))]

    def on_hand(self):
        return sum(l["qty"] for l in self.lots)

    def open_backorders(self):
        return sum(b["qty"] for b in self.backorders)

    # ------------------------------------------------------------ one day 04:00 -> 03:59
    def step(self, act: Action) -> dict:
        c, d = self.cfg, self.d
        r = np.random.default_rng([c.seed, 1, d])        # dynamics (paired across policies)
        r2 = np.random.default_rng([c.seed, 2, d])       # cosmetic fields
        shape = 1 / c.hourly_cv**2
        gam = r.standard_gamma(shape, 24) / shape         # mean-1 hourly multipliers
        wait_u = r.uniform(0.8, 1.2, 24)
        transit_u = r.uniform(0, c.transit_damage_max)
        day = self.date_of(d)
        base_dt = datetime(day.year, day.month, day.day)
        ts = lambda h: (base_dt + timedelta(hours=h)).isoformat()
        out = {k: [] for k in ("pos_hourly", "deliveries", "backorders", "waste", "stocktake", "sensors")}
        out["supplier_prices"] = self.pricelist(d)
        out["calendar"] = [{"date": str(day), "weekday": int(self.weekday[d]), "holiday": bool(self.holiday[d]),
                            "outdoor_temp_c": round(float(self.outdoor_T[d]), 1),
                            "precip_mm": round(float(self.rain_mm[d]), 1)}]
        fin = dict(revenue=0.0, purchase=0.0, ordering=0.0, holding=0.0, backorder=0.0, lost_sale=0.0)

        q_req = max(0.0, float(act.order_qty))
        q = min(q_req, max(0.0, c.storage_cap - self.on_hand() + self.open_backorders()))
        if q > 1e-6:
            self.po_ctr += 1
            j = max(i for i, bp in enumerate(c.breakpoints) if q >= bp - 1e-9)
            unit_cost = float(self.wholesale[d]) * (1 - c.discounts[j])
            damaged = q * transit_u
            self.lot_ctr += 1
            lot = {"lot_id": f"L{self.lot_ctr:05d}", "qty": q - damaged, "recv_day": d, "unit_cost": unit_cost}
            fin["purchase"] += q * unit_cost
            fin["ordering"] += c.K
            out["deliveries"].append({"timestamp": ts(4), "po_id": f"PO{self.po_ctr:05d}", "lot_id": lot["lot_id"],
                                      "qty_ordered_kg": round(q_req, 2), "qty_accepted_kg": round(q, 2),
                                      "qty_received_kg": round(q - damaged, 2), "capacity_clipped": q < q_req - 1e-6,
                                      "unit_cost": round(unit_cost, 3), "tier": j + 1,
                                      "pack_date": str(day - timedelta(days=1))})
            out["waste"].append({"date": str(day), "reason": "transit_damage", "qty_kg": round(damaged, 3)})
            for bo in list(self.backorders):
                serve = min(bo["qty"], lot["qty"])
                if serve <= 1e-9:
                    break
                lot["qty"] -= serve
                bo["qty"] -= serve
                fin["revenue"] += serve * bo["price"]
                out["pos_hourly"].append({"timestamp": ts(8), "channel": "backorder_pickup", "qty_kg": round(serve, 3),
                                          "unit_price": bo["price"], "markdown": False, "n_transactions": 1,
                                          "stockout": False})
                out["backorders"].append({"timestamp": ts(8), "bo_id": bo["bo_id"], "event": "fulfilled",
                                          "qty_kg": round(serve, 3)})
                if bo["qty"] <= 1e-9:
                    self.backorders.remove(bo)
            self.lots.append(lot)

        true_dem = lost = backlogged = 0.0
        lam_h = []
        for off in range(24):
            clock = (4 + off) % 24
            T = float(self.chiller_T[d, off])
            lam = c.lam_ref * c.Q10 ** ((T - c.T_ref) / 10)
            lam_h.append(lam)
            for l in self.lots:
                age = d - l["recv_day"] + off / 24
                fin["holding"] += l["qty"] * l["unit_cost"] * (c.g + c.f * age) / 24
                loss = l["qty"] * (1 - math.exp(-lam / 24))
                l["qty"] -= loss
                self.spoiled_since_count += loss
            fin["backorder"] += self.open_backorders() * c.C2 / 24
            I = self.on_hand()
            if clock in OPEN_HOURS:
                w = HOUR_W[clock - 8]
                price, md = act.price, False
                if c.markdown and clock >= 20 and I > 0.35 * max(q, 1):
                    price, md = round(act.price * 0.7, 2), True
                base = max(self.a_noisy[d] - c.b * price, 0.0)
                if I > 1e-9:
                    age = d - self.lots[0]["recv_day"] + off / 24
                    mu = base * w * math.exp(c.k * age) + c.beta * I / len(OPEN_HOURS)
                else:
                    mu = base * w
                D = mu * gam[off]
                true_dem += D
                sold, need = 0.0, D
                while need > 1e-9 and self.lots:
                    take = min(need, self.lots[0]["qty"])
                    self.lots[0]["qty"] -= take
                    sold += take
                    need -= take
                    if self.lots[0]["qty"] <= 1e-9:
                        self.lots.pop(0)
                fin["revenue"] += sold * price
                stockout = need > 1e-9 or self.on_hand() <= 1e-9
                if need > 1e-9:
                    tau = (act.next_delivery_in_days * 24 - off) / 24
                    wait = min(need, need / (1 + c.eta * max(tau, 0)) * wait_u[off])
                    if wait > 0.02:
                        self.bo_ctr += 1
                        self.backorders.append({"bo_id": f"B{self.bo_ctr:05d}", "qty": wait, "price": price})
                        out["backorders"].append({"timestamp": ts(clock), "bo_id": f"B{self.bo_ctr:05d}",
                                                  "event": "created", "qty_kg": round(wait, 3)})
                    else:
                        wait = 0.0
                    backlogged += wait
                    lost += need - wait
                    fin["lost_sale"] += (need - wait) * c.C4
                n_tx = int(r2.poisson(sold / c.basket_kg)) if sold > 0 else 0
                out["pos_hourly"].append({"timestamp": ts(clock), "channel": "shelf", "qty_kg": round(sold, 3),
                                          "unit_price": price, "markdown": md,
                                          "n_transactions": max(n_tx, 1 if sold > 0 else 0), "stockout": bool(stockout)})
            out["sensors"].append({"timestamp": (base_dt + timedelta(hours=4 + off)).isoformat(), "device": "CHILLER-01",
                                   "temp_c": round(T, 2), "rel_humidity": round(float(np.clip(r2.normal(92, 2), 80, 99)), 1),
                                   "door_openings": int(r2.poisson(12 * (clock in OPEN_HOURS)))})
            if clock == 22:
                out["waste"].append({"date": str(day), "reason": "spoiled", "qty_kg": round(self.spoiled_since_count, 3)})
                self.spoiled_since_count = 0.0
                expired = [l for l in self.lots if d - l["recv_day"] + 18 / 24 >= c.shelf_life_days]
                for l in expired:
                    self.lots.remove(l)
                out["waste"].append({"date": str(day), "reason": "expired",
                                     "qty_kg": round(sum(l["qty"] for l in expired), 3)})
                sold_today = sum(p["qty_kg"] for p in out["pos_hourly"])
                h = min(sold_today * c.handling_damage, self.on_hand())
                if h > 0 and self.lots:
                    self.lots[0]["qty"] -= h
                out["waste"].append({"date": str(day), "reason": "handling_damage", "qty_kg": round(h, 3)})
                out["stocktake"].append({"timestamp": ts(22), "on_hand_kg": round(self.on_hand(), 3),
                                         "n_lots": len(self.lots),
                                         "oldest_lot_age_days": round(d - self.lots[0]["recv_day"] + 0.75, 2) if self.lots else None,
                                         "open_backorders_kg": round(self.open_backorders(), 3)})
        fin = {k: round(v, 3) for k, v in fin.items()}
        fin["profit"] = round(fin["revenue"] - fin["purchase"] - fin["ordering"] - fin["holding"]
                              - fin["backorder"] - fin["lost_sale"], 3)
        out["finance"] = [{"date": str(day), **fin}]
        out["ground_truth"] = [{"date": str(day), "a_true": float(self.a_true[d]), "a_realised": float(self.a_noisy[d]),
                                "lambda_mean": float(np.mean(lam_h)), "true_demand_kg": true_dem, "lost_demand_kg": lost,
                                "backlogged_kg": backlogged, "demand_event": float(self.dem_event[d]),
                                "cost_event": float(self.cost_event[d]), "cold_fault": float(self.cold_event[d])}]
        self.d += 1
        return out


class Ledger:
    """The twin's data store: raw tables + one summary row per day."""
    TABLES = ("pos_hourly", "deliveries", "backorders", "waste", "stocktake", "sensors", "supplier_prices",
              "calendar", "finance", "ground_truth")

    def __init__(self, temp_alarm_c: float = 8.0):
        self.tables = {k: [] for k in self.TABLES}
        self.daily: list[dict] = []
        self.temp_alarm_c = temp_alarm_c

    def add(self, rec: dict, act: Action, extra: dict | None = None):
        for k in self.TABLES:
            self.tables[k].extend(rec.get(k, []))
        shelf = [p for p in rec["pos_hourly"] if p["channel"] == "shelf"]
        temps = [s["temp_c"] for s in rec["sensors"]]
        waste = {w["reason"]: w["qty_kg"] for w in rec["waste"]}
        # censoring correction: extrapolate demand from the hours that were in stock
        w_ok = sum(HOUR_W[int(p["timestamp"][11:13]) - 8] for p in shelf if not p["stockout"])
        s_ok = sum(p["qty_kg"] for p in shelf if not p["stockout"])
        shelf_sales = sum(p["qty_kg"] for p in shelf)
        bo_created = sum(b["qty_kg"] for b in rec["backorders"] if b["event"] == "created")
        demand_est = s_ok / w_ok if w_ok >= 0.15 else shelf_sales + bo_created
        row = {
            "date": rec["calendar"][0]["date"],
            "price": act.price, "order_kg": act.order_qty,
            "shelf_sales_kg": shelf_sales, "demand_est_kg": max(demand_est, shelf_sales),
            "sales_kg": sum(p["qty_kg"] for p in rec["pos_hourly"]),
            "bo_created_kg": bo_created,
            "stockout_hours": sum(p["stockout"] for p in shelf),
            "wholesale": rec["supplier_prices"][0]["base_wholesale"],
            "received_kg": sum(x["qty_received_kg"] for x in rec["deliveries"]),
            "purchased_kg": sum(x["qty_accepted_kg"] for x in rec["deliveries"]),
            "unit_cost_paid": rec["deliveries"][0]["unit_cost"] if rec["deliveries"] else None,
            "waste_kg": sum(waste.values()), "spoiled_kg": waste.get("spoiled", 0.0),
            "on_hand_kg": rec["stocktake"][0]["on_hand_kg"],
            "open_backorders_kg": rec["stocktake"][0]["open_backorders_kg"],
            "temp_mean_c": float(np.mean(temps)), "temp_max_c": float(np.max(temps)),
            "hours_above_alarm": int(sum(t > self.temp_alarm_c for t in temps)),
            **{f"fin_{k}": v for k, v in rec["finance"][0].items() if k != "date"},
            **{f"gt_{k}": v for k, v in rec["ground_truth"][0].items() if k != "date"},
        }
        if extra:
            row.update(extra)
        self.daily.append(row)
