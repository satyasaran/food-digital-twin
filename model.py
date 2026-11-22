"""
model.py - optimisation engine of the digital twin.

Implements the manuscript's model for one deteriorating item:
  * closed-form inventory, deterioration, holding-cost and backlog terms (Sec. 4)
  * closed-form optimal price per discount bracket (Proposition 1)
  * bracket feasibility as a price interval, closed brackets (Lemmas 1-2)
  * exact-reduction algorithm: enumerate n and j, grid + bounded Brent in t1 (Algorithm 1)
  * concavity certificate (Theorem 1) and shortage test (Proposition 2)
  * break-even discount for ordering exactly a breakpoint (Proposition 4)
plus an operational storage cap S <= cap, imposed exactly as an extra lower
bound on the price (S = D(p) S~(t1) is decreasing in p).

Units used by the twin: days, kg, CNY. All rates are per day.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class ModelParams:
    demand: str = "linear"        # "linear": D=a-bp ; "iso": D=a p^-r
    a: float = 40.0
    b: float = 1.35
    r: float = 1.5
    k: float = -0.10              # lot-age coefficient (per day)
    beta: float = 0.02            # display coefficient (per day)
    lam: float = 0.20             # deterioration rate (per day)
    delta: float = 0.5            # constant backlogged fraction (used if eta is None)
    eta: float | None = 4.0       # impatience: delta(tau) = 1/(1+eta tau)
    g: float = 0.01               # holding: c_j (g + f t) per kg per day
    f: float = 0.005
    K: float = 5.0                # ordering cost per order
    C2: float = 0.5               # backorder cost per kg per day
    C4: float = 2.0               # lost-sale cost per kg
    T: float = 28.0               # planning horizon (days)
    breakpoints: tuple = (0.0, 25.0, 50.0, 100.0)   # q_0=0 < q_1 < ... ; q_J = inf
    costs: tuple = (7.0, 6.86, 6.72, 6.58)          # c_1 > c_2 > ... (CNY/kg)
    storage_cap: float | None = 100.0               # max initial stock S (kg)
    integer_cycles: bool = True   # cycle length L = T/n must be a whole number of days
    n_max: int = 28
    M: int = 200                  # grid points in t1

    @property
    def omega(self):
        return self.lam + self.beta

    @property
    def J(self):
        return len(self.costs)


# ---------------------------------------------------------------- helpers
def _E1(x, t):
    """(e^{x t} - 1)/x with the limit t as x -> 0."""
    t = np.asarray(t, float)
    if abs(x) < 1e-10:
        return t + x * t**2 / 2
    return np.expm1(x * t) / x


def _Mint(x, t, g, f):
    """M(x) = int_0^t (g + f u) e^{x u} du."""
    t = np.asarray(t, float)
    if abs(x) < 1e-8:
        return g * (t + x * t**2 / 2) + f * (t**2 / 2 + x * t**3 / 3)
    E = np.exp(x * t)
    Jint = (t * E - (E - 1) / x) / x
    return g * (E - 1) / x + f * Jint


def backlog_terms(P: ModelParams, x):
    """B~(x) = int_0^x delta, A~(x) = int_0^x tau delta, delta(x)."""
    x = np.asarray(x, float)
    if P.eta is None:
        return P.delta * x, P.delta * x**2 / 2, np.full_like(x, P.delta)
    e = P.eta
    if e < 1e-12:
        return x.copy(), x**2 / 2, np.ones_like(x)
    lg = np.log1p(e * x)
    return lg / e, x / e - lg / e**2, 1 / (1 + e * x)


def unit_terms(P: ModelParams, t1, L):
    """Per-unit-of-D(p) quantities for stock-out time t1 and cycle length L."""
    t1 = np.asarray(t1, float)
    k, w, lam = P.k, P.omega, P.lam
    kw = k + w if abs(k + w) > 1e-10 else 1e-10
    S = _E1(kw, t1)
    int_I = (np.exp(k * t1) * _E1(w, t1) - _E1(k, t1)) / kw
    Dtau = lam * int_I
    H = (np.exp(kw * t1) * _Mint(-w, t1, P.g, P.f) - _Mint(k, t1, P.g, P.f)) / kw
    x = np.maximum(L - t1, 0.0)
    B, A, _ = backlog_terms(P, x)
    U = S - Dtau + B
    Qt = S + B
    return dict(S=S, Dtau=Dtau, H=H, B=B, A=A, U=U, Q=Qt, x=x)


def D_of(P, p):
    p = np.asarray(p, float)
    if P.demand == "linear":
        return np.maximum(P.a - P.b * p, 0.0)
    return P.a * np.power(np.maximum(p, 1e-12), -P.r)


def D_inv(P, y):
    """Price at which D(p) = y (y may be inf / 0)."""
    y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if P.demand == "linear":
            return np.where(np.isinf(y), 0.0, (P.a - y) / P.b)
        return np.where(np.isinf(y), 0.0, np.where(y <= 0, np.inf, np.power(P.a / np.maximum(y, 1e-300), 1 / P.r)))


def p_opt(P, W, U):
    """Proposition 1: unconstrained optimal price for a bracket."""
    if P.demand == "linear":
        return 0.5 * (P.a / P.b + W / U)
    return P.r / (P.r - 1) * W / U


def G_profile(P: ModelParams, j: int, t1, L):
    """Lemma 1-2: price-optimised profit per cycle in bracket j (0-based), with price."""
    u = unit_terms(P, t1, L)
    c = P.costs[j]
    W = c * (u["Q"] + u["H"]) + P.C2 * u["A"] + P.C4 * (u["x"] - u["B"])
    p0 = p_opt(P, W, u["U"])
    q = list(P.breakpoints) + [np.inf]
    lo = D_inv(P, q[j + 1] / u["Q"]) if np.isfinite(q[j + 1]) else np.zeros_like(u["Q"])
    if q[j] > 0:
        hi = D_inv(P, q[j] / u["Q"])
    else:
        hi = np.full_like(u["Q"], P.a / P.b if P.demand == "linear" else np.inf)
    lo = np.maximum(lo, 1e-9)
    if P.storage_cap is not None:
        lo = np.maximum(lo, D_inv(P, P.storage_cap / np.maximum(u["S"], 1e-12)))
    feas = lo <= hi
    p = np.clip(p0, lo, np.where(np.isinf(hi), np.maximum(p0, lo), hi))
    G = D_of(P, p) * (p * u["U"] - W) - P.K
    G = np.where(feas, G, -np.inf)
    return G, p, u, W


def cycle_counts(P: ModelParams):
    if P.integer_cycles and abs(P.T - round(P.T)) < 1e-9:
        T = int(round(P.T))
        return [n for n in range(1, min(P.n_max, T) + 1) if T % n == 0]
    return list(range(1, P.n_max + 1))


# ---------------------------------------------------------------- Algorithm 1
def _neg(v):
    v = float(v)
    return -v if np.isfinite(v) else 1e18


def solve(P: ModelParams, keep_profiles: bool = False) -> dict:
    best = None
    per_n = []
    for n in cycle_counts(P):
        L = P.T / n
        grid = L * np.arange(1, P.M + 1) / P.M
        best_n = None
        for j in range(P.J):
            G, _, _, _ = G_profile(P, j, grid, L)
            m = int(np.argmax(G))
            if not np.isfinite(G[m]):
                continue
            lo, hi = grid[max(m - 1, 0)] if m > 0 else 1e-6 * L, grid[min(m + 1, P.M - 1)]
            res = minimize_scalar(lambda t: _neg(G_profile(P, j, np.array([t]), L)[0][0]),
                                  bounds=(lo, hi), method="bounded", options={"xatol": 1e-7 * L})
            t1, g = (float(res.x), -float(res.fun)) if np.isfinite(res.fun) and -res.fun > G[m] else (grid[m], G[m])
            tap = n * g / P.T
            if best_n is None or tap > best_n["TAP"]:
                best_n = dict(n=n, j=j, t1=t1, TAP=tap)
        if best_n is None:
            continue
        per_n.append(best_n)
        if best is None or best_n["TAP"] > best["TAP"]:
            best = best_n
    if best is None:
        raise ValueError("No feasible plan (check prices, capacity and discount schedule).")
    plan = evaluate_plan(P, best["n"], best["t1"], best["j"])
    plan["TAP_by_n"] = per_n
    if keep_profiles:
        plan["profiles"] = profiles(P, best["n"])
    return plan


def profiles(P, n, M=300):
    L = P.T / n
    grid = L * np.arange(1, M + 1) / M
    out = {"nu": grid / L}
    for j in range(P.J):
        G, p, _, _ = G_profile(P, j, grid, L)
        out[j] = np.where(np.isfinite(G), n * G / P.T, np.nan)
    return out


def evaluate_plan(P: ModelParams, n, t1, j_hint=None) -> dict:
    """Plan details evaluated with the TRUE discount schedule (Lemma 2)."""
    L = P.T / n
    j = j_hint
    G, p, u, W = G_profile(P, j, np.array([t1]), L)
    p = float(p[0])
    D = float(D_of(P, p))
    u = {k: float(v[0]) if np.ndim(v) else float(v) for k, v in u.items()}
    Q = D * u["Q"]
    q = list(P.breakpoints) + [np.inf]
    j_true = max(i for i in range(P.J) if Q >= q[i] - 1e-9)
    c = P.costs[j_true]
    W = c * (u["Q"] + u["H"]) + P.C2 * u["A"] + P.C4 * (u["x"] - u["B"])
    pi = D * (p * u["U"] - W) - P.K
    plan = dict(n=n, L=L, t1=t1, nu=t1 / L, j=j_true, c=c, p=p, D=D, Q=Q, S=D * u["S"],
                TAP=n * pi / P.T, TP=n * pi, profit_cycle=pi,
                sold_cycle=D * u["U"], deteriorated_cycle=D * u["Dtau"],
                backlog_cycle=D * u["B"], lost_cycle=D * (u["x"] - u["B"]),
                waste_share=u["Dtau"] / u["S"] if u["S"] > 0 else 0.0,
                revenue_cycle=p * D * u["U"], purchase_cycle=c * Q, holding_cycle=c * D * u["H"],
                backorder_cost_cycle=P.C2 * D * u["A"], lost_cost_cycle=P.C4 * D * (u["x"] - u["B"]),
                at_breakpoint=any(abs(Q - b) < 1e-6 * max(1, b) for b in P.breakpoints[1:]))
    plan.update(theorem1(P, p, c, L))
    plan["Phi_L"] = phi(P, p, c, L, L)
    plan["shortage_profitable"] = plan["Phi_L"] < 0
    plan["break_even"] = break_even(P, plan, u)
    plan["storage_binding"] = P.storage_cap is not None and plan["S"] > P.storage_cap * (1 - 1e-6)
    return plan


# ---------------------------------------------------------------- theory checks
def _Gamma(P, p, c, t):
    w = P.omega
    return p - c - p * P.lam / w * (1 - math.exp(-w * t)) - c * float(_Mint(-w, t, P.g, P.f))


def phi(P, p, c, t1, L):
    """Proposition 2: Phi_j(t1) = D^-1 d pi / d t1."""
    w, k = P.omega, P.k
    E = math.exp((k + w) * t1)
    x = L - t1
    _, _, d = backlog_terms(P, np.array([x]))
    d = float(d[0])
    return E * _Gamma(P, p, c, t1) - (p - c) * d + P.C2 * x * d + P.C4 * (1 - d)


def theorem1(P, p, c, L):
    kw, w = P.k + P.omega, P.omega
    if p <= c:
        return dict(thm1_ok=False, thm1_case="p <= c_j: theorem not applicable")
    if kw < 0:
        ok = _Gamma(P, p, c, L) >= 0
        return dict(thm1_ok=ok, thm1_case=f"(i) k+omega<0, Gamma(L)={_Gamma(P, p, c, L):.3f}")
    lhs, rhs = kw * (p - c), math.exp(-w * L) * (p * P.lam + c * P.g)
    return dict(thm1_ok=lhs <= rhs, thm1_case=f"(ii) k+omega>0: {lhs:.3f} <= {rhs:.3f}")


def break_even(P, plan, u):
    """Proposition 4 for the next breakpoint above the planned lot."""
    q = list(P.breakpoints)
    j = plan["j"]
    if j + 1 >= P.J:
        return None
    qj = q[j + 1]
    Qt, H, U, x = u["Q"], u["H"], u["U"], u["x"]
    if P.demand == "linear" and qj / Qt >= P.a:
        return None
    cj = P.costs[j]
    B, A, _ = backlog_terms(P, np.array([x]))
    B, A = float(B[0]), float(A[0])
    R = P.C2 * A + P.C4 * (x - B)
    W = cj * (Qt + H) + R
    if P.demand == "linear":
        Pi0 = (P.a * U - P.b * W) ** 2 / (4 * P.b * U) if W / U < P.a / P.b else 0.0
    else:
        r = P.r
        Pi0 = P.a * (r - 1) ** (r - 1) * r ** (-r) * U**r * W ** (1 - r)
    pb = float(D_inv(P, qj / Qt))
    c_dag = (pb * U - R - (Qt / qj) * Pi0) / (Qt + H)
    d_dag = 1 - c_dag / cj
    d_off = 1 - P.costs[j + 1] / cj
    return dict(breakpoint=qj, d_dagger=d_dag, d_offered=d_off, meet=d_off >= d_dag)


def inventory_path(P, plan, npts=200):
    """I(t) over one cycle of the plan (negative = backlog)."""
    L, t1, D = plan["L"], plan["t1"], plan["D"]
    k, w = P.k, P.omega
    t = np.linspace(0, L, npts)
    kw = k + w
    pos = D / kw * (np.exp(k * t1 + w * (t1 - t)) - np.exp(k * t))
    Bx, _, _ = backlog_terms(P, np.array([L - t1]))
    Bt, _, _ = backlog_terms(P, np.maximum(L - t, 0))
    neg = -D * (Bx[0] - Bt)
    return t, np.where(t <= t1, pos, neg)


def schedule_from_discounts(base_cost, breakpoints, discounts):
    return tuple(round(base_cost * (1 - d), 6) for d in discounts)


def with_(P, **kw):
    return replace(P, **kw)
