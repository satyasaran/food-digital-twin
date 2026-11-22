"""Self-checks of model.py: closed forms vs numerical integration, Algorithm 1 vs brute force."""
import numpy as np
from scipy.integrate import quad, solve_ivp

from model import ModelParams, D_of, backlog_terms, evaluate_plan, phi, solve, unit_terms


def check_closed_forms(P, rng, trials=50):
    worst = 0.0
    for _ in range(trials):
        P = ModelParams(k=rng.uniform(-0.5, 0.5), beta=rng.uniform(0, 0.1), lam=rng.uniform(0.05, 0.5),
                        g=rng.uniform(0, 0.05), f=rng.uniform(0, 0.02), eta=rng.choice([None, rng.uniform(0, 10)]),
                        delta=rng.uniform(0.2, 1))
        L = rng.uniform(1, 10)
        t1 = rng.uniform(0.1, 1) * L
        u = unit_terms(P, np.array([t1]), L)
        w = P.omega
        sol = solve_ivp(lambda t, I: [-np.exp(P.k * t) - w * I[0]], [t1, 0], [0.0], rtol=1e-11, atol=1e-13,
                        dense_output=True)
        I = lambda t: sol.sol(t)[0]
        S = I(0.0)
        Dtau = P.lam * quad(I, 0, t1, epsabs=1e-12)[0]
        H = quad(lambda t: (P.g + P.f * t) * I(t), 0, t1, epsabs=1e-12)[0]
        x = L - t1
        dl = (lambda s: P.delta) if P.eta is None else (lambda s: 1 / (1 + P.eta * s))
        B = quad(dl, 0, x)[0]
        A = quad(lambda s: s * dl(s), 0, x)[0]
        for name, num in [("S", S), ("Dtau", Dtau), ("H", H), ("B", B), ("A", A)]:
            cf = float(u[name][0])
            worst = max(worst, abs(cf - num) / max(abs(num), 1e-9))
    return worst


def true_TAP(P, n, p, t1):
    L = P.T / n
    u = {k: float(v[0]) for k, v in unit_terms(P, np.array([t1]), L).items()}
    D = float(D_of(P, p))
    Q = D * u["Q"]
    if P.storage_cap is not None and D * u["S"] > P.storage_cap + 1e-9:
        return -np.inf
    q = list(P.breakpoints)
    j = max(i for i in range(P.J) if Q >= q[i])
    c = P.costs[j]
    W = c * (u["Q"] + u["H"]) + P.C2 * u["A"] + P.C4 * (u["x"] - u["B"])
    return n * (D * (p * u["U"] - W) - P.K) / P.T


def brute(P):
    best = -np.inf
    from model import cycle_counts
    for n in cycle_counts(P):
        L = P.T / n
        for t1 in L * np.linspace(0.02, 1, 120):
            for p in np.linspace(5, 29, 600):
                v = true_TAP(P, n, p, t1)
                best = max(best, v)
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("max relative error, closed forms vs quadrature:", f"{check_closed_forms(ModelParams(), rng):.2e}")
    for cap in (100.0, 40.0, None):
        for eta in (4.0, None):
            P = ModelParams(storage_cap=cap, eta=eta)
            plan = solve(P)
            bf = brute(P)
            print(f"cap={cap}, eta={eta}: Algorithm 1 TAP={plan['TAP']:.4f} (n={plan['n']}, p={plan['p']:.3f}, "
                  f"nu={plan['nu']:.3f}, Q={plan['Q']:.2f}), brute force={bf:.4f}, gap={plan['TAP'] - bf:+.4f}, "
                  f"Thm1={plan['thm1_ok']}")
    # Phi vs finite difference
    P = ModelParams()
    plan = solve(P)
    L, p, c = plan["L"], plan["p"], plan["c"]
    from model import G_profile
    def pi_fixed(t):
        u = {k: float(v[0]) for k, v in unit_terms(P, np.array([t]), L).items()}
        D = float(D_of(P, p))
        W = c * (u["Q"] + u["H"]) + P.C2 * u["A"] + P.C4 * (u["x"] - u["B"])
        return D * (p * u["U"] - W) - P.K
    t = 0.6 * L
    fd = (pi_fixed(t + 1e-5) - pi_fixed(t - 1e-5)) / 2e-5 / float(D_of(P, p))
    print(f"Phi closed form {phi(P, p, c, t, L):.6f} vs finite difference {fd:.6f}")
