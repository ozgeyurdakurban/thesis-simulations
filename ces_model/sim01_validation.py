"""Validation of the solver (Table: validation).  Three independent checks.
 (1) best response: numerical maximisation of the CES utility vs the formula;
 (2) closed forms: K_n and S_n formulas vs the iteration, on the interior;
 (3) uniqueness: least and greatest equilibrium coincide on the draws.
"""
import numpy as np, os, csv
from scipy.optimize import minimize_scalar
from ces_net import *

rng = np.random.default_rng(20261026)
os.makedirs("data", exist_ok=True)
out = {}

# (1) best response against direct utility maximisation
errs = []
for _ in range(3000):
    kind = rng.choice(["complete", "star"]); n = int(rng.integers(3, 8))
    G = adjacency(kind, n); i = int(rng.integers(0, n))
    beta = rng.uniform(1.05, 1.95); e = 15.0
    sigma = np.exp(rng.uniform(np.log(.25), np.log(5))); lam = rng.choice([0, 1, rng.uniform(0, 3)])
    B = rng.uniform(.01, .6); S = rng.uniform(0, e) * G[i].sum()
    d, th, Lam = structure(G, beta)
    pi = share(B, price(th, Lam, lam)[i], sigma)
    br = min(e, pi * e / th[i] + pi * beta / (d[i] - beta) * S)
    res = minimize_scalar(lambda c: -utility(c, S, d[i], th[i], Lam[i], B, beta, sigma, lam, e),
                          bounds=(1e-9, e), method="bounded", options={"xatol": 1e-11})
    errs.append(abs(res.x - br))
out["best_response_max_abs_error"] = max(errs); out["best_response_draws"] = len(errs)

# (2) closed forms on the interior
e1 = e2 = 0.0; k1 = k2 = 0
for _ in range(3000):
    n = int(rng.integers(3, 9)); beta = rng.uniform(1.05, 1.95); e = 15.0
    sigma = np.exp(rng.uniform(np.log(.3), np.log(4))); lam = rng.uniform(0, 3); B = rng.uniform(.01, .3)
    c = equilibrium(adjacency("complete", n), np.full(n, B), beta, sigma, lam, e)
    if c.max() < e - 1e-8:
        e1 = max(e1, abs(c[0] - cK_closed(n, B, beta, sigma, lam, e))); k1 += 1
    c = equilibrium(adjacency("star", n), np.full(n, B), beta, sigma, lam, e)
    if c.max() < e - 1e-8:
        cC, cL = star_closed(n, B, beta, sigma, lam, e)
        e2 = max(e2, abs(c[0] - cC), abs(c[1] - cL)); k2 += 1
out.update(K_closed_max_abs_error=e1, K_closed_interior_draws=k1,
           star_closed_max_abs_error=e2, star_closed_interior_draws=k2)

# (3) uniqueness / mixed regimes
gap = 0.0; corners = 0; R = 4000
for kind, n in [("complete", 3), ("star", 3), ("star", 5), ("cycle", 6)]:
    G = adjacency(kind, n)
    B = rng.beta(2, 12, size=(R, n)) * 1.5
    B = np.clip(B, .005, .9)
    sigma = np.exp(rng.uniform(np.log(.3), np.log(4), size=(R, 1)))
    lo, hi = equilibrium(G, B, 1.5, sigma, 1.0, 15.0, return_both=True)
    gap = max(gap, np.abs(hi - lo).max()); corners += int((hi > 15 - 1e-9).any(1).sum())
out["lo_hi_max_gap"] = gap; out["draws_with_corner"] = corners; out["uniqueness_draws"] = 4 * R

with open("data/validation.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["check", "value"])
    for k, v in out.items(): w.writerow([k, f"{v:.3e}" if isinstance(v, float) else v])
for k, v in out.items(): print(k, v)
assert out["best_response_max_abs_error"] < 1e-4 and e1 < 1e-8 and e2 < 1e-8 and gap < 1e-8
print("PASS")
