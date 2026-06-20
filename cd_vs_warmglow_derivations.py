"""
================================================================================
 Equilibrium comparison: impact-based Cobb--Douglas vs. warm-glow (baseline-T)
 -- Altruism in Networked Public Good Games
================================================================================

Symbolic verification of the results in the subsection
"Equilibrium Comparison with the Warm-Glow Model" (fully connected, Cobb--Douglas).

Two Cobb--Douglas utilities, sharing X_i and the exponents (A,B), A+B=1:
    warm-glow:     U^WG = X^A (T + c)^B ,   baseline T >= 0
    impact-based:  U^IB = X^A (lambda c)^B

Verified facts:
    c*_WG = (B n e - A (n-beta) T) / D ,   D = n - beta - B beta (n-1)
    c*_IB = B n e / D                       (= warm-glow at T=0)
    c*_IB - c*_WG = A (n-beta) T / D > 0     (impact-based gives strictly more)
    dc*_WG/dT = -A (n-beta) / D < 0          (baseline crowds out giving)
    existence condition identical: beta < n/(1+B(n-1))   (D is T-free)

Requirements:  Python 3,  sympy   (pip install sympy)
Run:           python cd_vs_warmglow_derivations.py
================================================================================
"""

import sympy as sp

# ------------------------------------------------------------------------------
# Symbols  (A+B=1 imposed: A = 1 - B)
# ------------------------------------------------------------------------------
e, beta, n = sp.symbols('e beta n', positive=True)   # endowment, return, group size
B          = sp.symbols('B', positive=True)          # altruism (A = 1 - B)
T          = sp.symbols('T', positive=True)          # warm-glow baseline
c          = sp.symbols('c', positive=True)          # symmetric contribution
A = 1 - B

theta = 1 - beta / n
D     = n - beta - B * beta * (n - 1)                # common denominator


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def is_zero(expr):
    """Identically zero? Symbolic simplification, then numeric fallback over the
    admissible domain (1 < beta < n, 0 < B < 1, T,e > 0)."""
    if sp.simplify(expr) == 0:
        return True
    pts = [
        {'e': 10, 'beta': 2, 'n': 4, 'B': sp.Rational(3, 10), 'T': 2, 'c': sp.Rational(3, 2)},
        {'e': 8, 'beta': sp.Rational(3, 2), 'n': 5, 'B': sp.Rational(1, 2), 'T': sp.Rational(5, 2), 'c': 1},
    ]
    for pt in pts:
        sub = {s: pt[s.name] for s in expr.free_symbols if s.name in pt}
        if complex(sp.N(expr.subs(sub))).__abs__() > 1e-9:
            return False
    return True


# ------------------------------------------------------------------------------
# Warm-glow symmetric equilibrium  (Proposition: warm-glow symmetric equilibrium)
# ------------------------------------------------------------------------------
# Symmetric FOC: -A theta / X + B / (T + c) = 0 ,  with C_{-i} = (n-1) c.
banner("Warm-glow symmetric equilibrium")
Xc   = e - theta * c + (beta / n) * (n - 1) * c          # X at symmetric profile
c_wg = sp.solve(sp.Eq(-A * theta / Xc + B / (T + c), 0), c)[0]
c_wg_claim = (B * n * e - A * (n - beta) * T) / D
print("c*_WG =", sp.simplify(c_wg))
print("matches [B n e - A(n-beta) T]/D ? ->", is_zero(c_wg - c_wg_claim))

# ------------------------------------------------------------------------------
# Impact-based symmetric equilibrium and the comparison
# ------------------------------------------------------------------------------
banner("Comparison (Proposition: impact-based vs. warm-glow)")
c_ib = B * n * e / D                                     # = c*_WG at T=0
gap  = sp.simplify(c_ib - c_wg_claim)
print("(i)  c*_IB - c*_WG =", gap)
print("     equals A(n-beta) T / D (>0)? ->", is_zero(gap - A * (n - beta) * T / D))
print("     coincide at T=0? ->", is_zero(c_ib - c_wg_claim.subs(T, 0)))

dWG_dT = sp.simplify(sp.diff(c_wg_claim, T))
print("(ii) dc*_WG/dT =", dWG_dT)
print("     equals -A(n-beta)/D (<0, crowding-out)? ->", is_zero(dWG_dT + A * (n - beta) / D))

print("(iii) existence denominator D is T-free (same condition both models)? ->",
      sp.simplify(sp.diff(D, T)) == 0)
print("      existence condition: beta < n/(1+B(n-1))  [D>0]")

# ------------------------------------------------------------------------------
# Numerical illustration of the gap (n=4, beta=2, e=10, B=0.3, T=2)
# ------------------------------------------------------------------------------
banner("Numerical illustration of the gap")
sub = {n: 4, beta: 2, e: 10, B: sp.Rational(3, 10), T: 2}
cib_v = float(c_ib.subs(sub))
cwg_v = float(c_wg_claim.subs(sub))
print(f"  c*_IB = {cib_v:.4f}   c*_WG = {cwg_v:.4f}   gap = {cib_v - cwg_v:.4f}")
print(f"  impact-based > warm-glow ? -> {cib_v > cwg_v}")

# ------------------------------------------------------------------------------
# Asymmetric comparison (heterogeneous altruism): T-removal reshapes the distribution
# ------------------------------------------------------------------------------
# WG optimality  c_i = kappa_i X_i - T   vs   IB  c_i = kappa_i X_i.
# Closed forms (full connectivity), phi_i = kappa_i/(1+kappa_i), mu = 1-(beta/N)S,
# Q = sum_j 1/(1+kappa_j):
#   c_i^IB = (e/mu) phi_i ,    c_i^WG = phi_i (e - (beta/N) T Q)/mu - T/(1+kappa_i),
#   gap_i  = c_i^IB - c_i^WG = T( (beta/N) phi_i Q / mu + 1/(1+kappa_i) ) > 0.
banner("Asymmetric comparison (heterogeneous altruism)")

def asym(N):
    th = 1 - beta / N
    Bs = sp.symbols(f'B1:{N+1}', positive=True)
    cs = sp.symbols(f'c1:{N+1}', positive=True)
    kap = [Bi / ((1 - Bi) * th) for Bi in Bs]
    Cc = sum(cs)
    X = [e - cs[i] + (beta / N) * Cc for i in range(N)]
    wg = sp.solve([sp.Eq(cs[i], kap[i] * X[i] - T) for i in range(N)], list(cs), dict=True)[0]
    ib = sp.solve([sp.Eq(cs[i], kap[i] * X[i])     for i in range(N)], list(cs), dict=True)[0]
    phi = [kap[i] / (1 + kap[i]) for i in range(N)]
    Q = sum(1 / (1 + kap[i]) for i in range(N))
    S = sum(phi); mu = 1 - (beta / N) * S
    wg_claim = [phi[i] * (e - (beta / N) * T * Q) / mu - T / (1 + kap[i]) for i in range(N)]
    ib_claim = [(e / mu) * phi[i] for i in range(N)]
    gap_claim = [T * ((beta / N) * phi[i] * Q / mu + 1 / (1 + kap[i])) for i in range(N)]
    ok_wg  = all(sp.simplify(wg[cs[i]] - wg_claim[i]) == 0 for i in range(N))
    ok_ib  = all(sp.simplify(ib[cs[i]] - ib_claim[i]) == 0 for i in range(N))
    ok_gap = all(sp.simplify((ib[cs[i]] - wg[cs[i]]) - gap_claim[i]) == 0 for i in range(N))
    return Bs, cs, wg, ib, ok_wg, ok_ib, ok_gap

# n = 2 (symbolic)
Bs2, cs2, wg2, ib2, okwg2, okib2, okgap2 = asym(2)
print("n=2 WG closed form ok? ->", okwg2)
print("n=2 IB closed form ok? ->", okib2)
print("n=2 gap = T((beta/N)phi Q/mu + 1/(1+kappa)) ok? ->", okgap2)
g1 = ib2[cs2[0]] - wg2[cs2[0]]; g2 = ib2[cs2[1]] - wg2[cs2[1]]
fac = sp.simplify(sp.factor((g1 - g2) / (Bs2[0] - Bs2[1])))
print("n=2 (gap1-gap2)/(B1-B2) =", fac, " (>0 on feasible region => more altruistic gains more)")

# n = 3 (closed form + numeric amplification)
Bs3, cs3, wg3, ib3, okwg3, okib3, okgap3 = asym(3)
print("n=3 WG/IB/gap closed forms ok? ->", okwg3 and okib3 and okgap3)
pos_ok = amp_ok = rank_ok = True
for bv in [sp.Rational(12, 10), sp.Rational(15, 10)]:
    for Tv in [1, 2]:
        for trip in [(0.1, 0.2, 0.3), (0.05, 0.15, 0.25), (0.2, 0.1, 0.05)]:
            v = {Bs3[0]: sp.Rational(str(trip[0])), Bs3[1]: sp.Rational(str(trip[1])),
                 Bs3[2]: sp.Rational(str(trip[2])), beta: bv, T: Tv, e: 15}
            cib = [float(ib3[cs3[i]].subs(v)) for i in range(3)]
            cwg = [float(wg3[cs3[i]].subs(v)) for i in range(3)]
            if all(0 < x < 15 for x in cib + cwg):
                pos_ok = pos_ok and all(cib[i] > cwg[i] for i in range(3))
                amp_ok = amp_ok and (max(cib) - min(cib)) >= (max(cwg) - min(cwg)) - 1e-9
                gaps = [cib[i] - cwg[i] for i in range(3)]
                rank_ok = rank_ok and (
                    sorted(range(3), key=lambda i: trip[i]) == sorted(range(3), key=lambda i: gaps[i]))
print("n=3 IB>WG for all agents (numeric):", pos_ok)
print("n=3 dispersion (max-min) amplified in IB vs WG (numeric):", amp_ok)
print("n=3 gap ranked by B -- more altruistic gains more (numeric):", rank_ok)

# ------------------------------------------------------------------------------
# Automated consistency checks
# ------------------------------------------------------------------------------
banner("Automated checks")
checks = {
    "WG symmetric c*":        is_zero(c_wg - c_wg_claim),
    "IB - WG gap":            is_zero(gap - A * (n - beta) * T / D),
    "IB = WG at T=0":         is_zero(c_ib - c_wg_claim.subs(T, 0)),
    "WG crowd-out sign":      is_zero(dWG_dT + A * (n - beta) / D),
    "existence D is T-free":  sp.simplify(sp.diff(D, T)) == 0,
    "numeric gap positive":   bool(cib_v > cwg_v),
    "asym n=2 WG form":       okwg2,
    "asym n=2 IB form":       okib2,
    "asym n=2 gap form":      okgap2,
    "asym n=3 forms":         okwg3 and okib3 and okgap3,
    "asym n=3 IB>WG":         pos_ok,
    "asym n=3 dispersion amplified": amp_ok,
    "asym n=3 gap ranked by B":      rank_ok,
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
assert all(checks.values()), "A derivation failed its consistency check."
print("\nAll checks passed.")
