"""
================================================================================
 Cobb--Douglas equilibrium derivations -- Altruism in Networked Public Good Games
================================================================================

Symbolic derivation and verification of the Cobb--Douglas (impact-based)
equilibrium analysis for the FULLY CONNECTED network. Reproduces, step by step,
the results in the "Equilibrium Analysis / Cobb--Douglas Specification" section.

Cobb--Douglas is the primary model: the altruism parameter B_i enters
equilibrium contributions DIRECTLY, through the explicit coefficient
    kappa_i = B_i / (A_i theta).

Map to the thesis:
    Lemma   -> Cobb--Douglas optimality and lambda-irrelevance  (c_i = kappa_i X_i)
    Step 1  -> Proposition (interior best response)
    Step 2  -> Proposition (symmetric interior Nash) + feasibility + comp. statics
    Step 3  -> Proposition (asymmetric Nash, general n; verified at n=2,3)
               + altruism ordering + numerical illustration
    Remark  -> Normalization A+B=1 is without loss of generality (scale invariance)

The equilibrium comparison with the warm-glow (baseline-T) model is verified in a
separate script, cd_vs_warmglow_derivations.py.

Requirements:  Python 3,  sympy   (pip install sympy)
Run:           python cd_equilibrium_derivations.py
================================================================================
"""

import sympy as sp

# ------------------------------------------------------------------------------
# Symbols
# ------------------------------------------------------------------------------
e, beta, n = sp.symbols('e beta n', positive=True)   # endowment, return, group size
A, B       = sp.symbols('A B', positive=True)        # CD exponents; B = altruism
c_i, C_mi, C, c = sp.symbols('c_i C_mi C c', positive=True)
lam_s      = sp.symbols('lambda', positive=True)     # generic lambda (for irrelevance test)


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def is_zero(expr):
    """Identically zero? Symbolic powsimp first, then numeric fallback over the
    admissible domain (1 < beta < n, A,B,vars > 0)."""
    reduced = sp.simplify(sp.powsimp(sp.expand_power_base(expr, force=True), force=True))
    if reduced == 0:
        return True
    pts = [
        {'e': 10, 'beta': 2, 'n': 4, 'A': sp.Rational(7, 10), 'B': sp.Rational(3, 10),
         'lambda': sp.Rational(8, 5), 'c_i': 2, 'C_mi': 3, 'C': 6, 'c': sp.Rational(3, 2),
         'A1': sp.Rational(7, 10), 'B1': sp.Rational(3, 10), 'A2': sp.Rational(1, 2),
         'B2': sp.Rational(1, 2), 'A3': sp.Rational(4, 5), 'B3': sp.Rational(1, 5),
         'kappa': sp.Rational(1, 2), 'k1': sp.Rational(2, 5), 'k2': sp.Rational(7, 10),
         'c1': sp.Rational(6, 5), 'c2': sp.Rational(9, 5), 'c3': 1, 't': sp.Rational(5, 2)},
        {'e': 8, 'beta': sp.Rational(3, 2), 'n': 5, 'A': sp.Rational(1, 2), 'B': sp.Rational(1, 2),
         'lambda': sp.Rational(9, 7), 'c_i': 1, 'C_mi': 2, 'C': 4, 'c': 1,
         'A1': sp.Rational(3, 5), 'B1': sp.Rational(2, 5), 'A2': sp.Rational(3, 4),
         'B2': sp.Rational(1, 4), 'A3': sp.Rational(9, 10), 'B3': sp.Rational(1, 10),
         'kappa': sp.Rational(3, 5), 'k1': sp.Rational(1, 5), 'k2': sp.Rational(4, 5),
         'c1': sp.Rational(1, 2), 'c2': sp.Rational(3, 2), 'c3': sp.Rational(7, 10), 't': sp.Rational(3, 4)},
    ]
    for pt in pts:
        sub = {s: pt[s.name] for s in expr.free_symbols if s.name in pt}
        if complex(sp.N(expr.subs(sub))).__abs__() > 1e-9:
            return False
    return True


# ------------------------------------------------------------------------------
# Step 0:  Primitives (fully connected)  and the lambda-irrelevance property
# ------------------------------------------------------------------------------
theta = 1 - beta / n
lam   = 1 + (n - 1) * beta / n
X_i   = e - c_i + (beta / n) * (c_i + C_mi)     # = e - theta c_i + (beta/n) C_{-i}

banner("Step 0  -  Primitives and lambda-irrelevance")
# U = X^A (lambda c)^B ; show d(log U)/dc_i does NOT depend on lambda.
logU = A * sp.log(X_i) + B * sp.log(lam_s * c_i)
foc  = sp.diff(logU, c_i)
print("d(logU)/dc_i =", sp.simplify(foc))
print("independent of lambda?  d/dlambda = 0 ->", sp.simplify(sp.diff(foc, lam_s)) == 0)

# ------------------------------------------------------------------------------
# Step 1:  Best response   (FOC -> c_i = kappa_i X_i ,  kappa_i = B_i/(A_i theta))
# ------------------------------------------------------------------------------
kappa = B / (A * theta)
banner("Step 1  -  Interior best response")
# FOC  -A theta / X + B / c = 0   <=>   c_i = kappa X_i
print("FOC <=> c_i = kappa X_i (kappa=B/(A theta))? ->",
      is_zero(foc.subs(lam_s, 1) - (B / c_i - A * theta / X_i)) and
      is_zero(B * X_i - A * theta * (kappa * X_i)))   # identity check on kappa
BR  = sp.solve(sp.Eq(c_i, kappa * (e - theta * c_i + (beta / n) * C_mi)), c_i)[0]
BRc = kappa * (e + (beta / n) * C_mi) / (1 + kappa * theta)
print("BR = kappa(e+(beta/n)C_-i)/(1+kappa theta)? ->", is_zero(BR - BRc))
print("strategic complementarity dBR/dC_-i =",
      sp.simplify(sp.diff(BRc, C_mi)), " (> 0)")

# ------------------------------------------------------------------------------
# Step 2:  Symmetric Nash equilibrium  (homogeneous altruism, general n)
# ------------------------------------------------------------------------------
banner("Step 2  -  Symmetric interior Nash equilibrium")
cstar    = sp.solve(sp.Eq(c, kappa * (e + (beta / n) * (n - 1) * c) / (1 + kappa * theta)), c)[0]
form_kap = kappa * e / (1 + kappa * (1 - beta))                 # kappa form (general A,B)
form_AB  = B * n * e / (n - beta - B * beta * (n - 1))          # explicit (A+B=1)
print("c* matches kappa form? ->", is_zero(cstar - form_kap))
print("kappa form == B-form under A=1-B? ->", is_zero(form_kap.subs(A, 1 - B) - form_AB))

# Feasibility
print("interiority c*<e <=> kappa<1/beta :  (1+kappa(1-beta))-kappa =",
      sp.simplify(1 + kappa * (1 - beta) - kappa), " (>0 <=> kappa*beta<1)")
print("existence D>0 <=> beta < n/(1+B(n-1)) ; D = n-beta-B*beta*(n-1)")

# Comparative statics on the explicit form (A = 1 - B)
banner("Step 2  -  Comparative statics (symmetric)")
D   = n - beta - B * beta * (n - 1)
csB = B * n * e / D
dB  = sp.simplify(sp.diff(csB, B))
de  = sp.simplify(sp.diff(csB, e))
dbe = sp.simplify(sp.diff(csB, beta))
print("dc*/dB =", dB, " ; matches n e (n-beta)/D^2? ->", is_zero(dB - n * e * (n - beta) / D**2))
print("dc*/de =", de, " ; matches B n / D? ->", is_zero(de - B * n / D))
print("dc*/dbeta =", dbe, " ; matches B n e (1+B(n-1))/D^2? ->",
      is_zero(dbe - B * n * e * (1 + B * (n - 1)) / D**2))

# ------------------------------------------------------------------------------
# Step 3:  Asymmetric Nash equilibrium  (heterogeneous altruism; n=2 and n=3)
# ------------------------------------------------------------------------------
banner("Step 3  -  Asymmetric interior Nash equilibrium")
A1, A2, A3, B1, B2, B3 = sp.symbols('A1 A2 A3 B1 B2 B3', positive=True)
c1, c2, c3 = sp.symbols('c1 c2 c3', positive=True)

# ---- n = 2 ----
th2 = 1 - beta / 2
def kp2(Ai, Bi):  return Bi / (Ai * th2)
def BR2(Ai, Bi, cj):
    k = kp2(Ai, Bi); return k * (e + (beta / 2) * cj) / (1 + k * th2)
s2 = sp.solve([sp.Eq(c1, BR2(A1, B1, c2)), sp.Eq(c2, BR2(A2, B2, c1))], [c1, c2], dict=True)[0]
def phi2(Ai, Bi):
    k = kp2(Ai, Bi); return k / (1 + k)
mu2 = 1 - (beta / 2) * (phi2(A1, B1) + phi2(A2, B2))
def cs2(Ai, Bi):  return (e / mu2) * phi2(Ai, Bi)
print("n=2 c1* matches (e/mu) phi_i? ->", is_zero(s2[c1] - cs2(A1, B1)))
print("n=2 c2* matches? ->", is_zero(s2[c2] - cs2(A2, B2)))

# ---- n = 3 ----
th3 = 1 - beta / 3
def kp3(Ai, Bi):  return Bi / (Ai * th3)
def BR3(Ai, Bi, cj, ck):
    k = kp3(Ai, Bi); return k * (e + (beta / 3) * (cj + ck)) / (1 + k * th3)
s3 = sp.solve([sp.Eq(c1, BR3(A1, B1, c2, c3)),
               sp.Eq(c2, BR3(A2, B2, c1, c3)),
               sp.Eq(c3, BR3(A3, B3, c1, c2))], [c1, c2, c3], dict=True)[0]
def phi3(Ai, Bi):
    k = kp3(Ai, Bi); return k / (1 + k)
mu3 = 1 - (beta / 3) * (phi3(A1, B1) + phi3(A2, B2) + phi3(A3, B3))
def cs3(Ai, Bi):  return (e / mu3) * phi3(Ai, Bi)
print("n=3 c1* matches (e/mu) phi_i? ->", is_zero(s3[c1] - cs3(A1, B1)))
print("n=3 c2* matches? ->", is_zero(s3[c2] - cs3(A2, B2)))
print("n=3 c3* matches? ->", is_zero(s3[c3] - cs3(A3, B3)))

# Consistency: homogeneous -> symmetric
hom3 = cs3(A1, B1).subs({A1: A, B1: B, A2: A, B2: B, A3: A, B3: B})
print("n=3 -> symmetric? ->", is_zero(hom3 - form_kap.subs(kappa, B / (A * th3))))

# Altruism ordering at n=3: sign(c1*-c2*) = sign(phi1-phi2) = sign(B1/A1 - B2/A2)
ord12 = sp.simplify(cs3(A1, B1) - cs3(A2, B2))
print("n=3 ordering c1*-c2* = (e/mu)(phi1-phi2)? ->",
      is_zero(ord12 - (e / mu3) * (phi3(A1, B1) - phi3(A2, B2))))

# ------------------------------------------------------------------------------
# Worked numerical example (n = 2, asymmetric): the altruism ordering
# ------------------------------------------------------------------------------
# Parameters: n=2, beta=1.2, e=15, A_i = 1 - B_i.
# Agent 1 is MORE altruistic (B1=0.2) than agent 2 (B2=0.1).
#   c_i* = (e/mu) phi_i ,  phi_i = kappa_i/(1+kappa_i),  kappa_i = B_i/(A_i theta),
#   mu = 1 - (beta/n) sum_j phi_j.
banner("Worked example: altruism ordering (n=2)")
ex_n, ex_beta, ex_e = 2, sp.Rational(6, 5), 15          # beta = 1.2
B1_v, B2_v = sp.Rational(1, 5), sp.Rational(1, 10)      # 0.2 and 0.1
theta_v = 1 - ex_beta / ex_n                            # = 0.4
def _kappa(Bv):                                         # A_i = 1 - B_i
    return Bv / ((1 - Bv) * theta_v)
def _phi(Bv):
    k = _kappa(Bv)
    return k / (1 + k)
phi1_v, phi2_v = _phi(B1_v), _phi(B2_v)
mu_v = 1 - (ex_beta / ex_n) * (phi1_v + phi2_v)
c1_v = (ex_e / mu_v) * phi1_v
c2_v = (ex_e / mu_v) * phi2_v
for nm, v in [("theta", theta_v), ("kappa_1", _kappa(B1_v)), ("kappa_2", _kappa(B2_v)),
              ("phi_1", phi1_v), ("phi_2", phi2_v), ("mu", mu_v),
              ("c_1*", c1_v), ("c_2*", c2_v)]:
    print(f"  {nm:8s} = {float(v):.4f}")
print(f"  ordering: c_1* > c_2*  ({float(c1_v):.4f} > {float(c2_v):.4f})  since B_1 > B_2")
print(f"  interior: 0 < c_i* < e={ex_e} ? -> {bool(0 < c2_v and c1_v < ex_e)}")

# ------------------------------------------------------------------------------
# Normalization A+B=1 is without loss of generality (scale invariance)
# ------------------------------------------------------------------------------
# Equilibrium contributions depend on (A,B) only through kappa = B/(A theta),
# homogeneous of degree 0: scaling (A,B) -> (t A, t B) leaves kappa -- and hence
# every best response and equilibrium contribution -- unchanged. Imposing A+B=1
# is therefore WLOG (the monotone rescaling U -> U^{1/(A+B)}); B is then the
# altruism share.
banner("Normalization A+B=1 is WLOG (scale invariance)")
t = sp.symbols('t', positive=True)
print("kappa = B/(A theta) invariant under (A,B)->(tA,tB)? ->",
      is_zero((t * B) / ((t * A) * theta) - kappa))
print("symmetric c* (kappa form) invariant under (A,B)->(tA,tB)? ->",
      is_zero(form_kap.subs({A: t * A, B: t * B}) - form_kap))
print("monotone-transform exponents A/(A+B), B/(A+B) sum to 1? ->",
      sp.simplify(A / (A + B) + B / (A + B)) == 1)

# ------------------------------------------------------------------------------
# Automated consistency checks
# ------------------------------------------------------------------------------
banner("Automated checks")
checks = {
    "FOC lambda-irrelevant":  sp.simplify(sp.diff(foc, lam_s)) == 0,
    "BR closed form":         is_zero(BR - BRc),
    "symmetric c* (kappa)":   is_zero(cstar - form_kap),
    "symmetric c* (B-form)":  is_zero(form_kap.subs(A, 1 - B) - form_AB),
    "dc*/dB":                 is_zero(dB - n * e * (n - beta) / D**2),
    "dc*/de":                 is_zero(de - B * n / D),
    "dc*/dbeta":              is_zero(dbe - B * n * e * (1 + B * (n - 1)) / D**2),
    "n=2 c1* gen form":       is_zero(s2[c1] - cs2(A1, B1)),
    "n=2 c2* gen form":       is_zero(s2[c2] - cs2(A2, B2)),
    "n=3 c1* gen form":       is_zero(s3[c1] - cs3(A1, B1)),
    "n=3 c2* gen form":       is_zero(s3[c2] - cs3(A2, B2)),
    "n=3 c3* gen form":       is_zero(s3[c3] - cs3(A3, B3)),
    "n=3 -> symmetric":       is_zero(hom3 - form_kap.subs(kappa, B / (A * th3))),
    "n=3 ordering":           is_zero(ord12 - (e / mu3) * (phi3(A1, B1) - phi3(A2, B2))),
    "example ordering c1>c2": bool(c1_v > c2_v),
    "example interiority":    bool(0 < c2_v and c1_v < ex_e),
    "kappa scale-invariant":  is_zero((t * B) / ((t * A) * theta) - kappa),
    "c* scale-invariant":     is_zero(form_kap.subs({A: t * A, B: t * B}) - form_kap),
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
assert all(checks.values()), "A derivation failed its consistency check."
print("\nAll checks passed.")
