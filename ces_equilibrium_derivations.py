"""
================================================================================
 CES equilibrium derivations  --  Altruism in Networked Public Good Games
================================================================================

Symbolic derivation and verification of the CES equilibrium analysis for the
FULLY CONNECTED network. Reproduces, step by step, the results in the
"Equilibrium Analysis / CES Framework" section of the thesis.

Each block is self-contained and prints its result. The final block runs
assertions that the closed forms are correct and mutually consistent
(symmetric <-> asymmetric, general N <-> N=2).

Map to the thesis:
    Step 1  -> Lemma (CES ratio rule)  +  Proposition (interior best response)
    Step 2  -> Proposition (symmetric interior Nash equilibrium)
    Feas.   -> Feasibility condition  k < lambda/beta
    Comp.   -> Proposition (comparative statics of the symmetric equilibrium)
    Asym.   -> Two-agent asymmetric Nash equilibrium (heterogeneous altruism)

Requirements:  Python 3,  sympy  (pip install sympy)
Run:           python ces_equilibrium_derivations.py
================================================================================
"""

import sympy as sp

# ------------------------------------------------------------------------------
# Symbols
# ------------------------------------------------------------------------------
# Economic parameters
e     = sp.symbols('e', positive=True)        # endowment
beta  = sp.symbols('beta', positive=True)     # public-good marginal return (1 < beta < n)
n     = sp.symbols('n', positive=True)        # number of agents
rho   = sp.symbols('rho')                     # CES substitution parameter (rho < 1, rho != 0)
delta = sp.symbols('delta', positive=True)    # CES weight on MATERIAL payoff (in (0,1))
                                              #   -> altruism weight is (1 - delta)

# Choice variables
c_i    = sp.symbols('c_i', positive=True)     # own contribution
C_mi   = sp.symbols('C_mi', positive=True)    # C_{-i} = sum of others' contributions
c      = sp.symbols('c', positive=True)       # symmetric contribution

# Reduced-form response constant (Y_i = k_i * X_i in any interior optimum)
k      = sp.symbols('k', positive=True)


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def is_zero(expr):
    """Return True if `expr` is identically zero.

    Symbolic simplification of expressions containing *symbolic* powers (e.g.
    X**(rho-1)) is not always reduced to 0 by sympy, because splitting
    (a*b)**(rho-1) into a**(rho-1)*b**(rho-1) requires positivity assumptions.
    We therefore (i) force power simplification, and if that does not return 0,
    (ii) fall back to evaluating the expression at several admissible numeric
    parameter points (1 < beta < n, delta,a in (0,1), rho < 1, all vars > 0).
    A genuine identity evaluates to 0 at every point.
    """
    reduced = sp.simplify(
        sp.powsimp(sp.expand_power_base(expr, force=True), force=True)
    )
    if reduced == 0:
        return True
    test_points = [
        {'n': 4, 'beta': 2, 'e': 10, 'delta': sp.Rational(3, 10), 'a': sp.Rational(2, 5),
         'rho': -1, 'k': sp.Rational(1, 2), 'k1': sp.Rational(2, 5), 'k2': sp.Rational(7, 10),
         'c_i': 2, 'C_mi': 3, 'c': sp.Rational(3, 2), 'c1': sp.Rational(6, 5), 'c2': sp.Rational(9, 5)},
        {'n': 5, 'beta': sp.Rational(3, 2), 'e': 8, 'delta': sp.Rational(1, 2), 'a': sp.Rational(3, 10),
         'rho': sp.Rational(1, 2), 'k': sp.Rational(3, 5), 'k1': sp.Rational(1, 5), 'k2': sp.Rational(4, 5),
         'c_i': 1, 'C_mi': 2, 'c': 1, 'c1': sp.Rational(1, 2), 'c2': sp.Rational(3, 2)},
    ]
    for pt in test_points:
        subs = {s: pt[s.name] for s in expr.free_symbols if s.name in pt}
        val = complex(sp.N(expr.subs(subs)))
        if abs(val) > 1e-9:
            return False
    return True


# ------------------------------------------------------------------------------
# Step 0:  Primitives under full connectivity (d_i = n for all i)
# ------------------------------------------------------------------------------
# theta : local private NET marginal cost of contribution      theta = 1 - beta/n
# lam   : impact multiplier (common to all agents under full connectivity)
#                                                              lam = 1 + (n-1)beta/n
theta = 1 - beta / n
lam   = 1 + (n - 1) * beta / n

# Material component  X_i = e - c_i + (beta/n) C   with  C = c_i + C_{-i}
#                         = e - theta c_i + (beta/n) C_{-i}
X_i = e - c_i + (beta / n) * (c_i + C_mi)
# Moral (impact-adjusted) component  Y_i = lambda * c_i
Y_i = lam * c_i

banner("Step 0  -  Primitives (fully connected)")
print("theta  =", theta)
print("lambda =", sp.simplify(lam))
print("X_i    =", sp.simplify(X_i), "   (= e - theta c_i + (beta/n) C_{-i})")
print("Y_i    =", Y_i)


# ------------------------------------------------------------------------------
# Step 1:  Interior best response  (FOC -> CES ratio rule -> linear BR)
# ------------------------------------------------------------------------------
# Maximizing the CES utility is equivalent to maximizing the inner index
#     V_i = delta * X_i^rho + (1 - delta) * Y_i^rho
# because the CES aggregator is a strictly increasing transform of V_i.
V_i = delta * X_i**rho + (1 - delta) * Y_i**rho

# First-order condition dV_i/dc_i = 0
foc = sp.diff(V_i, c_i)

# Hand-written "clean" FOC using dX/dc = -theta, dY/dc = +lambda:
#     delta * X^(rho-1) * (-theta) + (1 - delta) * Y^(rho-1) * lambda = 0
# Note: foc = rho * foc_clean, so foc/rho should equal foc_clean exactly.
foc_clean = delta * X_i**(rho - 1) * (-theta) + (1 - delta) * Y_i**(rho - 1) * lam

banner("Step 1a  -  First-order condition")
print("dX_i/dc_i =", sp.diff(X_i, c_i), "   (= -theta)")
print("dY_i/dc_i =", sp.diff(Y_i, c_i), "   (= +lambda)")
print("clean FOC matches dV/dc (up to factor rho)? ->", is_zero(foc_clean - foc / rho))

# The clean FOC rearranges to the CES ratio rule:
#     (Y_i / X_i)^(rho-1) = delta*theta / ((1-delta)*lambda)
# Define the response constant k from the ratio rule:
k_def = (delta * theta / ((1 - delta) * lam))**(1 / (rho - 1))

banner("Step 1b  -  CES ratio rule  =>  Y_i = k_i X_i")
print("Ratio rule:  (Y_i/X_i)^(rho-1) = delta*theta/((1-delta)*lambda)")
print("k_i =", k_def)

# Impose Y_i = k * X_i  and solve for the best response c_i = BR_i(C_{-i}).
BR_eq = sp.Eq(lam * c_i, k * (e - theta * c_i + (beta / n) * C_mi))
BR    = sp.solve(BR_eq, c_i)[0]

# Closed form claimed in the thesis:  BR = k (e + (beta/n) C_{-i}) / (lambda + k theta)
BR_claim = k * (e + (beta / n) * C_mi) / (lam + k * theta)

banner("Step 1c  -  Interior best response  BR_i(C_{-i})")
print("BR_i =", sp.simplify(BR))
print("matches  k(e+(beta/n)C_{-i})/(lambda+k theta)? ->", is_zero(BR - BR_claim))

# Strategic complementarity:  dBR/dC_{-i} > 0
slope = sp.simplify(sp.diff(BR_claim, C_mi))
banner("Step 1d  -  Strategic complementarity")
print("dBR_i/dC_{-i} =", slope, "  (> 0  =>  strategic complements)")


# ------------------------------------------------------------------------------
# Step 2:  Symmetric Nash equilibrium  (homogeneous altruism, general N)
# ------------------------------------------------------------------------------
# In a symmetric profile  C_{-i} = (n-1) c  and  c_i = c, so the fixed point is
#     c = BR(C_{-i}) = k ( e + (beta/n)(n-1) c ) / (lambda + k theta).
sym_eq = sp.Eq(c, k * (e + (beta / n) * (n - 1) * c) / (lam + k * theta))
c_star = sp.solve(sym_eq, c)[0]

# Closed form claimed in the thesis:  c* = k e / (lambda + k(1-beta))
c_star_claim = k * e / (lam + k * (1 - beta))

banner("Step 2  -  Symmetric interior Nash equilibrium (general N)")
print("c* =", sp.simplify(c_star))
print("matches  k e / (lambda + k(1-beta))? ->", is_zero(c_star - c_star_claim))
print("c* (N=2) =", sp.simplify(c_star_claim.subs(n, 2)), "   (lambda = 1 + beta/2)")


# ------------------------------------------------------------------------------
# Feasibility:  interior symmetric equilibrium  <=>  k < lambda/beta
# ------------------------------------------------------------------------------
# With D = lambda + k(1-beta):  c* = k e / D.
#   c* < e  <=>  k < D  <=>  k*beta < lambda  <=>  k < lambda/beta.
D = lam + k * (1 - beta)
banner("Feasibility  -  interior condition")
print("D = lambda + k(1-beta) =", sp.simplify(D))
print("D - k =", sp.simplify(D - k), "   (>0  <=>  k*beta < lambda  <=>  c* < e)")
print("lambda/beta =", sp.simplify(lam / beta), "   (= 1/beta + (n-1)/n)")


# ------------------------------------------------------------------------------
# Comparative statics of the symmetric equilibrium  (treating k as a parameter)
# ------------------------------------------------------------------------------
cs = c_star_claim  # = k e / D
banner("Comparative statics  (symmetric interior equilibrium)")
print("dc*/de =", sp.simplify(sp.diff(cs, e)), "   (= k/D > 0)")
print("dc*/dk =", sp.simplify(sp.diff(cs, k)), "   (= lambda e / D^2 > 0)")

# Altruism a = 1 - delta enters c* ONLY through k = k(a).  Sign of dk/da:
a = sp.symbols('a', positive=True)                 # altruism = 1 - delta
k_of_a = ((1 - a) * theta / (a * lam))**(1 / (rho - 1))
dk_da  = sp.simplify(sp.diff(k_of_a, a))
banner("Altruism comparative static  (a = 1 - delta)")
print("k(a)  =", k_of_a)
print("dk/da =", dk_da)
print("=> for rho < 1, dk/da > 0, hence dc*/da = (lambda e/D^2) dk/da > 0")

# Effect of beta, holding k fixed (the 'direct' channel):
dcs_dbeta_k = sp.simplify(sp.diff(cs, beta))
banner("Effect of beta on c*  (k held fixed)")
print("dc*/dbeta|_k =", dcs_dbeta_k)
print("sign tracks (k - (n-1)/n)? ->",
      is_zero(dcs_dbeta_k - k * e * (k - (n - 1) / n) / D**2))


# ------------------------------------------------------------------------------
# Asymmetric Nash equilibrium:  two agents, heterogeneous altruism (N = 2)
# ------------------------------------------------------------------------------
# Agents differ only in their altruism, hence in k_1 != k_2.
# Topology is symmetric (fully connected dyad), so lambda is common.
banner("Asymmetric Nash equilibrium  (N = 2, heterogeneous altruism)")

k1, k2 = sp.symbols('k1 k2', positive=True)
c1, c2 = sp.symbols('c1 c2', positive=True)

# N = 2 primitives
theta2 = (1 - beta / 2)        # theta at n=2
lam2   = (1 + beta / 2)        # lambda at n=2

# Best responses (C_{-i} = c_j):  c_i = k_i (e + (beta/2) c_j) / (lambda + k_i theta)
BR1 = k1 * (e + (beta / 2) * c2) / (lam2 + k1 * theta2)
BR2 = k2 * (e + (beta / 2) * c1) / (lam2 + k2 * theta2)

sol = sp.solve([sp.Eq(c1, BR1), sp.Eq(c2, BR2)], [c1, c2], dict=True)[0]
c1_star = sp.simplify(sol[c1])
c2_star = sp.simplify(sol[c2])

# Compact closed form claimed in the thesis:
#   c1* = k1 e (lambda + k2) / Delta ,   c2* = k2 e (lambda + k1) / Delta
#   Delta = lambda^2 + lambda theta (k1 + k2) + k1 k2 (1 - beta)
Delta    = lam2**2 + lam2 * theta2 * (k1 + k2) + k1 * k2 * (1 - beta)
c1_claim = k1 * e * (lam2 + k2) / Delta
c2_claim = k2 * e * (lam2 + k1) / Delta

print("c1* matches compact form? ->", is_zero(c1_star - c1_claim))
print("c2* matches compact form? ->", is_zero(c2_star - c2_claim))

# Ordering result: who contributes more?  c1* - c2* has the sign of (k1 - k2).
diff_12 = sp.simplify(c1_claim - c2_claim)
print("c1* - c2* =", diff_12)
print("equals e*lambda*(k1-k2)/Delta? ->",
      is_zero(diff_12 - e * lam2 * (k1 - k2) / Delta))
print("=> the MORE ALTRUISTIC agent (higher k_i) contributes strictly more.")

# Consistency: setting k1 = k2 = k must reproduce the symmetric solution at n=2.
c1_to_sym = sp.simplify(c1_claim.subs({k1: k, k2: k}))
c_sym_n2  = sp.simplify(c_star_claim.subs(n, 2))
print("asymmetric -> symmetric (k1=k2=k) at n=2? ->", is_zero(c1_to_sym - c_sym_n2))


# ------------------------------------------------------------------------------
# Asymmetric Nash equilibrium:  general-N closed form  (verified at N = 3)
# ------------------------------------------------------------------------------
# Aggregate-form best response (use theta + beta/n = 1):
#     c_i (lambda + k_i) = k_i ( e + (beta/n) C ),   C = sum_k c_k
#  => c_i = r_i ( e + (beta/n) C ),   with  r_i = k_i / (lambda + k_i).
# Summing over i gives the general-N closed form:
#     C*   = e S / mu ,   S = sum_j r_j ,   mu = 1 - (beta/n) S ,
#     c_i* = (e/mu) r_i = (e/mu) * k_i / (lambda + k_i).
banner("Asymmetric Nash equilibrium  (general N, verified at N = 3)")

k3, c3 = sp.symbols('k3 c3', positive=True)
lam3   = 1 + 2 * beta / 3          # lambda at n = 3
th3    = 1 - beta / 3

def BR_n3(ki, cj, ck):             # C_{-i} = cj + ck
    return ki * (e + (beta / 3) * (cj + ck)) / (lam3 + ki * th3)

sol3 = sp.solve(
    [sp.Eq(c1, BR_n3(k1, c2, c3)),
     sp.Eq(c2, BR_n3(k2, c1, c3)),
     sp.Eq(c3, BR_n3(k3, c1, c2))],
    [c1, c2, c3], dict=True)[0]

# General closed form, specialized to n = 3
def r(ki):
    return ki / (lam3 + ki)
S3  = r(k1) + r(k2) + r(k3)
mu3 = 1 - (beta / 3) * S3
def cstar3(ki):
    return (e / mu3) * r(ki)

print("N=3 c1* matches (e/mu) k_i/(lambda+k_i)? ->", is_zero(sol3[c1] - cstar3(k1)))
print("N=3 c2* matches? ->", is_zero(sol3[c2] - cstar3(k2)))
print("N=3 c3* matches? ->", is_zero(sol3[c3] - cstar3(k3)))

# Consistency: k1=k2=k3=k  ->  symmetric c* = e k / (lambda + k(1-beta))
c_sym_n3 = sp.simplify(cstar3(k1).subs({k1: k, k2: k, k3: k}))
print("N=3 -> symmetric? ->", is_zero(c_sym_n3 - k * e / (lam3 + k * (1 - beta))))

# Ordering: c_i* - c_j* = lambda e (k_i - k_j) / ( mu (lambda+k_i)(lambda+k_j) )
ord_claim_n3 = lam3 * e * (k1 - k2) / (mu3 * (lam3 + k1) * (lam3 + k2))
print("N=3 ordering c1*-c2* = lam e (k1-k2)/(mu(lam+k1)(lam+k2))? ->",
      is_zero(cstar3(k1) - cstar3(k2) - ord_claim_n3))


# ------------------------------------------------------------------------------
# Automated consistency checks (raise if any derivation is wrong)
# ------------------------------------------------------------------------------
banner("Automated checks")
checks = {
    "FOC -> clean FOC":   is_zero(foc_clean - foc / rho),
    "BR closed form":     is_zero(BR - BR_claim),
    "symmetric c*":       is_zero(c_star - c_star_claim),
    "asymmetric c1*":     is_zero(c1_star - c1_claim),
    "asymmetric c2*":     is_zero(c2_star - c2_claim),
    "ordering c1*-c2*":   is_zero(diff_12 - e * lam2 * (k1 - k2) / Delta),
    "asym -> sym (n=2)":  is_zero(c1_to_sym - c_sym_n2),
    "N=3 c1* gen. form":  is_zero(sol3[c1] - cstar3(k1)),
    "N=3 c2* gen. form":  is_zero(sol3[c2] - cstar3(k2)),
    "N=3 c3* gen. form":  is_zero(sol3[c3] - cstar3(k3)),
    "N=3 -> symmetric":   is_zero(c_sym_n3 - k * e / (lam3 + k * (1 - beta))),
    "N=3 ordering":       is_zero(cstar3(k1) - cstar3(k2) - ord_claim_n3),
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
assert all(checks.values()), "A derivation failed its consistency check."
print("\nAll checks passed.")
