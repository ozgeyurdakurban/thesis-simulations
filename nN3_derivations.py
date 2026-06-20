"""
================================================================================
 N = 3 asymmetric closed forms  (for the three-player experiment)
================================================================================
Verifies the explicit n=3 asymmetric equilibria added to the equilibrium chapter:
  - Cobb--Douglas, no network (= cycle, since C_3 = K_3):  c_i = e phi_i / mu
  - CES no-network benchmark:                              c_i = e r_i  / mu
  - Star (center 1, leaves 2,3):  leaves linear in center; center closed form;
        leaf ordering by B; center>leaf threshold B_1* = 2 B_P(3-beta)/(6-3beta+B_P beta).
A_i + B_i = 1 ; B_i is the prosocial weight.

Run:  python nN3_derivations.py
================================================================================
"""
import sympy as sp

e, beta = sp.symbols('e beta', positive=True)
B1, B2, B3 = sp.symbols('B1 B2 B3', positive=True)
c1, c2, c3 = sp.symbols('c1 c2 c3', positive=True)

def banner(t): print("\n"+"="*70+"\n"+t+"\n"+"="*70)
def zero(x): return sp.simplify(x) == 0

# ---------------------------------------------------------------- CD / cycle n=3
banner("Cobb--Douglas, no network (= cycle), n = 3")
th = 1 - beta/3
def kp(B): return B/((1-B)*th)
def X(ci): return e - ci + (beta/3)*(c1+c2+c3)
sol = sp.solve([sp.Eq(c1, kp(B1)*X(c1)), sp.Eq(c2, kp(B2)*X(c2)), sp.Eq(c3, kp(B3)*X(c3))],
               [c1, c2, c3], dict=True)[0]
def phi(B): k = kp(B); return k/(1+k)
mu = 1 - (beta/3)*(phi(B1)+phi(B2)+phi(B3))
def cstar(B): return e*phi(B)/mu
cd_ok = zero(sol[c1]-cstar(B1)) and zero(sol[c2]-cstar(B2)) and zero(sol[c3]-cstar(B3))
print("c_i = e phi_i / mu  (phi_i = B_i/((1-B_i)theta+B_i), theta=1-beta/3)?  ->", cd_ok)

# --------------------------------------------------------------- CES no-net n=3
banner("CES no-network benchmark, n = 3")
rho, d1, d2, d3 = sp.symbols('rho delta1 delta2 delta3', positive=True)
Lam = 1 + 2*beta/3
# CES FOC gives Y=k X, k = (delta theta/((1-delta) Lambda))^(1/(rho-1)); behavior via r=k/(Lam+k)
k1, k2, k3 = sp.symbols('k1 k2 k3', positive=True)
def Xc(ci): return e - ci + (beta/3)*(c1+c2+c3)
# best response from Y=k X with Y=Lam c: Lam c = k(e-theta c+(beta/3)(C - c))... use aggregate form
# c_i (Lam + k_i) = k_i (e + (beta/3) C)  ->  c_i = r_i (e+(beta/3)C), r_i=k_i/(Lam+k_i)
solC = sp.solve([sp.Eq(ci*(Lam+ki), ki*(e+(beta/3)*(c1+c2+c3)))
                 for ci,ki in [(c1,k1),(c2,k2),(c3,k3)]], [c1,c2,c3], dict=True)[0]
def r(ki): return ki/(Lam+ki)
muC = 1 - (beta/3)*(r(k1)+r(k2)+r(k3))
def cstarC(ki): return e*r(ki)/muC
ces_ok = all(zero(solC[ci]-cstarC(ki)) for ci,ki in [(c1,k1),(c2,k2),(c3,k3)])
print("c_i = e r_i / mu  (r_i = k_i/(Lambda+k_i), Lambda=1+2beta/3)?  ->", ces_ok)
print("CD limit: r_i -> phi_i as rho->0 (k_i/Lambda -> kappa_i) gives the CD n=3 form")

# ------------------------------------------------------------------- Star n=3
banner("Star, n = 3 (center 1, leaves 2,3)")
th1 = 1 - beta/3; thL = 1 - beta/2
def kc(B): return B/((1-B)*th1)
def kl(B): return B/((1-B)*thL)
X1 = e - c1 + (beta/3)*(c1+c2+c3)
X2 = e - c2 + (beta/2)*(c1+c2)
X3 = e - c3 + (beta/2)*(c1+c3)
sols = sp.solve([sp.Eq(c1, kc(B1)*X1), sp.Eq(c2, kl(B2)*X2), sp.Eq(c3, kl(B3)*X3)],
                [c1, c2, c3], dict=True)[0]
def ab(B):
    k = kl(B); return (k*e/(1+k*thL), k*(beta/2)/(1+k*thL))
a2, b2 = ab(B2); a3, b3 = ab(B3); k1c = kc(B1)
c1_closed = k1c*(e + (beta/3)*(a2+a3)) / ((1+k1c*th1) - k1c*(beta/3)*(b2+b3))
star_c1_ok = zero(sols[c1]-c1_closed)
print("center c_1 matches substitution closed form? ->", star_c1_ok)

# leaf ordering: sign(c2-c3) = sign(B2-B3) on feasible interior points
leaf_ord = True
def num(x,v): return float(x.subs(v))
for bb in [1.2,1.5,1.8]:
    for (b1,b2,b3) in [(0.2,0.1,0.3),(0.1,0.25,0.15),(0.3,0.2,0.05)]:
        v={B1:sp.Rational(str(b1)),B2:sp.Rational(str(b2)),B3:sp.Rational(str(b3)),beta:sp.Rational(str(bb)),e:15}
        cc=[num(sols[c],v) for c in (c1,c2,c3)]
        if all(0<x<15 for x in cc):
            leaf_ord = leaf_ord and ((cc[1]>cc[2])==(b2>b3))
print("leaves ordered by B (c2>c3 iff B2>B3, interior)? ->", leaf_ord)

# threshold: center=leaf at B1=B1* (common leaves B_P)
BP = sp.symbols('B_P', positive=True)
gap = (sols[c1]-sols[c2]).subs({B2:BP, B3:BP})
thr = 2*BP*(3-beta)/(6-3*beta+BP*beta)
thr_ok = True
for bp in [0.1,0.2]:
    for bb in [1.2,1.5]:
        val = abs(float(gap.subs({B1:thr, BP:sp.Rational(str(bp)), beta:sp.Rational(str(bb)), e:15})))
        thr_ok = thr_ok and (val < 1e-9)
print("center=leaf exactly at B1 = B1* = 2B_P(3-beta)/(6-3beta+B_P beta)? ->", thr_ok)
print("  B1* > B_P (center needs more prosociality), e.g. beta=1.2,B_P=0.1 -> B1* =",
      float(thr.subs({BP:sp.Rational(1,10),beta:sp.Rational(6,5)})))

banner("Automated checks")
checks = {"CD/cycle n=3": cd_ok, "CES n=3": ces_ok,
          "star center n=3": star_c1_ok, "star leaf ordering": leaf_ord,
          "star threshold B1*": thr_ok}
for k,v in checks.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(checks.values())
print("\nAll n=3 checks passed.")
