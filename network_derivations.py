"""
================================================================================
 Network equilibria (cycle and star) -- Altruism in Networked Public Good Games
================================================================================
Symbolic/numeric verification of the Cobb--Douglas equilibria on the cycle and
star, as in "Equilibrium on the Cycle and the Star". A_i + B_i = 1; B_i is the
prosocial weight. The impact multiplier factors out of the Cobb--Douglas FOC, so
contributions depend on (A_i,B_i) and on network position only through d_i.

Verified:
  CYCLE   c*_cyc = kappa e/(1+kappa(1-beta)) = 3Be/(3-beta-2Bbeta),  theta=1-beta/3
          N=4 asymmetric: homogeneous -> symmetric; contributions ordered by B
  STAR    n=3 closed forms c_C, c_P; c_P - c_C = A B beta e / Delta_S > 0 (periphery>center)
          heterogeneous threshold: center > leaf  iff  B_1 > 2 B_P(3-beta)/(6-3beta+B_P beta)
          general-n: peripheries contribute more than the center on the interior region

Run:  python network_derivations.py
================================================================================
"""
import sympy as sp

e, beta, A, B = sp.symbols('e beta A B', positive=True)

def banner(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)

def is_zero(x):
    return sp.simplify(x) == 0

# ==============================================================================
# CYCLE  (every node: two neighbors, d_i = 3, theta = 1 - beta/3)
# ==============================================================================
banner("CYCLE -- symmetric equilibrium")
th_c = 1 - beta/3
kap_c = B/(A*th_c)
c = sp.symbols('c', positive=True)
# symmetric FOC: c = kappa (e - theta c + (2 beta/3) c), with -theta+2beta/3 = beta-1
c_cyc = sp.solve(sp.Eq(c, kap_c*(e - th_c*c + (2*beta/3)*c)), c)[0]
form_kappa = kap_c*e/(1 + kap_c*(1-beta))
form_B = 3*B*e/(3 - beta - 2*B*beta)
print("c*_cyc matches kappa form? ->", is_zero(c_cyc - form_kappa))
print("kappa form == 3Be/(3-beta-2Bbeta) under A=1-B? ->",
      is_zero(form_kappa.subs(A, 1-B) - form_B))
dB = sp.diff(form_B, B)
print("dc*_cyc/dB sign (beta=1.5,B=0.2):",
      sp.sign(dB.subs({beta: sp.Rational(3,2), B: sp.Rational(1,5), e: 15})))

banner("CYCLE -- N=4 asymmetric")
B1,B2,B3,B4 = sp.symbols('B1 B2 B3 B4', positive=True)
c1,c2,c3,c4 = sp.symbols('c1 c2 c3 c4', positive=True)
def kp_c(Bi): return Bi/((1-Bi)*th_c)
def Xc(ci,cl,cr): return e - ci + (beta/3)*(cl+ci+cr)
cyc4 = sp.solve([sp.Eq(c1, kp_c(B1)*Xc(c1,c4,c2)),
                 sp.Eq(c2, kp_c(B2)*Xc(c2,c1,c3)),
                 sp.Eq(c3, kp_c(B3)*Xc(c3,c2,c4)),
                 sp.Eq(c4, kp_c(B4)*Xc(c4,c3,c1))], [c1,c2,c3,c4], dict=True)[0]
cyc4_hom = is_zero(cyc4[c1].subs({B1:B,B2:B,B3:B,B4:B}) - form_B)
print("N=4 homogeneous -> symmetric 3Be/(3-beta-2Bbeta)? ->", cyc4_hom)
rank_ok = True
for bb in [1.2, 1.8, 2.4]:
    for tr in [(0.1,0.15,0.2,0.25),(0.2,0.1,0.2,0.1),(0.05,0.1,0.2,0.3)]:
        v = {B1:sp.Rational(str(tr[0])),B2:sp.Rational(str(tr[1])),
             B3:sp.Rational(str(tr[2])),B4:sp.Rational(str(tr[3])),
             beta:sp.Rational(str(bb)), e:15}
        cc = [float(cyc4[x].subs(v)) for x in (c1,c2,c3,c4)]
        if all(0 < x < 15 for x in cc):
            rank_ok = rank_ok and (sorted(range(4), key=lambda i: tr[i])
                                   == sorted(range(4), key=lambda i: cc[i]))
print("N=4 interior draws ordered by B (more prosocial -> more)? ->", rank_ok)

# ==============================================================================
# STAR  (center d=n; leaves d=2, theta_leaf=1-beta/2)
# ==============================================================================
banner("STAR -- n=3, homogeneous prosocial weight")
th1_3 = 1 - beta/3            # center, n=3
th2   = 1 - beta/2            # leaf
cC,cP = sp.symbols('c_C c_P', positive=True)
k1 = B/(A*th1_3); k2 = B/(A*th2)
# center sees c2+c3=2cP ; leaf sees c1=cC
eqC = sp.Eq(cC, k1*(e - cC + (beta/3)*(cC+2*cP)))
eqP = sp.Eq(cP, k2*(e - cP + (beta/2)*(cC+cP)))
sol = sp.solve([eqC,eqP],[cC,cP],dict=True)[0]
cC_s, cP_s = sp.simplify(sol[cC]), sp.simplify(sol[cP])
Delta_S = A**2*beta**2-5*A**2*beta+6*A**2+2*A*B*beta**2-10*A*B*beta+12*A*B-B**2*beta**2-5*B**2*beta+6*B**2
cC_claim = B*e*(6*A-3*A*beta+B*beta+6*B)/Delta_S
cP_claim = B*e*(6*A-2*A*beta+B*beta+6*B)/Delta_S
print("center c_C matches closed form? ->", is_zero(cC_s - cC_claim))
print("leaf   c_P matches closed form? ->", is_zero(cP_s - cP_claim))
gap = sp.simplify(cP_s - cC_s)
print("c_P - c_C = A B beta e / Delta_S ? ->", is_zero(gap - A*B*beta*e/Delta_S))

banner("STAR -- n=3, heterogeneous: center vs leaf threshold")
B1s,BP = sp.symbols('B1 B_P', positive=True)
def kphet(Bi,thi): return Bi/((1-Bi)*thi)
kC = kphet(B1s, th1_3); kL = kphet(BP, th2)
e1 = sp.Eq(cC, kC*(e - cC + (beta/3)*(cC+2*cP)))
e2 = sp.Eq(cP, kL*(e - cP + (beta/2)*(cC+cP)))
sh = sp.solve([e1,e2],[cC,cP],dict=True)[0]
gap_het = sp.simplify(sh[cC]-sh[cP])     # center - leaf
thr = 2*BP*(3-beta)/(6-3*beta+BP*beta)   # claimed B1*
# check: center>leaf  <=> B1 > thr, on feasible interior points
ok_thr = True
for bb in [1.2,1.5]:
    for bp in [0.1,0.2]:
        thr_v = float(thr.subs({BP:sp.Rational(str(bp)),beta:sp.Rational(str(bb))}))
        for b1 in [thr_v*0.6, thr_v*1.4]:
            if 0 < b1 < 0.95:
                v={B1s:sp.nsimplify(round(b1,4)),BP:sp.Rational(str(bp)),beta:sp.Rational(str(bb)),e:15}
                cc=float(sh[cC].subs(v)); cl=float(sh[cP].subs(v))
                if 0<cc<15 and 0<cl<15:
                    ok_thr = ok_thr and ((cc>cl) == (b1>thr_v))
print("threshold B1* = 2 B_P(3-beta)/(6-3beta+B_P beta): center>leaf iff B1>B1*? ->", ok_thr)
print("  example beta=1.2,B_P=0.1 -> B1* =",
      float(thr.subs({BP:sp.Rational(1,10),beta:sp.Rational(6,5)})))

banner("STAR -- general n: peripheries contribute more (interior region)")
N = sp.symbols('N', positive=True)
th1N = 1 - beta/N
kC_N = B/(A*th1N); kL = B/(A*th2)
eqCN = sp.Eq(cC, kC_N*(e - cC + (beta/N)*(cC+(N-1)*cP)))
eqPN = sp.Eq(cP, kL*(e - cP + (beta/2)*(cC+cP)))
solN = sp.solve([eqCN,eqPN],[cC,cP],dict=True)[0]
posPC = True
for NN in [3,4,6]:
    for bb in [1.2,1.6]:
        for BB in [0.1,0.2,0.3]:
            v={N:NN,beta:sp.Rational(str(bb)),B:sp.Rational(str(BB)),A:1-sp.Rational(str(BB)),e:15}
            cc=float(solN[cC].subs(v)); cp=float(solN[cP].subs(v))
            if 0<cc<15 and 0<cp<15:
                posPC = posPC and (cp > cc)
print("general-n interior draws: peripheries > center? ->", posPC)

# ==============================================================================
# Automated checks
# ==============================================================================
banner("Automated checks")
checks = {
    "cycle c* (kappa form)":      is_zero(c_cyc - form_kappa),
    "cycle c* (B-form)":          is_zero(form_kappa.subs(A,1-B) - form_B),
    "cycle N=4 -> symmetric":     cyc4_hom,
    "cycle N=4 ordered by B":     rank_ok,
    "star n=3 center form":       is_zero(cC_s - cC_claim),
    "star n=3 leaf form":         is_zero(cP_s - cP_claim),
    "star n=3 gap = ABbeta e/Δ":  is_zero(gap - A*B*beta*e/Delta_S),
    "star threshold B1*":         ok_thr,
    "star general-n periphery>center": posPC,
}
for k,v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
assert all(checks.values()), "A derivation failed its consistency check."
print("\nAll checks passed.")
