"""Separation of the elasticity of substitution and the impact weight by price
(rebate) and impact (match) variation -- Section "Separating sigma and lambda".

Environment e = (theta, nu): one unit of giving costs theta units of private
consumption and delivers nu units of material benefit to others, so that
X = W - theta*g, Y = (1 + lam*nu)*g, p = theta/(1 + lam*nu).  Network model:
theta_i = 1 - beta/d_i, nu_i = Lambda_i - 1.  A dictator-type decision has W = e.

Checks:
 (A) closed-form share (ces_net.share) against direct utility maximization;
 (B) exact recovery of (sigma, lambda, B) from a baseline, a rebate cell and a
     match cell by the inversion formulas of the Proposition;
 (C) equal-price rebate and match: ratio of net transfers = pi_match/pi_rebate
     = [(1+lam)/(1-r+lam)]^(1-sigma) in odds, > 1 iff sigma < 1.
Writes data/rebate_match.csv (illustration table)."""
import numpy as np, csv, os
from ces_net import share, price

E = 100.0

def direct_share(B, sigma, lam, theta, nu, W=E):
    """Giving share theta*g/W from direct maximization of the CES utility on the
    budget line (golden-section search; the objective is strictly concave in g)."""
    rho = 1 - 1 / sigma
    Lt = 1 + lam * nu
    def U(g):
        X = W - theta * g; Y = Lt * g
        return ((1 - B) * X ** rho + B * Y ** rho) ** (1 / rho)
    a, b = 1e-10, W / theta * (1 - 1e-9)
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(200):
        if U(c) > U(d): b = d
        else: a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    return theta * 0.5 * (a + b) / W

def logodds(p): return np.log(p / (1 - p))

def recover(pi0, piP, piI, theta0, thetaP, nu0, nuI):
    l0, lP, lI = logodds(pi0), logodds(piP), logodds(piI)
    sigma = 1 - (lP - l0) / np.log(thetaP / theta0)
    R = np.exp((lI - l0) / (sigma - 1))
    lam = (R - 1) / (nuI - R * nu0)
    p0 = theta0 / (1 + lam * nu0)
    b = (l0 - (1 - sigma) * np.log(p0)) / sigma
    return sigma, lam, 1 / (1 + np.exp(-b))

# environments: baseline (charity dictator: 1 lira costs 1, delivers 1), rebate r, match m
th0, nu0 = 1.0, 1.0
r = 0.5; m = r / (1 - r)                     # equal price of the recipient's benefit
thP, nuI = th0 - r, nu0 * (1 + m)

# (A) validation
maxerr = 0
for B in [.05, .1, .2]:
    for sigma in [.5, .8, 1.25, 2, 4]:
        for lam in [0, .5, 1, 3]:
            for (th, nu) in [(th0, nu0), (thP, nu0), (th0, nuI)]:
                pi_cf = float(share(B, price(th, 1 + nu, lam), sigma))
                maxerr = max(maxerr, abs(pi_cf - direct_share(B, sigma, lam, th, nu)))
print("(A) max |closed form - direct maximization| =", f"{maxerr:.2e}")

# (B) recovery and (C) equal-price comparison
rows = []; maxrec = 0
for B in [.05, .1, .2]:
    for sigma in [.5, .8, 1.25, 2, 4]:
        for lam in [0, .5, 1, 3]:
            pis = [direct_share(B, sigma, lam, th, nu) for (th, nu) in [(th0, nu0), (thP, nu0), (th0, nuI)]]
            s, l, b = recover(*pis, th0, thP, nu0, nuI)
            maxrec = max(maxrec, abs(s - sigma), abs(l - lam), abs(b - B))
            ratio = pis[2] / pis[1]
            theo = (1 + lam) / (1 - r + lam)
            rows.append([B, sigma, lam, *pis, ratio, theo ** (1 - sigma)])
print("(B) max recovery error (sigma, lambda, B) =", f"{maxrec:.2e}")
for sg in [.5, 2]:
    for lm in [0, 1]:
        x = [q for q in rows if q[0] == .1 and q[1] == sg and q[2] == lm][0]
        print(f"    B=.10 sigma={sg} lam={lm}: pi_base={x[3]:.4f} pi_rebate={x[4]:.4f} pi_match={x[5]:.4f} net match/rebate={x[6]:.3f}")
os.makedirs("data", exist_ok=True)
with open("data/rebate_match.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["B", "sigma", "lambda", "pi_base", "pi_rebate", "pi_match", "net_ratio_match_over_rebate", "odds_ratio_theory"])
    w.writerows(rows)
