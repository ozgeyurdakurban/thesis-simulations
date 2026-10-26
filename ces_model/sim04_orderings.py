"""Ceilings (table) and positional-ordering plane (Figure 13).
 Table: interiority ceilings B-bar for the regular member (closed form) and the
        star (bisection on the coupled system), by sigma and lambda.
 Figure 13: sign of c_leaf - c_centre in the (lambda, sigma) plane at B=0.05,
        with the analytic boundary sigma*(lambda) and the threshold lambda-bar_n.
"""
import numpy as np, os, csv, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from ces_net import *
plt.rcParams.update({"font.size": 9})
e, beta, n = 15.0, 1.5, 3
Gs = adjacency("star", n)

def ceil_regular(sigma, lam):
    th = 1 - beta / n; Lt = 1 + lam * (n - 1) * beta / n
    r = th * Lt ** ((1 - sigma) / sigma) * beta ** (-1 / sigma)
    return r / (1 + r)

def ceil_star(sigma, lam, which):
    """smallest common B at which position `which` (0 centre, 1 leaf) hits e."""
    lo, hi = 1e-4, .9
    for _ in range(60):
        mid = .5 * (lo + hi)
        c = equilibrium(Gs, np.full(n, mid), beta, sigma, lam, e)
        if c[which] > e - 1e-9: hi = mid
        else: lo = mid
    return hi

rows = []
for lam in [0., 1., 2.]:
    for s in [.5, .75, 1., 1.5, 2.]:
        rows.append([lam, s, ceil_regular(s, lam), ceil_star(s, lam, 0), ceil_star(s, lam, 1)])
with open("data/ceilings_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lambda", "sigma", "B_regular", "B_star_centre", "B_star_leaf"]); w.writerows(rows)
for r in rows: print([round(x, 4) for x in r])

# Figure 13
B = .05
lg = np.linspace(0, 6, 97); sg = np.exp(np.linspace(np.log(.25), np.log(16), 97))
D = np.zeros((len(sg), len(lg)))
for j, l in enumerate(lg):
    for i, s in enumerate(sg):
        c = equilibrium(Gs, np.full(n, B), beta, s, l, e); D[i, j] = np.nan if c.max() > e - 1e-9 else c[1] - c[0]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
m = ax.pcolormesh(lg, sg, D, cmap="RdBu_r", norm=TwoSlopeNorm(0, min(np.nanmin(D), -1e-6), max(np.nanmax(D), 1e-6)), shading="auto")
ss = np.array([sigma_star(n, beta, l) if l > lam_bar(n, beta) else np.nan for l in lg])
ax.plot(lg, ss, color="k", lw=1.2, label=r"analytic boundary $\sigma^\ast(\lambda)$")
ax.axvline(lam_bar(n, beta), color="k", lw=.7, ls="--", label=rf"$\bar\lambda_3={lam_bar(n,beta):.2g}$")
ax.axhline(1, color="k", lw=.5, ls=":")
ax.set_yscale("log", base=2); ax.set_ylim(.25, 16); ax.set_yticks([.25, .5, 1, 2, 4, 8, 16]); ax.set_yticklabels(["1/4", "1/2", "1", "2", "4", "8", "16"])
ax.set_facecolor("#d9d9d9")
ax.set_xlabel(r"impact weight $\lambda$"); ax.set_ylabel(r"$\sigma$")
fig.colorbar(m, label=r"$c_{\rm leaf}-c_{\rm centre}$"); ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig13_ordering_lambda_sigma.png", dpi=200); plt.close(fig)
print("min leaf-centre (interior):", np.nanmin(D), " interior reversals:", int((D < 0).sum()), " corner cells:", int(np.isnan(D).sum()))
# check numerical sign change agrees with sigma*
bad = 0
for j, l in enumerate(lg):
    if l > lam_bar(n, beta) + 1e-9 and np.isfinite(ss[j]) and ss[j] < 16:
        i = np.argmin(abs(sg - ss[j]))
        a_, b_ = D[min(i + 3, len(sg) - 1), j], D[max(i - 3, 0), j]
        if np.isfinite(a_) and np.isfinite(b_) and (np.sign(a_) != -1 or np.sign(b_) != 1): bad += 1
print("boundary mismatches:", bad)
