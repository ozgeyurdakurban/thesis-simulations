"""Regime maps.
 Figure 11: (B, sigma) plane at beta=1.5, lambda=1: interior / leaf at ceiling /
            centre and leaf at ceiling (n=3 star), with the closed-form ceiling of
            the regular member overlaid.
 Figure 12: (B, beta) plane for sigma in {1/2,1,2}: joint feasibility of the
            three-person star design (dilemma, social productivity, interiority).
"""
import numpy as np, os, csv, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ces_net import *
plt.rcParams.update({"font.size": 9})
os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)
e, n = 15.0, 3
Gs = adjacency("star", n)
tol = 1e-9

def ceil_regular(beta, sigma, lam, n=3):
    th = 1 - beta / n; Lt = 1 + lam * (n - 1) * beta / n
    r = th * Lt ** ((1 - sigma) / sigma) * beta ** (-1 / sigma)
    return r / (1 + r)

beta, lam = 1.5, 1.0
Bg = np.linspace(.005, .40, 160); sg = np.exp(np.linspace(np.log(.25), np.log(4), 120))
Z = np.zeros((len(sg), len(Bg)))
for i, s in enumerate(sg):
    c = equilibrium(Gs, np.broadcast_to(Bg[:, None], (len(Bg), n)).copy(), beta, s, lam, e)
    Z[i] = np.where(c[:, 0] > e - tol, 2, np.where(c[:, 1:].max(1) > e - tol, 1, 0))
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.pcolormesh(Bg, sg, Z, cmap=ListedColormap(["#e9f0f7", "#f3cfc4", "#c8553d"]), shading="auto", vmin=0, vmax=2)
ax.plot([ceil_regular(beta, s, lam) for s in sg], sg, color="#4d4d4d", lw=1.4, label=r"regular member ceiling $\bar B$ (closed form)")
ax.set_yscale("log", base=2); ax.set_yticks([.25, .5, 1, 2, 4]); ax.set_yticklabels(["1/4", "1/2", "1", "2", "4"])
ax.axhline(1, color="k", lw=.5, ls=":")
ax.set_xlabel(r"prosocial weight $B$"); ax.set_ylabel(r"$\sigma$"); ax.set_xlim(0, .4)
ax.legend(frameon=False, fontsize=7, loc="lower right"); fig.tight_layout()
fig.savefig("figures/fig11_regimes_B_sigma.png", dpi=200); plt.close(fig)
print("regime shares:", [(Z == k).mean().round(3) for k in range(3)])

bg = np.linspace(1.0, 2.2, 97); Bg2 = np.linspace(.005, .45, 120)
fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.1), sharey=True); rows = []
for a, s in zip(ax, [.5, 1.0, 2.0]):
    F = np.zeros((len(bg), len(Bg2)))
    for i, b in enumerate(bg):
        d, th, Lam = structure(Gs, b); sprod = Gs @ (b / d) + b / d
        ok = (b < d).all() and (b > 1) and (sprod > 1).all()
        c = equilibrium(Gs, np.broadcast_to(Bg2[:, None], (len(Bg2), n)).copy(), b, s, lam, e)
        F[i] = ok * (c.max(1) < e - tol)
    a.pcolormesh(Bg2, bg, F, cmap=ListedColormap(["#ffffff", "#9fc5e8"]), shading="auto")
    a.set_title(rf"$\sigma={s:g}$"); a.set_xlabel(r"$B$"); a.axhline(1.5, color="#c8553d", lw=.8, ls="--")
    i15 = np.argmin(abs(bg - 1.5)); Bc = Bg2[np.where(F[i15] > 0)[0].max()] if F[i15].any() else np.nan
    rows.append([s, Bc, ceil_regular(1.5, s, 1.0)])
ax[0].set_ylabel(r"$\beta$"); fig.tight_layout(); fig.savefig("figures/fig12_feasible_experiment_sigma.png", dpi=200); plt.close(fig)
with open("data/ceilings_beta15.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["sigma", "B_ceiling_star_all_interior", "B_ceiling_regular"]); w.writerows(rows)
print(rows)
