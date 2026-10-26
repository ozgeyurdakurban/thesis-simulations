"""Equilibrium contributions by position as functions of sigma and lambda
(Figure: positions; Table: positions).  e=15, n=3, beta=1.5, B=0.10."""
import numpy as np, os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ces_net import *
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)
e, beta, n, B = 15.0, 1.5, 3, 0.10
Gk, Gs = adjacency("complete", n), adjacency("star", n)
sig = np.exp(np.linspace(np.log(.25), np.log(4), 121))
lams = [0.0, 1.0, 2.0]
col = {"regular": "#4d4d4d", "centre": "#1b6ca8", "leaf": "#c8553d"}

fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.1), sharey=True)
rows = []
for a, lam in zip(ax, lams):
    ck = np.array([equilibrium(Gk, np.full(n, B), beta, s, lam, e)[0] for s in sig])
    cs = np.array([equilibrium(Gs, np.full(n, B), beta, s, lam, e) for s in sig])
    a.plot(sig, ck, color=col["regular"], label="regular member")
    a.plot(sig, cs[:, 0], color=col["centre"], label="star centre")
    a.plot(sig, cs[:, 1], color=col["leaf"], label="star leaf")
    a.axvline(1, color="k", lw=.6, ls=":")
    a.set_xscale("log", base=2); a.set_xticks([.25, .5, 1, 2, 4]); a.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
    a.set_title(rf"$\lambda={lam:g}$"); a.set_xlabel(r"$\sigma$")
    for s, k, c0, c1 in zip(sig, ck, cs[:, 0], cs[:, 1]):
        rows.append([lam, s, k, c0, c1])
    if lam > lam_bar(n, beta):
        ss = sigma_star(n, beta, lam)
        if ss < 4: a.axvline(ss, color=col["leaf"], lw=.6, ls="--"); a.text(ss * 1.04, a.get_ylim()[1] * .02 + 1, r"$\sigma^*$", fontsize=8, color=col["leaf"])
ax[0].set_ylabel("equilibrium contribution $c^\\star$"); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig10_positions_sigma_lambda.png", dpi=200); plt.close(fig)
with open("data/positions_sigma_lambda.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lambda", "sigma", "c_regular", "c_centre", "c_leaf"]); w.writerows(rows)

# table for the text
tab = []
for lam in lams:
    for s in [.5, .75, 1, 1.5, 2]:
        ck = equilibrium(Gk, np.full(n, B), beta, s, lam, e)[0]
        cs = equilibrium(Gs, np.full(n, B), beta, s, lam, e)
        pK, pC, pL = [share(B, p, s) for p in (price(1 - beta / 3, 1 + 2 * beta / 3, lam), price(1 - beta / 3, 1 + beta, lam), price(1 - beta / 2, 1 + beta / 3, lam))]
        tab.append([lam, s, ck, cs[0], cs[1], pK, pC, pL])
with open("data/positions_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lambda", "sigma", "c_regular", "c_centre", "c_leaf", "pi_regular", "pi_centre", "pi_leaf"]); w.writerows(tab)
print("lam_bar(3,1.5) =", lam_bar(3, 1.5), " sigma*(lam=3) =", sigma_star(3, 1.5, 3.0), " sigma*(lam=4)=", sigma_star(3, 1.5, 4.0))
for r in tab: print([round(x, 3) for x in r])
