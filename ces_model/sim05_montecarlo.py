"""Monte Carlo over heterogeneous prosocial weights (Figure 14, Table).
B_i ~ Beta(mean 0.10, sd s), lambda = 1, beta = 1.5, e = 15, no tremble.
Outcomes by topology over R = 20000 groups: mean contributions with MC standard
errors, Pr(c_leaf > c_centre) within star groups, Pr(any agent at the corner),
and the mean social efficiency SE = sum_i X_i relative to common-B benchmark."""
import numpy as np, os, csv, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgp import *
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(20261027)
lam, R, mean = 1.0, 20000, .10

def SE(c, G):
    d, th, Lam = structure(G, BETA)
    s = Lam - 1 + BETA / d
    return N * E + ((s - 1) * c).sum(1)

rows = []; grid_sd = [0.01, 0.02, 0.04, 0.06, 0.08]
for sigma in [.5, 1., 2.]:
    ck0 = equilibrium(GK, np.full(N, mean), BETA, sigma, lam, E)
    cs0 = equilibrium(GS, np.full(N, mean), BETA, sigma, lam, E)
    for sd in grid_sd:
        Bs = draw_types(rng, R, mean, sd); Bk = draw_types(rng, R, mean, sd)
        cs = equilibrium(GS, Bs, BETA, sigma, lam, E); ck = equilibrium(GK, Bk, BETA, sigma, lam, E)
        mk, mc, ml = ck.mean(), cs[:, 0].mean(), cs[:, 1:].mean()
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
        pLC = (cs[:, 1:].mean(1) > cs[:, 0]).mean()
        corner_s = (cs.max(1) > E - 1e-9).mean(); corner_k = (ck.max(1) > E - 1e-9).mean()
        SEs, SEk = SE(cs, GS).mean(), SE(ck, GK).mean()
        rows.append([sigma, sd, mk, se(ck.mean(1)), mc, se(cs[:, 0]), ml, se(cs[:, 1:].mean(1)), pLC,
                     (cs[:, 0] - ck.mean(1)[:len(cs)]).mean(), corner_s, corner_k, SEs, SEk,
                     SE(cs, GS).mean() - SE(np.broadcast_to(cs0, cs.shape), GS).mean(),
                     mk - ck0[0]])
hdr = ["sigma", "sd", "c_reg", "se_reg", "c_cen", "se_cen", "c_leaf", "se_leaf", "P_leaf_gt_centre", "gap_centre_minus_reg",
       "P_corner_star", "P_corner_K3", "SE_star", "SE_K3", "dSE_star_vs_commonB", "dc_reg_vs_commonB"]
with open("data/montecarlo.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
for r in rows: print([round(x, 3) for x in r])

fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.0))
cols = {.5: "#c8553d", 1.: "#4d4d4d", 2.: "#1b6ca8"}
for s in [.5, 1., 2.]:
    sub = [r for r in rows if r[0] == s]; x = [r[1] for r in sub]
    ax[0].plot(x, [r[8] for r in sub], "o-", color=cols[s], label=rf"$\sigma={s:g}$")
    ax[1].plot(x, [r[10] for r in sub], "o-", color=cols[s])
    ax[2].plot(x, [r[4] - r[2] for r in sub], "o-", color=cols[s])
ax[0].set_ylabel(r"$\Pr(\bar c_{\rm leaf}>c_{\rm centre})$"); ax[1].set_ylabel("Pr(some agent at corner), star")
ax[2].set_ylabel(r"$\bar c_{\rm centre}-\bar c_{\rm regular}$")
for a in ax: a.set_xlabel("sd of $B_i$")
ax[0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig("figures/fig14_montecarlo_heterogeneity.png", dpi=200); plt.close(fig)
