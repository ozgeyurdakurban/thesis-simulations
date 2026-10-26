"""Power and size of the design (Figure 15, Table).
Test of sigma = 1 (Cobb-Douglas): equal giving shares across the three positions.
For Ng star groups and Ng K3 groups: group-level mean shares by position (centre,
mean of the two leaves, K3 mean); two Welch t-tests (centre vs regular, leaf vs
regular), reject if either |t| exceeds the Bonferroni critical value (alpha/2 each).
A second test: leaf > centre (paired within star groups, one-sided).
Resampling design: pools of 20000 groups per topology from the DGP (types Beta,
tremble N(0, 1.5^2), clipping), Ng groups resampled R = 2000 times."""
import numpy as np, os, csv, warnings
warnings.filterwarnings("ignore")
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgp import *
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(20261028)
alpha, R, lam = .05, 2000, 1.0
NG = [10, 20, 30, 50, 80, 120]

def reject_equal_shares(sp, kp, ng):
    i = rng.integers(0, len(sp["share"]), size=(R, ng)); k = rng.integers(0, len(kp["share"]), size=(R, ng))
    C = sp["share"][:, 0][i]; L = sp["share"][:, 1:].mean(1)[i]; K = kp["share"].mean(1)[k]
    def welch(a, b):
        va, vb = a.var(1, ddof=1) / ng, b.var(1, ddof=1) / ng
        t = (a.mean(1) - b.mean(1)) / np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (ng - 1) + vb ** 2 / (ng - 1))
        return 2 * stats.t.sf(np.abs(t), df)
    p = np.minimum(welch(C, K), welch(L, K))
    return (p < alpha / 2).mean()

def power_leaf_gt_centre(sp, ng):
    i = rng.integers(0, len(sp["obs"]), size=(R, ng))
    d = sp["obs"][:, 1:].mean(1)[i] - sp["obs"][:, 0][i]
    t = d.mean(1) / (d.std(1, ddof=1) / np.sqrt(ng))
    return (stats.t.sf(t, ng - 1) < alpha).mean()

rows = []
for sd in [.02, .04, .06]:
    for sigma in [.5, .75, 1., 1.5, 2.]:
        sp, kp = pool(rng, sigma, lam, .10, sd, 1.5)
        for ng in NG:
            rows.append([sd, sigma, ng, reject_equal_shares(sp, kp, ng), power_leaf_gt_centre(sp, ng)])
        print(sd, sigma, [round(r[3], 2) for r in rows[-len(NG):]], [round(r[4], 2) for r in rows[-len(NG):]])
with open("data/power.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["sd", "sigma", "n_groups", "reject_equal_shares", "reject_leaf_gt_centre"]); w.writerows(rows)

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
cols = {.5: "#c8553d", .75: "#e39b5c", 1.: "#4d4d4d", 1.5: "#6aa3d0", 2.: "#1b6ca8"}
for s in [.5, .75, 1., 1.5, 2.]:
    sub = [r for r in rows if r[0] == .04 and r[1] == s]
    ax[0].plot([r[2] for r in sub], [r[3] for r in sub], "o-", color=cols[s], label=rf"$\sigma={s:g}$")
    ax[1].plot([r[2] for r in sub], [r[4] for r in sub], "o-", color=cols[s])
for a in ax: a.axhline(.05, color="k", lw=.5, ls=":"); a.axhline(.8, color="k", lw=.5, ls="--"); a.set_xlabel("groups per topology"); a.set_ylim(0, 1.02)
ax[0].set_ylabel("rejection rate"); ax[0].set_title(r"test of $\sigma=1$ (equal shares)", fontsize=9); ax[1].set_title("test of leaf $>$ centre", fontsize=9)
ax[1].legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=1, loc="center right"); fig.tight_layout(); fig.savefig("figures/fig15_power.png", dpi=200); plt.close(fig)
