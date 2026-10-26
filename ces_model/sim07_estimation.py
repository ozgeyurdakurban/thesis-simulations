"""Parameter recovery for (B, sigma, lambda) (Figure 16, Table).
Estimator: minimum distance on the three position means of the giving share,
    m_pos = pi(B; p_pos(lambda), sigma),  pos in {regular, centre, leaf},
over (B, sigma, lambda) in (0,.9) x [0.2,6] x [0,6], least squares with a grid of
starting values.  The DGP is richer than the estimating model (heterogeneous types,
trembles, corner censoring), so the exercise reports bias and RMSE of the
estimator under misspecification.  Ng groups per topology, R = 300 replications,
resampled from pools of 20000 groups.  Truth: B_mean = 0.10, sd(B) = 0.02."""
import numpy as np, os, csv, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dgp import *
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(20261029)
beta = BETA
dK, thK, LK = structure(GK, beta); dS, thS, LS = structure(GS, beta)
TH = np.array([thK[0], thS[0], thS[1]]); LAM = np.array([LK[0], LS[0], LS[1]])  # regular, centre, leaf

def model_shares(par):
    B, sigma, lam = par
    return share(B, price(TH, LAM, lam), sigma)

STARTS = [(.1, s, l) for s in (.5, 1., 2.) for l in (.5, 2.)]
LB, UB = [1e-3, .2, 0.], [.9, 6., 6.]
def estimate(m):
    best = None
    for x0 in STARTS:
        r = least_squares(lambda p: model_shares(p) - m, x0, bounds=(LB, UB), xtol=1e-10, ftol=1e-12)
        if best is None or r.cost < best.cost: best = r
    return best.x

R, NG, sd, tau = 300, [30, 100, 300], .02, 1.5
cells = [(.5, 0.), (.5, 1.), (.5, 2.), (2., 0.), (2., 1.), (2., 2.), (1., 1.)]
rows = []
for sigma, lam in cells:
    sp, kp = pool(rng, sigma, lam, .10, sd, tau, R=20000)
    for ng in NG:
        est = np.zeros((R, 3))
        for r in range(R):
            i = rng.integers(0, 20000, ng); k = rng.integers(0, 20000, ng)
            m = np.array([kp["share"][k].mean(), sp["share"][i, 0].mean(), sp["share"][i, 1:].mean()])
            est[r] = estimate(m)
        bnd = ((est[:, 1] < .2 + 1e-6) | (est[:, 1] > 6 - 1e-6)).mean(), ((est[:, 2] < 1e-6) | (est[:, 2] > 6 - 1e-6)).mean()
        tr = np.array([.10, sigma, lam])
        med = np.median(est, 0); q = np.percentile(est, [25, 75], 0)
        rmse = np.sqrt(((est - tr) ** 2).mean(0))
        rows.append([sigma, lam, ng, *med, *q[0], *q[1], *rmse, *bnd])
        print(sigma, lam, ng, "med", med.round(3), "rmse", rmse.round(3), "bnd", np.round(bnd, 2), flush=True)
hdr = ["sigma", "lambda", "n_groups", "B_med", "sigma_med", "lambda_med", "B_q25", "sigma_q25", "lambda_q25", "B_q75", "sigma_q75", "lambda_q75",
       "B_rmse", "sigma_rmse", "lambda_rmse", "sigma_at_bound", "lambda_at_bound"]
with open("data/estimation.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
lab = [f"({s:g},{l:g})" for s, l in cells]
for j, (k, name, true) in enumerate([(1, r"$\hat\sigma$", lambda c: c[0]), (2, r"$\hat\lambda$", lambda c: c[1])]):
    for gi, ng in enumerate(NG):
        sub = [r for r in rows if r[2] == ng]
        x = np.arange(len(cells)) + (gi - 1) * .22
        med = np.array([r[3 + k] for r in sub]); lo = np.array([r[6 + k] for r in sub]); hi = np.array([r[9 + k] for r in sub])
        ax[j].errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", ms=3.5, capsize=2, lw=.9, color=["#c8553d", "#4d4d4d", "#1b6ca8"][gi], label=f"{ng} groups")
    ax[j].plot(np.arange(len(cells)), [true(c) for c in cells], "k_", ms=18, mew=1.5)
    ax[j].set_xticks(np.arange(len(cells))); ax[j].set_xticklabels(lab, rotation=30, fontsize=7); ax[j].set_ylabel(name + " (median, IQR)")
    ax[j].set_xlabel(r"true $(\sigma,\lambda)$"); ax[j].set_yscale("log" if k == 1 else "linear")
ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figures/fig16_parameter_recovery.png", dpi=200); plt.close(fig)
