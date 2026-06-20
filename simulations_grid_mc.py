"""
simulations_grid_mc.py
======================
Two complementary, explicitly labelled simulation designs for the interior-
solution analysis of the impact-based Cobb-Douglas networked public-goods model
(e=15, n=3, beta=1.5 unless varied):

  PART A. DETERMINISTIC GRID
    A1. Fine homogeneous grid over (B, beta): interiority ceilings B*(beta) per
        position (regular / star center / star periphery).
    A2. Exhaustive ASYMMETRIC grid over preference triples (B1,B2,B3): for each
        topology, the share of profiles that are fully interior, the share with
        any corner, and a monotonicity check (does the contribution ranking
        match the prosocial-weight ranking on the symmetric topology?).
    Outputs: grid_ceilings.csv, grid_asymmetric_summary.csv

  PART B. MONTE CARLO
    Random preference profiles B_i drawn i.i.d. from several Beta distributions;
    G groups per replication, R replications to obtain Monte-Carlo standard
    errors on the summary statistics (mean contribution by position, corner
    probability, P(periphery > center), mean material welfare). A convergence
    diagnostic tracks the running estimate against the number of draws.
    Outputs: montecarlo_summary.csv, fig5_montecarlo_convergence.png

Equilibria are computed by VECTORISED clipped best-response iteration (batches of
groups solved simultaneously). The solver is validated against the closed forms
in simulations_interior_solutions.py; here we re-check one identity at startup.
"""

import os
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
E = 15.0
N = 3
BETA0 = 1.5
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
DATADIR = os.path.join(HERE, "data")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)
EPS = 1e-9

C_REG, C_CEN, C_PER, C_GREY = "#0072B2", "#D55E00", "#009E73", "#999999"

# d_i per position for each topology (n=3): index 0 is the focal/center
D_REGULAR = np.array([3.0, 3.0, 3.0])     # complete = cycle at n=3
D_STAR = np.array([3.0, 2.0, 2.0])        # center, leaf, leaf


# ----------------------------------------------------------------------
# Vectorised clipped best-response solver
#   B : array (G, 3) of prosocial weights
#   topo : "regular" or "star"
# returns c : array (G, 3) of equilibrium contributions
# ----------------------------------------------------------------------
def equilibrium_batch(B, topo, beta=BETA0, e=E, iters=600, tol=1e-12):
    B = np.asarray(B, float)
    d = (D_REGULAR if topo == "regular" else D_STAR)[None, :]
    theta = 1.0 - beta / d
    kap = B / ((1.0 - B) * theta)
    denom = 1.0 + kap * theta
    alpha = kap * e / denom
    gamma = kap * (beta / d) / denom
    c = np.full_like(B, e / 2.0)
    for _ in range(iters):
        if topo == "regular":
            tot = c.sum(axis=1, keepdims=True)
            S = tot - c                      # neighbour sum = total - own
        else:  # star: center sees both leaves; each leaf sees the center
            S = np.empty_like(c)
            S[:, 0] = c[:, 1] + c[:, 2]
            S[:, 1] = c[:, 0]
            S[:, 2] = c[:, 0]
        c_new = np.clip(alpha + gamma * S, 0.0, e)
        if np.max(np.abs(c_new - c)) < tol:
            return c_new
        c = c_new
    return c


def welfare(c, topo, beta=BETA0, e=E):
    """Material welfare W = n e + sum_i (sigma_i - 1) c_i, sigma_i = sum_{k in N[i]} beta/d_k."""
    if topo == "regular":
        sigma = np.array([beta, beta, beta])               # regular: sigma_i = beta
    else:
        s_c = beta / 3 + 2 * (beta / 2)                     # center
        s_l = beta / 2 + beta / 3                           # leaf
        sigma = np.array([s_c, s_l, s_l])
    return N * e + (c * (sigma[None, :] - 1.0)).sum(axis=1)


# ----------------------------------------------------------------------
# Startup self-check against the closed form on the regular graph
# ----------------------------------------------------------------------
def _selfcheck():
    B = 0.20
    c = equilibrium_batch(np.array([[B, B, B]]), "regular")[0, 0]
    closed = B * N * E / (N - BETA0 - B * BETA0 * (N - 1))
    assert abs(c - closed) < 1e-6, (c, closed)
    # C_3 = K_3: star with equal everything is NOT the same; only regular check here
    print(f"  self-check OK: regular B=0.20 -> c*={c:.4f} (closed {closed:.4f})")


# ======================================================================
# PART A. DETERMINISTIC GRID
# ======================================================================
def position_value_batch(Bgrid, topo, position, beta):
    """Equilibrium value at a position for a homogeneous-B vector (1-D Bgrid)."""
    BB = np.repeat(Bgrid[:, None], 3, axis=1)
    c = equilibrium_batch(BB, topo, beta=beta)
    return c[:, position]


def ceiling(beta, topo, position, lo=1e-4, hi=0.95):
    """Largest homogeneous B that keeps the position interior, by bisection."""
    def interior(B):
        v = equilibrium_batch(np.array([[B, B, B]]), topo, beta=beta)[0, position]
        return EPS < v < E - EPS
    if not interior(lo):
        return 0.0
    if interior(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if interior(mid) else (lo, mid)
    return 0.5 * (lo + hi)


def gridA1_ceilings():
    betas = np.round(np.arange(1.05, 1.96, 0.05), 2)
    rows = [("beta", "regular", "star_center", "star_periphery")]
    for b in betas:
        rows.append((b,
                     round(ceiling(b, "regular", 0), 4),
                     round(ceiling(b, "star", 0), 4),
                     round(ceiling(b, "star", 1), 4)))
    with open(os.path.join(DATADIR, "grid_ceilings.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    # report the operating point
    b = BETA0
    print(f"  [A1] interiority ceilings at beta={b}: "
          f"regular={ceiling(b,'regular',0):.3f}, center={ceiling(b,'star',0):.3f}, "
          f"periphery={ceiling(b,'star',1):.3f}")
    return rows


def gridA2_asymmetric(step=0.04, bmax=0.46):
    """Exhaustive grid over preference triples; interiority and monotonicity."""
    vals = np.round(np.arange(step, bmax + 1e-9, step), 4)
    G1, G2, G3 = np.meshgrid(vals, vals, vals, indexing="ij")
    B = np.column_stack([G1.ravel(), G2.ravel(), G3.ravel()])  # (M, 3)
    summary = [("topology", "n_profiles", "share_all_interior",
                "share_any_corner", "share_monotone_in_B")]
    out = {}
    for topo in ("regular", "star"):
        c = equilibrium_batch(B, topo)
        interior_mask = (c > EPS) & (c < E - EPS)
        all_interior = interior_mask.all(axis=1)
        any_corner = (c >= E - 1e-6).any(axis=1)
        if topo == "regular":
            # symmetric topology: contribution ranking should match B ranking
            mono = (np.argsort(np.argsort(B, axis=1), axis=1)
                    == np.argsort(np.argsort(c, axis=1), axis=1)).all(axis=1)
            share_mono = mono.mean()
        else:
            share_mono = float("nan")  # positions differ; ranking not by B alone
        summary.append((topo, len(B), round(all_interior.mean(), 4),
                        round(any_corner.mean(), 4),
                        round(share_mono, 4) if share_mono == share_mono else "n/a"))
        out[topo] = dict(share_interior=all_interior.mean(),
                         share_corner=any_corner.mean())
    with open(os.path.join(DATADIR, "grid_asymmetric_summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(summary)
    print(f"  [A2] asymmetric grid: {len(B)} profiles "
          f"(B in [{vals[0]:.2f},{vals[-1]:.2f}], step {step})")
    for topo in ("regular", "star"):
        print(f"        {topo:8s}: all-interior {out[topo]['share_interior']*100:5.1f}%, "
              f"any-corner {out[topo]['share_corner']*100:5.1f}%")
    return summary


# ======================================================================
# PART B. MONTE CARLO
# ======================================================================
DISTS = {
    "Beta(2,18) mean .10": (2.0, 18.0),
    "Beta(2,9) mean .18": (2.0, 9.0),
    "Beta(2,5) mean .29": (2.0, 5.0),
}


def mc_statistics(c_reg, c_star):
    """Return a dict of summary statistics for one batch of groups."""
    per = c_star[:, 1:].ravel()
    cen = c_star[:, 0]
    return dict(
        mean_reg=c_reg.mean(),
        mean_star_center=cen.mean(),
        mean_star_periphery=per.mean(),
        corner_reg=(c_reg >= E - 1e-6).mean(),
        corner_star=((c_star >= E - 1e-6).mean()),
        p_periphery_gt_center=np.mean(c_star[:, 1:].mean(axis=1) > c_star[:, 0]),
        welfare_reg=welfare(c_reg, "regular").mean(),
        welfare_star=welfare(c_star, "star").mean(),
    )


def montecarlo(G=6000, R=30, seed0=20260620):
    rows = [("distribution", "statistic", "estimate", "mc_std_error")]
    for name, (a, b) in DISTS.items():
        reps = {k: [] for k in
                ("mean_reg", "mean_star_center", "mean_star_periphery",
                 "corner_reg", "corner_star", "p_periphery_gt_center",
                 "welfare_reg", "welfare_star")}
        for r in range(R):
            rng = np.random.default_rng(seed0 + r)
            B = rng.beta(a, b, size=(G, N))
            c_reg = equilibrium_batch(B, "regular", iters=400)
            c_star = equilibrium_batch(B, "star", iters=400)
            st = mc_statistics(c_reg, c_star)
            for k in reps:
                reps[k].append(st[k])
        for k, v in reps.items():
            v = np.array(v)
            rows.append((name, k, round(v.mean(), 4), round(v.std(ddof=1) / np.sqrt(R), 4)))
        print(f"  [MC] {name}: periphery {np.mean(reps['mean_star_periphery']):.3f}, "
              f"center {np.mean(reps['mean_star_center']):.3f}, "
              f"regular {np.mean(reps['mean_reg']):.3f}, "
              f"star-corner {np.mean(reps['corner_star'])*100:.1f}%, "
              f"P(per>cen) {np.mean(reps['p_periphery_gt_center'])*100:.1f}%")
    with open(os.path.join(DATADIR, "montecarlo_summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return rows


def mc_convergence(maxdraws=60000, seed=7, dist=("Beta(2,9) mean .18", (2.0, 9.0))):
    """Running Monte-Carlo estimate of mean periphery contribution vs draws."""
    name, (a, b) = dist
    rng = np.random.default_rng(seed)
    B = rng.beta(a, b, size=(maxdraws, N))
    c_star = equilibrium_batch(B, "star", iters=400)
    per = c_star[:, 1:].mean(axis=1)            # per-group periphery mean
    running = np.cumsum(per) / np.arange(1, maxdraws + 1)
    grid = np.unique(np.linspace(50, maxdraws, 400).astype(int))
    se = np.array([per[:g].std(ddof=1) / np.sqrt(g) for g in grid])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(np.arange(1, maxdraws + 1), running, color=C_PER, lw=1.6,
            label="running mean (star periphery)")
    ax.fill_between(grid, running[grid - 1] - 1.96 * se, running[grid - 1] + 1.96 * se,
                    color=C_PER, alpha=0.25, label="95% MC band")
    ax.axhline(running[-1], color=C_GREY, ls="--", lw=1.0)
    ax.set_xlabel("number of simulated groups")
    ax.set_ylabel(r"mean periphery $c^\star$")
    ax.set_title(rf"Monte-Carlo convergence, $B_i\sim${name}, $\beta=1.5$")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_montecarlo_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [MC] convergence: estimate {running[-1]:.4f} at {maxdraws} draws")


# ======================================================================
if __name__ == "__main__":
    print("Self-check ...")
    _selfcheck()
    print("PART A — deterministic grid ...")
    gridA1_ceilings()
    gridA2_asymmetric()
    print("PART B — Monte Carlo ...")
    montecarlo()
    mc_convergence()
    print("Data written to", DATADIR, "; figure to", FIGDIR)
