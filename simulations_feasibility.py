"""
simulations_feasibility.py
==========================
Feasible parameter regions of the impact-based Cobb-Douglas networked model:
the set of (B, beta) at which ALL of the model's conditions and assumptions hold
JOINTLY, so that the theory delivers a well-posed interior Nash equilibrium of a
genuine local public-goods dilemma. (e=15, n=3.) Self-contained.

Conditions mapped (per network position with closed-neighborhood size d_i):
  (C1) Dilemma / Assumption 3.3:        1 < beta < d_i        (theta_i = 1-beta/d_i in (0,1))
  (C2) Social productivity of giving:   sigma_i > 1,  sigma_i = sum_{k in N[i]} beta/d_k
                                         regular: beta;  star center: 4beta/3;  leaf: 5beta/6
  (C3) Interior Nash equilibrium:        0 < c_i^* < e
                                         <=> B < B*(beta) = (d_i-beta)/(d_i+beta(d_i-1))
                                         (periphery via the constrained best-response solver)
  (C4) Admissible preferences:           B in (0,1),  A = 1-B > 0
e enters only as a scale (c_i^* proportional to e), so the feasible region in
(B, beta) is independent of e.

Outputs (./figures):
  fig6_feasible_by_position.png   feasible region in (B,beta) for each position
  fig7_feasible_experiment.png    joint feasibility for the 3-person star design
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E, N, BETA0 = 15.0, 3, 1.5
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)
C_REG, C_CEN, C_PER, C_GREY, C_FEAS = "#0072B2", "#D55E00", "#009E73", "#444444", "#7fbf7b"
D_STAR = np.array([3.0, 2.0, 2.0])   # center, leaf, leaf


def star_equilibrium(B, beta=BETA0, e=E, iters=800, tol=1e-13):
    """Constrained best-response equilibrium of the homogeneous-B star (returns c_center, c_leaf)."""
    d = D_STAR[None, :]
    theta = 1.0 - beta / d
    BB = np.full((1, 3), B)
    kap = BB / ((1.0 - BB) * theta)
    denom = 1.0 + kap * theta
    alpha = kap * e / denom
    gamma = kap * (beta / d) / denom
    c = np.full((1, 3), e / 2.0)
    for _ in range(iters):
        S = np.empty_like(c)
        S[:, 0] = c[:, 1] + c[:, 2]
        S[:, 1] = c[:, 0]
        S[:, 2] = c[:, 0]
        cn = np.clip(alpha + gamma * S, 0.0, e)
        if np.max(np.abs(cn - c)) < tol:
            c = cn
            break
        c = cn
    return c[0, 0], c[0, 1]


def ceiling_periphery(beta, lo=1e-4, hi=0.97):
    """Largest interior B for the star periphery at given beta (constrained)."""
    f = lambda B: 1e-7 < star_equilibrium(B, beta)[1] < E - 1e-4
    if beta <= 1.0 or beta >= 2.0 or not f(lo):
        return 0.0
    if f(hi):
        return hi
    for _ in range(55):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if f(m) else (lo, m)
    return 0.5 * (lo + hi)


def interiority_ceiling(beta, key):
    if key in ("regular", "center"):     # d=3
        return max(0.0, (3.0 - beta) / (3.0 + 2.0 * beta))
    return ceiling_periphery(beta)        # periphery d=2


POSCONF = {
    "Regular (complete / cycle)": dict(key="regular", d=3, b_soc=1.0, col=C_REG),
    "Star center":                dict(key="center",  d=3, b_soc=0.75, col=C_CEN),
    "Star periphery":             dict(key="periphery", d=2, b_soc=1.2, col=C_PER),
}


def fig6_by_position(nB=420, nbeta=420):
    Bs = np.linspace(0.001, 0.6, nB)
    betas = np.linspace(0.85, 3.15, nbeta)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True, sharey=True)
    for ax, (title, cfg) in zip(axes, POSCONF.items()):
        key, d, b_soc = cfg["key"], cfg["d"], cfg["b_soc"]
        b_lo = max(1.0, b_soc)
        Z = np.zeros((nbeta, nB)); ceil_curve = np.full(nbeta, np.nan)
        for ib, b in enumerate(betas):
            if b <= b_lo or b >= d:
                continue
            Bstar = interiority_ceiling(b, key)
            ceil_curve[ib] = Bstar
            Z[ib, :] = (Bs < Bstar) & (Bs > 0)
        ax.contourf(Bs, betas, Z, levels=[0.5, 1.5], colors=[cfg["col"]], alpha=0.30)
        ax.axhline(1.0, color=C_GREY, lw=1.2)
        ax.text(0.59, 1.02, r"dilemma lower $\beta>1$", ha="right", fontsize=7.5, color=C_GREY)
        ax.axhline(d, color=C_GREY, lw=1.2)
        ax.text(0.59, d - 0.10, rf"dilemma upper $\beta<d_i={d:g}$", ha="right", fontsize=7.5, color=C_GREY)
        if b_soc > 1.0:
            ax.axhline(b_soc, color="black", lw=1.4, ls="-.")
            ax.text(0.01, b_soc + 0.03, rf"social productivity $\sigma_i>1\ (\beta>{b_soc:g})$",
                    fontsize=7.5, color="black")
        good = np.isfinite(ceil_curve)
        ax.plot(ceil_curve[good], betas[good], color=cfg["col"], lw=2.0)
        ax.text(interiority_ceiling(1.7, key) + 0.012, 1.72, r"interiority $B<B^\star(\beta)$",
                fontsize=7.5, color=cfg["col"], rotation=-70)
        ax.axhline(BETA0, color="red", lw=1.0, ls=":")
        ax.text(0.59, BETA0 + 0.03, r"$\beta=1.5$", ha="right", fontsize=8, color="red")
        ax.set_title(title, fontsize=11); ax.set_xlabel(r"prosocial weight $B$")
        ax.set_xlim(0, 0.6); ax.set_ylim(0.85, 3.15)
    axes[0].set_ylabel(r"pool productivity $\beta$")
    fig.suptitle(r"Feasible region (shaded): dilemma $1<\beta<d_i$, social productivity $\sigma_i>1$, "
                 r"and interior equilibrium $0<c^\star<e$ all hold", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_feasible_by_position.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_experiment(nB=500, nbeta=500):
    Bs = np.linspace(0.001, 0.45, nB)
    betas = np.linspace(0.95, 2.15, nbeta)
    Z = np.zeros((nbeta, nB)); ceil_per = np.full(nbeta, np.nan)
    for ib, b in enumerate(betas):
        if not (1.2 < b < 2.0):
            continue
        Bstar = interiority_ceiling(b, "periphery")
        ceil_per[ib] = Bstar
        Z[ib, :] = (Bs < Bstar)
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.contourf(Bs, betas, Z, levels=[0.5, 1.5], colors=[C_FEAS], alpha=0.55)
    ax.axhline(2.0, color=C_GREY, lw=1.5)
    ax.text(0.44, 1.95, r"periphery dilemma $\beta<d_{\rm leaf}=2$", ha="right", fontsize=8.5, color=C_GREY)
    ax.axhline(1.2, color="black", lw=1.6, ls="-.")
    ax.text(0.005, 1.13, r"periphery social productivity $\sigma_{\rm leaf}>1\ (\beta>6/5)$",
            fontsize=8.5, color="black")
    ax.axhline(1.0, color=C_GREY, lw=1.0, ls="--")
    ax.text(0.005, 0.93 + 0.0, r"dilemma lower $\beta>1$", fontsize=8.5, color=C_GREY)
    good = np.isfinite(ceil_per)
    ax.plot(ceil_per[good], betas[good], color=C_PER, lw=2.4)
    ax.text(0.30, 1.42, r"interiority $B<B^\star_{\rm periphery}(\beta)$", color=C_PER, fontsize=9, rotation=-70)
    bstar15 = interiority_ceiling(BETA0, "periphery")
    ax.axhline(BETA0, color="red", lw=1.2, ls=":")
    ax.plot([0, bstar15], [BETA0, BETA0], color="red", lw=3.5, solid_capstyle="butt", alpha=0.9)
    ax.text(0.445, BETA0 + 0.02, rf"$\beta=1.5$: feasible $B\in(0,\,{bstar15:.3f})$",
            ha="right", va="bottom", fontsize=9, color="red")
    ax.set_xlabel(r"prosocial weight $B$ (homogeneous)")
    ax.set_ylabel(r"pool productivity $\beta$")
    ax.set_title("Joint feasible region for the 3-person star experiment\n"
                 "(all positions: dilemma, socially productive, interior)")
    ax.set_xlim(0, 0.45); ax.set_ylim(0.95, 2.15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_feasible_experiment.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return bstar15


if __name__ == "__main__":
    print("Building feasibility figures ...")
    fig6_by_position()
    bs = fig7_experiment()
    print(f"  beta=1.5 joint-feasible B interval (binding periphery): (0, {bs:.4f})")
    print("Figures written to", FIGDIR)
