"""
simulations_interior_solutions.py
=================================
Interior-solution ranges of the impact-based Cobb-Douglas networked public-goods
model (e=15, n=3, beta=1.5 unless varied), computed with the CONSTRAINED
equilibrium (clipped best response), which is exact in every regime --- including
the mixed regimes in which some positions are at the corner c=e while others are
interior. (The all-interior closed forms of Section 4 are used only to validate
the solver and to draw the analytic ceiling on the regular graph.)

Model:
  X_i = e - theta_i c_i + (beta/d_i) sum_{j in G_i} c_j,  theta_i = 1 - beta/d_i
  Y_i = (1 + sum_{j in G_i} beta/d_j) c_i                 (impact multiplier; inert under CD)
  U_i = X_i^{A_i} Y_i^{B_i},  A_i = 1 - B_i               (prosocial weight B_i)
  interior FOC: c_i = kappa_i X_i, kappa_i = B_i/(A_i theta_i)
  reduced BR:   c_i = clip( alpha_i + gamma_i sum_{j in G_i} c_j , 0, e )

No lower corner ever binds (B_i log c_i drives marginal moral utility to infinity
as c_i -> 0), so the only active constraint is the upper corner c_i = e.

Figures (./figures):
  fig1_phase_diagrams.png  interior vs full-contribution corner in (B, beta)
  fig2_cstar_vs_B.png      c*(B) at beta=1.5
  fig3_cstar_vs_beta.png   c*(beta) and the shrinking interiority ceiling
  fig4_heterogeneous.png   contribution distribution under heterogeneous B_i
See simulations_grid_mc.py for the deterministic-grid and Monte-Carlo studies.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

E = 15.0
N = 3
BETA0 = 1.5
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)
C_REG, C_CEN, C_PER, C_GREY = "#0072B2", "#D55E00", "#009E73", "#999999"
EPS = 1e-7

D_REGULAR = np.array([3.0, 3.0, 3.0])     # complete = cycle at n=3
D_STAR = np.array([3.0, 2.0, 2.0])        # center, leaf, leaf


# ----------------------------------------------------------------------
# Vectorised constrained equilibrium (beta scalar or column array of shape (G,1))
# ----------------------------------------------------------------------
def equilibrium_batch(B, topo, beta=BETA0, e=E, iters=800, tol=1e-13):
    B = np.asarray(B, float)
    d = (D_REGULAR if topo == "regular" else D_STAR)[None, :]
    beta = np.asarray(beta, float)
    theta = 1.0 - beta / d
    kap = B / ((1.0 - B) * theta)
    denom = 1.0 + kap * theta
    alpha = kap * e / denom
    gamma = kap * (beta / d) / denom
    c = np.full_like(B, e / 2.0)
    for _ in range(iters):
        if topo == "regular":
            S = c.sum(axis=1, keepdims=True) - c
        else:
            S = np.empty_like(c)
            S[:, 0] = c[:, 1] + c[:, 2]
            S[:, 1] = c[:, 0]
            S[:, 2] = c[:, 0]
        c_new = np.clip(alpha + gamma * S, 0.0, e)
        if np.max(np.abs(c_new - c)) < tol:
            return c_new
        c = c_new
    return c


def hom(Bvec, topo, beta=BETA0):
    """Equilibrium for homogeneous B given as a 1-D vector; returns array (len, 3)."""
    Bvec = np.atleast_1d(np.asarray(Bvec, float))
    BB = np.repeat(Bvec[:, None], 3, axis=1)
    bb = np.full((len(Bvec), 1), beta) if np.ndim(beta) == 0 else np.asarray(beta).reshape(-1, 1)
    return equilibrium_batch(BB, topo, beta=bb)


# ----------------------------------------------------------------------
# Closed forms for validation / analytic ceiling (regular graph)
# ----------------------------------------------------------------------
def cstar_regular_closed(B, beta=BETA0, e=E, n=N):
    denom = n - beta - B * beta * (n - 1)
    return np.inf if denom <= 0 else B * n * e / denom


def ceil_regular_closed(beta=BETA0, n=N):
    return (n - beta) / (n + beta * (n - 1))


def validate():
    ok = True
    for B in (0.05, 0.10, 0.20, 0.24):
        c = hom([B], "regular")[0, 0]
        cc = cstar_regular_closed(B)
        if cc < E and abs(c - cc) > 1e-5:
            ok = False; print(f"  [FAIL] regular B={B}: {c:.6f} vs {cc:.6f}")
    # cycle C_3 == complete K_3 (same d=3 structure) — solver identical by construction
    print("VALIDATION:", "PASS" if ok else "FAILURES ABOVE")
    return ok


# ----------------------------------------------------------------------
# Position interiority + ceiling via the constrained solver
# ----------------------------------------------------------------------
POS = {"regular": ("regular", 0), "center": ("star", 0), "periphery": ("star", 1)}


def value(Bvec, key, beta=BETA0):
    topo, idx = POS[key]
    return hom(Bvec, topo, beta=beta)[:, idx]


def ceiling(beta, key, lo=1e-4, hi=0.97):
    f = lambda B: EPS < value(np.array([B]), key, beta=beta)[0] < E - 1e-4
    if not f(lo):
        return 0.0
    if f(hi):
        return hi
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# FIGURE 1 — phase diagrams (B, beta), constrained interiority per position
# ----------------------------------------------------------------------
def fig1_phase(nB=240, nbeta=240):
    Bs = np.linspace(0.004, 0.6, nB)
    betas = np.linspace(1.02, 1.98, nbeta)
    # one big batch per topology: rows = (beta, B) combos
    BB, Bgrid = np.meshgrid(betas, Bs, indexing="ij")  # (nbeta, nB)
    beta_col = BB.reshape(-1, 1)
    Bvec = Bgrid.reshape(-1)
    panels = [("Regular (complete / cycle)", "regular", C_REG),
              ("Star center", "center", C_CEN),
              ("Star periphery", "periphery", C_PER)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)
    for ax, (title, key, col) in zip(axes, panels):
        topo, idx = POS[key]
        Bmat = np.repeat(Bvec[:, None], 3, axis=1)
        c = equilibrium_batch(Bmat, topo, beta=beta_col)[:, idx].reshape(nbeta, nB)
        Z = ((c > EPS) & (c < E - 1e-4)).astype(float)
        ax.imshow(Z, origin="lower", aspect="auto",
                  extent=[Bs[0], Bs[-1], betas[0], betas[-1]],
                  cmap=mcolors.ListedColormap(["#ffffff", col]), vmin=0, vmax=1, alpha=0.9)
        ax.plot([ceiling(b, key) for b in betas], betas, color="black", lw=1.5)
        ax.axhline(BETA0, color=C_GREY, ls="--", lw=1.2)
        ax.axhline(1.2, color=C_GREY, ls=":", lw=1.0)
        ax.text(0.595, BETA0 + 0.012, r"$\beta=1.5$", ha="right", fontsize=8)
        ax.text(0.595, 1.2 + 0.012, r"$\beta=1.2$", ha="right", fontsize=8, color=C_GREY)
        ax.set_title(title, fontsize=11); ax.set_xlabel(r"prosocial weight $B$")
    axes[0].set_ylabel(r"pool productivity $\beta$")
    fig.suptitle(r"Interior region (coloured, $0<c^\star<e$) vs full-contribution corner $c^\star=e$ (white)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_phase_diagrams.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# FIGURE 2 — c*(B) at beta = 1.5 (constrained)
# ----------------------------------------------------------------------
def fig2_cstar_vs_B(nB=700):
    Bs = np.linspace(0.001, 0.6, nB)
    reg = hom(Bs, "regular")[:, 0]
    star = hom(Bs, "star")
    cen, per = star[:, 0], star[:, 1]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for arr, col, lab, key in [(reg, C_REG, "Regular (complete / cycle)", "regular"),
                               (cen, C_CEN, "Star center", "center"),
                               (per, C_PER, "Star periphery", "periphery")]:
        ax.plot(Bs, arr, color=col, lw=2.2, label=lab)
        bstar = ceiling(BETA0, key)
        if 0 < bstar < Bs[-1]:
            ax.axvline(bstar, color=col, ls=":", lw=1.0, alpha=0.7)
            ax.text(bstar, 0.4, f"$B^\\star={bstar:.2f}$", color=col, fontsize=8, rotation=90, va="bottom")
    ax.axhline(E, color=C_GREY, ls="--", lw=1.2)
    ax.text(0.0, E - 0.4, r"$c^\star=e=15$ (corner)", fontsize=9, color=C_GREY, va="top")
    ax.set_xlabel(r"prosocial weight $B$"); ax.set_ylabel(r"equilibrium contribution $c^\star$")
    ax.set_title(r"Comparative statics in $B$ at $\beta=1.5,\ e=15,\ n=3$")
    ax.set_ylim(0, E + 1); ax.legend(fontsize=10); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_cstar_vs_B.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# FIGURE 3 — c*(beta) and the shrinking interiority ceiling (constrained)
# ----------------------------------------------------------------------
def fig3_cstar_vs_beta(nbeta=260):
    betas = np.linspace(1.02, 1.98, nbeta)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for B, ls in [(0.10, "-"), (0.20, "--"), (0.30, ":")]:
        c = hom(np.full(nbeta, B), "regular", beta=betas)[:, 0]
        ax.plot(betas, c, lw=2.0, ls=ls, color=C_REG, label=f"$B={B:.2f}$")
    ax.axhline(E, color=C_GREY, ls="--", lw=1.2)
    for bx in (1.2, 1.5):
        ax.axvline(bx, color=C_GREY, ls=":", lw=1.0)
        ax.text(bx, 0.4, rf"$\beta={bx}$", rotation=90, fontsize=8, va="bottom", ha="right", color=C_GREY)
    ax.set_title(r"Regular graph: $c^\star$ rises in $\beta$, then corners")
    ax.set_xlabel(r"pool productivity $\beta$"); ax.set_ylabel(r"$c^\star$")
    ax.set_ylim(0, E + 1); ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax = axes[1]
    for key, col, lab in [("regular", C_REG, "Regular"), ("center", C_CEN, "Star center"),
                          ("periphery", C_PER, "Star periphery")]:
        ax.plot(betas, [ceiling(b, key) for b in betas], lw=2.2, color=col, label=lab)
    for bx in (1.2, 1.5):
        ax.axvline(bx, color=C_GREY, ls=":", lw=1.0)
    ax.text(1.5, 0.01, r"$\beta=1.5$", rotation=90, fontsize=8, va="bottom", ha="right", color=C_GREY)
    ax.set_title(r"Interiority ceiling $B^\star(\beta)$ (interior below each curve)")
    ax.set_xlabel(r"pool productivity $\beta$"); ax.set_ylabel(r"largest interior $B$")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_cstar_vs_beta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# FIGURE 4 — heterogeneous B_i predicted contribution distribution
# ----------------------------------------------------------------------
def fig4_heterogeneous(seed=20260620, ngroups=6000):
    rng = np.random.default_rng(seed)
    B = rng.beta(2.0, 9.0, size=(ngroups, N))  # mean ~0.18
    cs = equilibrium_batch(B, "star")
    cr = equilibrium_batch(B, "regular")
    per, cen, reg = cs[:, 1:].ravel(), cs[:, 0], cr.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    bins = np.linspace(0, E, 31)
    ax = axes[0]
    ax.hist(per, bins=bins, color=C_PER, alpha=0.6, density=True, label="Star periphery")
    ax.hist(cen, bins=bins, color=C_CEN, alpha=0.6, density=True, label="Star center")
    ax.set_title(r"Star: predicted $c^\star$ distribution, $B_i\sim\mathrm{Beta}(2,9)$")
    ax.set_xlabel(r"$c^\star$"); ax.set_ylabel("density"); ax.legend(fontsize=9)
    ax.text(0.98, 0.82, f"mean periphery {per.mean():.2f}\nmean center {cen.mean():.2f}\n"
                        f"corner share {np.mean(np.r_[per,cen] >= E - 1e-6) * 100:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=C_GREY))
    ax = axes[1]
    ax.hist(reg, bins=bins, color=C_REG, alpha=0.7, density=True, label="Regular (complete/cycle)")
    ax.set_title(r"Regular: predicted $c^\star$ distribution")
    ax.set_xlabel(r"$c^\star$"); ax.set_ylabel("density"); ax.legend(fontsize=9)
    ax.text(0.98, 0.82, f"mean {reg.mean():.2f}\ncorner share {np.mean(reg >= E - 1e-6) * 100:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=C_GREY))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_heterogeneous.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Validating constrained solver against closed forms ...")
    validate()
    print(f"Interiority ceilings at beta={BETA0}: "
          f"regular {ceiling(BETA0,'regular'):.3f}, center {ceiling(BETA0,'center'):.3f}, "
          f"periphery {ceiling(BETA0,'periphery'):.3f}")
    print("Generating figures ...")
    fig1_phase(); fig2_cstar_vs_B(); fig3_cstar_vs_beta(); fig4_heterogeneous()
    print("Figures written to", FIGDIR)
