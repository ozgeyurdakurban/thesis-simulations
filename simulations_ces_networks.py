"""
CES on networks: equilibrium contributions, interiority ceilings, and the
act-based vs impact-based comparison as functions of the elasticity of
substitution sigma = 1/(1-rho).

Companion to the thesis section "The CES Specification on Networks".

Under CES the interior first-order condition is c_i = kappa_i * X_i with

    kappa_i(sigma) = ( a_i / (delta_i * theta_i) )**sigma * Lambda_i**(sigma-1),

where a_i = 1 - delta_i is the prosocial weight, theta_i = 1 - beta/d_i the
local net marginal cost, and Lambda_i = 1 + sum_{j in G_i} beta/d_j the
position-dependent impact multiplier. The act-based benchmark sets
Lambda_i = 1. At sigma = 1 (Cobb-Douglas) the two coincide: the multiplier
is behaviorally neutral only at unit elasticity.

Outputs:
    figures/fig8_ces_sigma_contributions.png
    figures/fig9_ces_sigma_ceilings.png
    data/ces_sigma_table.csv

Parameters mirror the experimental design: e = 15, n = 3, beta = 1.5.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

E, N, BETA = 15.0, 3, 1.5
TOL, MAXIT = 1e-12, 200000

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)


def kappa(a, theta, lam, sigma):
    """CES decision coefficient kappa_i(sigma); lam=1 gives the act-based model."""
    return (a / ((1.0 - a) * theta)) ** sigma * lam ** (sigma - 1.0)


def star_equilibrium(a, beta=BETA, sigma=1.0, impact=True, e=E, n=N):
    """Clipped best-response iteration on the star (center + n-1 leaves),
    homogeneous prosocial weight a. Returns (c_center, c_leaf)."""
    th_c, th_l = 1.0 - beta / n, 1.0 - beta / 2.0
    lam_c = 1.0 + (n - 1) * beta / 2.0 if impact else 1.0
    lam_l = 1.0 + beta / n if impact else 1.0
    k_c, k_l = kappa(a, th_c, lam_c, sigma), kappa(a, th_l, lam_l, sigma)
    c_c = c_l = 1.0
    for _ in range(MAXIT):
        n_c = min(e, k_c * (e + (beta / n) * (n - 1) * c_l) / (1.0 + k_c * th_c))
        n_l = min(e, k_l * (e + (beta / 2.0) * c_c) / (1.0 + k_l * th_l))
        if abs(n_c - c_c) < TOL and abs(n_l - c_l) < TOL:
            break
        c_c, c_l = n_c, n_l
    return c_c, c_l


def regular_equilibrium(a, beta=BETA, sigma=1.0, impact=True, e=E, d=3):
    """Symmetric equilibrium on a regular topology with closed-neighborhood
    size d (complete K3 / cycle C3): c* = kappa e / (1 + kappa (1 - beta))."""
    th = 1.0 - beta / d
    lam = 1.0 + (d - 1) * beta / d if impact else 1.0
    k = kappa(a, th, lam, sigma)
    c = k * e / (1.0 + k * (1.0 - beta))
    return min(e, c)


def regular_ceiling(beta, sigma, impact=True, d=3):
    """Interiority ceiling a*(sigma) on a regular topology: kappa < 1/beta."""
    th = 1.0 - beta / d
    lam = 1.0 + (d - 1) * beta / d if impact else 1.0
    w = (beta * lam ** (sigma - 1.0)) ** (-1.0 / sigma)
    return th * w / (1.0 + th * w)


def periphery_ceiling(beta, sigma, impact=True):
    """Interiority ceiling for the star periphery from the coupled system."""
    lo, hi = 1e-3, 0.75
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        _, c_l = star_equilibrium(mid, beta, sigma, impact)
        if c_l < E - 1e-9:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- figure 8
sigmas = np.linspace(0.25, 3.0, 120)
a0 = 0.10
cc_imp, cl_imp, cc_act, cl_act, creg_imp, creg_act = [], [], [], [], [], []
for s in sigmas:
    cc, cl = star_equilibrium(a0, sigma=s, impact=True)
    ca, la = star_equilibrium(a0, sigma=s, impact=False)
    cc_imp.append(cc); cl_imp.append(cl)
    cc_act.append(ca); cl_act.append(la)
    creg_imp.append(regular_equilibrium(a0, sigma=s, impact=True))
    creg_act.append(regular_equilibrium(a0, sigma=s, impact=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ax = axes[0]
ax.plot(sigmas, cl_imp, label="periphery", lw=2)
ax.plot(sigmas, cc_imp, label="center", lw=2)
ax.plot(sigmas, creg_imp, label="regular", lw=2)
ax.axvline(1.0, ls=":", c="k", lw=1)
ax.set_title("Impact-based ($Y_i=\\Lambda_i c_i$)")
ax.set_xlabel("elasticity of substitution $\\sigma$")
ax.set_ylabel("equilibrium contribution $c^\\star$")
ax.legend(frameon=False)
ax = axes[1]
ax.plot(sigmas, cl_act, label="periphery", lw=2)
ax.plot(sigmas, cc_act, label="center", lw=2)
ax.plot(sigmas, creg_act, label="regular", lw=2)
ax.axvline(1.0, ls=":", c="k", lw=1)
ax.set_title("Act-based ($Y_i=c_i$)")
ax.set_xlabel("elasticity of substitution $\\sigma$")
ax.legend(frameon=False)
fig.suptitle(f"CES equilibrium contributions by position, $a={a0}$, "
             f"$\\beta={BETA}$, $e={int(E)}$, $n={N}$ "
             f"(dotted line: Cobb--Douglas, $\\sigma=1$)", y=1.02)
fig.tight_layout()
fig.savefig("figures/fig8_ces_sigma_contributions.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 9
sig_grid = np.linspace(0.25, 3.0, 60)
ceil_reg_imp = [regular_ceiling(BETA, s, True) for s in sig_grid]
ceil_reg_act = [regular_ceiling(BETA, s, False) for s in sig_grid]
ceil_per_imp = [periphery_ceiling(BETA, s, True) for s in sig_grid]
ceil_per_act = [periphery_ceiling(BETA, s, False) for s in sig_grid]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(sig_grid, ceil_reg_imp, lw=2, label="regular / center, impact-based")
ax.plot(sig_grid, ceil_reg_act, lw=2, ls="--",
        label="regular / center, act-based")
ax.plot(sig_grid, ceil_per_imp, lw=2, label="periphery, impact-based")
ax.plot(sig_grid, ceil_per_act, lw=2, ls="--", label="periphery, act-based")
ax.axvline(1.0, ls=":", c="k", lw=1)
ax.annotate("$\\sigma=1$: the two models coincide", xy=(1.0, 0.05),
            xytext=(1.35, 0.045), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("elasticity of substitution $\\sigma$")
ax.set_ylabel("interiority ceiling $a^\\star(\\sigma)$")
ax.set_title(f"Interior-region ceilings under CES, $\\beta={BETA}$")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig9_ces_sigma_ceilings.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- table
rows = [("sigma", "model", "c_center", "c_periphery", "gap",
         "c_regular", "ceiling_regular", "ceiling_periphery")]
for s in (0.5, 1.0, 2.0):
    for impact, name in ((True, "impact"), (False, "act")):
        cc, cl = star_equilibrium(a0, sigma=s, impact=impact)
        cr = regular_equilibrium(a0, sigma=s, impact=impact)
        rows.append((s, name, round(cc, 3), round(cl, 3), round(cl - cc, 3),
                     round(cr, 3), round(regular_ceiling(BETA, s, impact), 4),
                     round(periphery_ceiling(BETA, s, impact), 4)))
with open("data/ces_sigma_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)

# ------------------------------------------------------- consistency checks
# 1) sigma = 1 reproduces the Cobb-Douglas closed forms and ceilings.
assert abs(regular_equilibrium(a0, sigma=1.0, impact=True)
           - 3 * a0 * E / (3 - BETA - 2 * a0 * BETA)) < 1e-9
assert abs(regular_ceiling(BETA, 1.0, True) - (3 - BETA) / (3 + 2 * BETA)) < 1e-12
assert abs(periphery_ceiling(BETA, 1.0, True) - 1.0 / 6.0) < 1e-4
# 2) at sigma = 1 the impact-based and act-based models coincide.
for s_ in (1.0,):
    assert np.allclose(star_equilibrium(a0, sigma=s_, impact=True),
                       star_equilibrium(a0, sigma=s_, impact=False), atol=1e-9)
# 3) positional ordering periphery > center holds for all sigma on the grid.
for s_ in sigmas:
    cc, cl = star_equilibrium(a0, sigma=s_, impact=True)
    assert cl >= cc - 1e-9

print("All CES consistency checks passed.")
print("Figures written to figures/, table to data/ces_sigma_table.csv.")
