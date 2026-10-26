"""Finite-sample performance of the rebate/match design (Proposition on price and
impact variation).  Monte Carlo for one donor observed in K environments e_k=(theta_k, nu_k).

Model: l_k = a + (1 - sigma) * log p_k,  p_k = theta_k / (1 + lam * nu_k),  a = sigma * log(B/(1-B)).
Observation: l_obs = l + e, e ~ N(0, s^2 / m)  (m decisions per cell, iid logit-normal noise).
Estimator: for each lam on a grid (lam >= 0) regress l_obs on log p_k(lam); slope = 1 - sigma_hat;
choose lam minimizing the sum of squared residuals (profile least squares; no scipy needed).
Designs:  D3 = baseline, rebate, match (exactly identified)
          D5 = D3 + second rebate + second match   (2 over-identifying restrictions)
          D5e = D3 + equal-price pair (adds the cell that isolates the act motive)
Outputs:  data/rebate_match_power.csv, printed summary.
Also: misspecification check -- a constant-gross-contribution rule (Davis, Millner & Reilly 2005)
fed into the same estimator."""
import numpy as np, csv, os
rng = np.random.default_rng(20261026)

BASE = (1.0, 1.0)
DESIGNS = {
    "D3":  [(1.0, 1.0), (0.5, 1.0), (1.0, 2.0)],
    "D5":  [(1.0, 1.0), (0.5, 1.0), (0.75, 1.0), (1.0, 2.0), (1.0, 4.0)],
    "D5e": [(1.0, 1.0), (0.5, 1.0), (1.0, 2.0), (0.75, 1.0), (4/3, 1.0 + 1e-12) if False else (1.0, 4/3)],
}
# D5e: baseline, rebate (theta=.5), match (nu=2) [equal price: theta/nu = .5 both], plus a milder pair (theta=.75, nu=4/3)
DESIGNS["D5e"] = [(1.0, 1.0), (0.5, 1.0), (1.0, 2.0), (0.75, 1.0), (1.0, 4/3)]

LGRID = np.concatenate([[0.0], np.logspace(-2, 2, 160)])

def logit(x): return np.log(x / (1 - x))

def l_true(B, sigma, lam, cells):
    th = np.array([c[0] for c in cells]); nu = np.array([c[1] for c in cells])
    p = th / (1 + lam * nu)
    return sigma * logit(B) + (1 - sigma) * np.log(p)

def estimate(L, cells):
    """L: (S,K) observed log-odds.  Returns sigma_hat, lam_hat, B_hat, ssr arrays."""
    th = np.array([c[0] for c in cells]); nu = np.array([c[1] for c in cells])
    S, K = L.shape
    best = np.full(S, np.inf); lam_h = np.zeros(S); sl_h = np.zeros(S); ic_h = np.zeros(S)
    Lc = L - L.mean(1, keepdims=True)
    for lam in LGRID:
        x = np.log(th / (1 + lam * nu)); xc = x - x.mean()
        den = (xc ** 2).sum()
        if den < 1e-12: continue
        slope = (Lc * xc).sum(1) / den
        res = Lc - slope[:, None] * xc
        ssr = (res ** 2).sum(1)
        m = ssr < best
        best[m] = ssr[m]; lam_h[m] = lam; sl_h[m] = slope[m]
        ic_h[m] = (L.mean(1) - slope * x.mean())[m]
    sig = 1 - sl_h
    return sig, lam_h, best

def ssr_lam0(L, cells):
    th = np.array([c[0] for c in cells]); nu = np.array([c[1] for c in cells])
    x = np.log(th); xc = x - x.mean(); den = (xc ** 2).sum()
    Lc = L - L.mean(1, keepdims=True)
    if den < 1e-12: return np.full(L.shape[0], np.nan)
    slope = (Lc * xc).sum(1) / den
    return ((Lc - slope[:, None] * xc) ** 2).sum(1)

S = 4000
B = 0.10
rows = []
print("design  sigma lam  s    m | med sig_hat  med lam_hat  RMSE(log lam)  P(|sig_hat-sig|<.25)  power(lam=0)")
for dname, cells in ([] if os.environ.get("POOLED_ONLY") else DESIGNS.items()):
    for sigma in (0.5, 0.75, 1.5, 2.0):
        for lam in (0.5, 1.0, 2.0):
            for s in (0.15, 0.30, 0.60, 1.20, 2.50):  # 2.5 = log-odds noise estimated from Diederich et al. (sim10)
                for m in (1, 4):
                    sd = s / np.sqrt(m)
                    lt = l_true(B, sigma, lam, cells)
                    L = lt + rng.normal(0, sd, (S, len(cells)))
                    sh, lh, _ = estimate(L, cells)
                    # critical value under the null lam = 0 (same sigma, s, m)
                    L0 = l_true(B, sigma, 0.0, cells) + rng.normal(0, sd, (S, len(cells)))
                    sh0, lh0, _ = estimate(L0, cells)
                    crit = np.quantile(lh0, 0.95)
                    power = float(np.mean(lh > crit))
                    ok = np.isfinite(lh)
                    # RMSE of log(1+lam) is more stable than of lam near the grid bound
                    rm = float(np.sqrt(np.mean((np.log1p(lh) - np.log1p(lam)) ** 2)))
                    p_sig = float(np.mean(np.abs(sh - sigma) < 0.25))
                    rows.append(dict(design=dname, sigma=sigma, lam=lam, s=s, m=m,
                                     med_sigma_hat=float(np.median(sh)), med_lam_hat=float(np.median(lh)),
                                     rmse_log1p_lam=rm, p_sigma_within_025=p_sig, power_lam_gt0=power,
                                     crit_lam=float(crit)))
os.makedirs("data", exist_ok=True)
if rows:
  with open("data/rebate_match_power.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- summary table for the thesis: lam = 1, s = 0.30, m = 1 and 4 -----------------------------------------
print("\nSummary (lambda=1, s=0.30):")
print("design sigma m | med sigma_hat | med lambda_hat | P(|dsigma|<.25) | power")
for r in rows:
    if r["lam"] == 1.0 and r["s"] == 0.30:
        print(f'{r["design"]:4s} {r["sigma"]:5.2f} {r["m"]} | {r["med_sigma_hat"]:6.2f} | {r["med_lam_hat"]:6.2f} | {r["p_sigma_within_025"]:5.2f} | {r["power_lam_gt0"]:5.2f}')

# ---- misspecification: constant gross contribution rule --------------------------------------------------
print("\nConstant-gross-contribution donors (gross share g/W = 0.5 in every cell, so pi = 0.5*theta), noise s=0.30:")
for dname, cells in DESIGNS.items():
    th = np.array([c[0] for c in cells])
    pi = 0.5 * th
    pi = np.clip(pi, 1e-6, 1 - 1e-6)
    lt = logit(pi)
    L = lt + rng.normal(0, 0.30, (S, len(cells)))
    sh, lh, ssr = estimate(L, cells)
    # fit diagnostic: noiseless residual of the best CES fit
    sh_n, lh_n, ssr_n = estimate(lt[None, :], cells)
    print(f"{dname}: noiseless fit sigma_hat={sh_n[0]:.3f} lam_hat={lh_n[0]:.2f} SSR={ssr_n[0]:.4f};  "
          f"noisy median sigma_hat={np.median(sh):.2f} lam_hat={np.median(lh):.2f}")

# ---- population-level recovery: median of individual lambda_hat across N donors ---------------------------
print("\nPopulation level (identical donors, B=.10; s and N as in the csv), N donors; median of individual lam_hat")
pool = []
R = 2000
for dname, s0, Nd in [(d_, s_, n_) for d_ in ("D3", "D5", "D5e") for s_, n_ in [((0.30, 100), (2.50, 113), (2.50, 1000))[int(i)] for i in os.environ.get("POOL_SETTINGS", "012")]]:
    cells = DESIGNS[dname]
    for sigma in (0.5, 0.75, 1.5, 2.0):
        for lam in (0.5, 1.0, 2.0):
            for m in (1, 4):
                sd = s0 / np.sqrt(m)
                L = l_true(B, sigma, lam, cells) + rng.normal(0, sd, (20000, len(cells)))
                sh, lh, _ = estimate(L, cells)
                L0 = l_true(B, sigma, 0.0, cells) + rng.normal(0, sd, (20000, len(cells)))
                sh0, lh0, _ = estimate(L0, cells)
                idx = rng.integers(0, 20000, (R, Nd))
                med = np.median(lh[idx], 1); med0 = np.median(lh0[idx], 1)
                crit = np.quantile(med0, 0.95)
                q = np.quantile(lh, [0.10, 0.25, 0.50, 0.75, 0.90])
                pool.append(dict(design=dname, s=s0, sigma=sigma, lam=lam, m=m, N=Nd,
                                 q10=q[0], q25=q[1], q50=q[2], q75=q[3], q90=q[4],
                                 med_of_med=float(np.median(med)), sd_med=float(med.std()),
                                 power_pooled=float(np.mean(med > crit)),
                                 med_sigma=float(np.median(np.median(sh[idx], 1)))))
if os.environ.get("POOLED_ONLY") and os.path.exists("data/rebate_match_pooled.csv"):
    old = [r for r in csv.DictReader(open("data/rebate_match_pooled.csv"))]
    keep = [r for r in old if r.get("s") not in (None, "") and not any(float(r["s"]) == q["s"] and int(r["N"]) == q["N"] for q in pool)]
    for r in keep: r.update({k: (float(v) if k not in ("design",) else v) for k, v in r.items()})
    pool = keep + pool
with open("data/rebate_match_pooled.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(pool[0].keys())); w.writeheader(); w.writerows(pool)
print("design sigma lam m | lam_hat q25 q50 q75 | median over N=100: mean (sd) | pooled power")
for r in pool:
    if r["lam"] == 1.0 and r["design"] in ("D3", "D5"):
        print(f'{r["design"]:3s} {r["sigma"]:4.2f} {r["lam"]:3.1f} {r["m"]} | {r["q25"]:5.2f} {r["q50"]:5.2f} {r["q75"]:6.2f} | {r["med_of_med"]:5.2f} ({r["sd_med"]:4.2f}) | {r["power_pooled"]:4.2f}')
