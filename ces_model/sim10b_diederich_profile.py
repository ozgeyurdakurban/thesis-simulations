"""sim10b: format-by-format sigma and profile likelihood for lambda on the Diederich et al. panel (see sim10)."""
import numpy as np, pandas as pd, os, sys
from scipy.optimize import minimize
import sim10_diederich_estimation as S
tn = S.d.TreatmentName.astype(str).values
HERE = S.HERE

def sub(cells): return np.where(np.isin(tn, cells))[0]

def fit_fixed(idx, sigma=None, lam=None, frame=False, loglam_bounds=(-4.6, 4.6)):
    A = (S.g[idx], S.th[idx], S.nu[idx], S.gm[idx], S.isR[idx], S.isM[idx], False, frame)
    def f(x):
        a, logs = x[0], x[1]; k = 2
        sg = sigma if sigma is not None else x[k]; k += (sigma is None)
        ll = np.log(lam) if lam is not None else np.clip(x[k], *loglam_bounds); k += (lam is None)
        par = [a, sg, ll, logs] + (list(x[k:k+2]) if frame else [])
        return -S.loglik_i(np.array(par), *A).sum()
    n = 2 + (sigma is None) + (lam is None) + (2 if frame else 0)
    best = None
    for st in (0.0, 1.0):
        x0 = np.array([-1.5, 0.8] + ([st] if sigma is None else []) + ([0.0] if lam is None else []) + ([0.0, 0.0] if frame else []))
        r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun: best = r
    return best

rows = []
sets = {"Direct (Control, DirectHi, DirectLo)": ["Control", "DirectHi", "DirectLo"],
        "Direct only (Hi vs Lo)": ["DirectHi", "DirectLo"],
        "Rebate only (Hi vs Lo)": ["RebateHi", "RebateLo"],
        "Match only (Hi vs Lo)": ["MatchHi", "MatchLo"]}
nb = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(11); uid = np.unique(S.ids)
for name, cells in sets.items():
    idx = sub(cells)
    r = fit_fixed(idx, lam=1.0)            # nu=1 in all but match; lambda irrelevant outside match
    r1 = fit_fixed(idx, sigma=1.0, lam=1.0)
    est = r.x[2]; lr = 2 * (r1.fun - r.fun)
    lo = hi = np.nan
    if nb:
        bs = []
        for b in range(nb):
            dr = rng.choice(uid, len(uid)); ii = np.concatenate([idx[np.isin(S.ids[idx], [u])] for u in dr])
            try: bs.append(fit_fixed(ii, lam=1.0).x[2])
            except Exception: pass
        lo, hi = np.percentile(bs, [5, 95])
    rows.append(dict(sample=name, sigma_hat=est, lo90=lo, hi90=hi, LR_sigma_eq_1=lr, s=np.exp(r.x[1])))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "data", "diederich_sigma_by_format.csv"), index=False)

# profile likelihood over lambda (all cells, format intercepts), sigma free
allidx = np.arange(len(S.g)); prof = []
for lam in [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]:
    for frame in (False, True):
        r = fit_fixed(allidx, lam=lam, frame=frame)
        prof.append(dict(lam=lam, frame=frame, loglik=-r.fun, sigma=r.x[2]))
        print(prof[-1], flush=True)
pd.DataFrame(prof).to_csv(os.path.join(HERE, "data", "diederich_profile_lambda.csv"), index=False)
