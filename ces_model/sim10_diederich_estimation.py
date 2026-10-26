"""sim10: population-level estimation of (sigma, lambda, B) on the Diederich-Eckel-Epperson-Goeschl-Grossman
(Experimental Economics 25, 734-758, 2022; data: heiDATA, doi:10.11588/DATA/GANTNY, v1.1, restricted access) within-subject panel (113 MTurk subjects x 7 cells).
Model (single-recipient case of Prop. 2.4.33):  l = logit(pi) = a + (1-sigma)[log theta - log(1+lam*nu)],  a = sigma*logit(B).
theta = net price per donated package, nu = packages received per donated package (impact), W = $2.
Observed integer gift g is interval-censored: g=j  <=>  pi in (theta(j-.5)/W, theta(j+.5)/W); upper cap Gmax per cell.
l* = mu + eps, eps ~ N(0, s^2)  (ordered probit on a known threshold grid, cf. the authors' own oprobcens).
Variants: mix = never-giver mass phi; frame = format intercepts (rebate, match) absorbing framing.
Subject-clustered bootstrap (90% percentile intervals). Output: data/diederich_estimates.csv.
Usage: python3 sim10_diederich_estimation.py [nboot]
"""
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "data_external", "diederich_etal_dataverse", "data", "WS_data_noIP_noMTID_panel.dta")
d = pd.read_stata(P)
W = 2.0
cell = {  # theta, nu, Gmax, format
 "Control": (0.50, 1.0, 4, "d"), "DirectHi": (1/3, 1.0, 6, "d"), "DirectLo": (0.25, 1.0, 8, "d"),
 "RebateHi": (1/3, 1.0, 4, "r"), "RebateLo": (0.25, 1.0, 4, "r"),
 "MatchHi": (0.50, 1.5, 4, "m"), "MatchLo": (0.50, 2.0, 4, "m")}
tn = d.TreatmentName.astype(str)
th = tn.map(lambda c: cell[c][0]).values.astype(float)
nu = tn.map(lambda c: cell[c][1]).values.astype(float)
gm = tn.map(lambda c: cell[c][2]).values.astype(int)
fr = tn.map(lambda c: cell[c][3]).values
g = d.PacksDonated.values.astype(int)
ids = pd.factorize(d.ID)[0]
isR, isM = (fr == "r").astype(float), (fr == "m").astype(float)
logit = lambda x: np.log(x / (1 - x))

def loglik_i(par, g, th, nu, gm, isR, isM, mix, frame):
    a, sig, loglam, logs = par[:4]; k = 4
    dR = dM = 0.0
    if frame: dR, dM = par[k], par[k+1]; k += 2
    phi = 1/(1+np.exp(-par[k])) if mix else 0.0
    lam = np.exp(np.clip(loglam, -7, 7)); s = np.exp(np.clip(logs, -3, 3))
    mu = a + (1-sig)*(np.log(th) - np.log(1+lam*nu)) + dR*isR + dM*isM
    lo = np.where(g == 0, -np.inf, logit(th*(g-.5)/W))
    hi = np.where(g >= gm, np.inf, logit(th*(g+.5)/W))
    p = np.clip(norm.cdf((hi-mu)/s) - norm.cdf((lo-mu)/s), 1e-300, 1)
    return np.log(phi*(g == 0) + (1-phi)*p)

def fit(idx, mix, frame):
    A = (g[idx], th[idx], nu[idx], gm[idx], isR[idx], isM[idx], mix, frame)
    f = lambda par: -loglik_i(par, *A).sum()
    best = None
    for st in ([-1.5, 0.5, 0.0, 0.5], [-1.0, 0.0, -1.0, 0.0], [-2.0, 0.9, 1.0, 1.0]):
        st = list(st) + ([0.0, 0.0] if frame else []) + ([-0.5] if mix else [])
        r = minimize(f, st, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
        r = minimize(f, r.x, method="BFGS")
        if best is None or r.fun < best.fun: best = r
    return best

def summ(par, mix, frame):
    a, sig, loglam, logs = par[:4]
    out = dict(sigma=sig, lam=np.exp(loglam), s=np.exp(logs), a=a,
               B=(1/(1+np.exp(-a/sig)) if abs(sig) > 1e-6 else np.nan))
    k = 4
    if frame: out["dR"], out["dM"] = par[k], par[k+1]; k += 2
    if mix: out["phi"] = 1/(1+np.exp(-par[k]))
    return out

if __name__ == "__main__":
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rng = np.random.default_rng(7)
    uid = np.unique(ids); allidx = np.arange(len(g))
    rows = []
    for mix in (False, True):
        for frame in (False, True):
            r = fit(allidx, mix, frame); est = summ(r.x, mix, frame)
            bs = []
            for b in range(nboot):
                draw = rng.choice(uid, len(uid))
                idx = np.concatenate([np.where(ids == u)[0] for u in draw])
                try: bs.append(summ(fit(idx, mix, frame).x, mix, frame))
                except Exception: pass
            bs = pd.DataFrame(bs)
            row = dict(model=f"mix={int(mix)},frame={int(frame)}", loglik=-r.fun, nparam=len(r.x), n=len(g))
            for k, v in est.items():
                row[k] = v
                row[k+"_lo"] = bs[k].quantile(.05) if k in bs else np.nan
                row[k+"_hi"] = bs[k].quantile(.95) if k in bs else np.nan
            rows.append(row)
            print(pd.Series(row).dropna().astype(str).to_string(), "\n", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "data", "diederich_estimates.csv"), index=False)
