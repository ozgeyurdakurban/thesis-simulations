"""sim11: the same interval-censored likelihood as sim10, on the between-subjects (BS) sample of
Diederich, Eckel, Epperson, Goeschl and Grossman (2022), n = 558, seven cells, one choice per participant.
Data: heiDATA doi:10.11588/DATA/GANTNY v1.1 (restricted access), file BS_data_noIP_noMTID.dta.
Output: data/diederich_bs_estimates.csv, data/diederich_bs_sigma_by_format.csv, data/diederich_bs_profile_lambda.csv"""
import numpy as np, pandas as pd, os, sys
import sim10_diederich_estimation as S
from scipy.optimize import minimize
HERE = S.HERE
b = pd.read_stata(os.path.join(HERE, "data_external", "diederich_etal_dataverse", "data", "BS_data_noIP_noMTID.dta"))
b = b[b.PacksDonated.notna()].copy()
name = {("Control", "CONTROL"): "Control", ("Direct High Price", "DIRECT"): "DirectHi", ("Direct Low Price", "DIRECT"): "DirectLo",
        ("Rebate High Price", "REBATE"): "RebateHi", ("Rebate Low Price", "REBATE"): "RebateLo",
        ("Match High Price", "MATCH"): "MatchHi", ("Match Low Price", "MATCH"): "MatchLo"}
b["cell"] = [name[(str(t), str(s))] for t, s in zip(b.Treatment, b.SubsType)]
c = S.cell
S.g = b.PacksDonated.values.astype(int)
S.th = b.cell.map(lambda x: c[x][0]).values.astype(float)
S.nu = b.cell.map(lambda x: c[x][1]).values.astype(float)
S.gm = b.cell.map(lambda x: c[x][2]).values.astype(int)
fr = b.cell.map(lambda x: c[x][3]).values
S.isR, S.isM = (fr == "r").astype(float), (fr == "m").astype(float)
S.ids = np.arange(len(b))
tn = b.cell.values
print("n =", len(b), b.cell.value_counts().to_dict())

def sub(cells): return np.where(np.isin(tn, cells))[0]

def fit_fixed(idx, sigma=None, lam=None, frame=False):
    A = (S.g[idx], S.th[idx], S.nu[idx], S.gm[idx], S.isR[idx], S.isM[idx], False, frame)
    def f(x):
        a, logs = x[0], x[1]; k = 2
        sg = sigma if sigma is not None else x[k]; k += (sigma is None)
        ll = np.log(lam) if lam is not None else np.clip(x[k], -4.6, 4.6); k += (lam is None)
        par = [a, sg, ll, logs] + (list(x[k:k+2]) if frame else [])
        return -S.loglik_i(np.array(par), *A).sum()
    best = None
    for st in (0.0, 1.0):
        x0 = np.array([-1.5, 0.8] + ([st] if sigma is None else []) + ([0.0] if lam is None else []) + ([0.0, 0.0] if frame else []))
        r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun: best = r
    return best

if __name__ == "__main__":
    rows = []
    sets = {"Direct (Control, DirectHi, DirectLo)": ["Control", "DirectHi", "DirectLo"], "Direct only (Hi vs Lo)": ["DirectHi", "DirectLo"],
            "Rebate only (Hi vs Lo)": ["RebateHi", "RebateLo"], "Match only (Hi vs Lo)": ["MatchHi", "MatchLo"]}
    for nm, cells in sets.items():
        idx = sub(cells); r = fit_fixed(idx, lam=1.0); r1 = fit_fixed(idx, sigma=1.0, lam=1.0)
        rows.append(dict(sample=nm, n=len(idx), sigma_hat=r.x[2], LR_sigma_eq_1=2*(r1.fun-r.fun), s=np.exp(r.x[1]))); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "data", "diederich_bs_sigma_by_format.csv"), index=False)
    allidx = np.arange(len(S.g)); prof = []
    for lam in [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]:
        for frame in (False, True):
            r = fit_fixed(allidx, lam=lam, frame=frame)
            prof.append(dict(lam=lam, frame=frame, loglik=-r.fun, sigma=r.x[2])); print(prof[-1], flush=True)
    pd.DataFrame(prof).to_csv(os.path.join(HERE, "data", "diederich_bs_profile_lambda.csv"), index=False)
    free = fit_fixed(allidx, frame=True); print("free fit (frame): loglik", -free.fun, "sigma", free.x[2], "loglam", free.x[3], "s", np.exp(free.x[1]))
