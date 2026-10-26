"""sim12: match-ratio variation in Karlan and List (2007), n = 50,083 prior donors, control and 1:1, 2:1, 3:1 matches.
Data: Karlan, D. and List, J. A. (2014), Harvard Dataverse, doi:10.7910/DVN/27853, V4, file 'AER merged.dta'.
Mapping: theta = 1 (one dollar costs one dollar), nu = 1 + ratio (dollars delivered per dollar given): control 1, matches 2, 3, 4.
With small giving shares the odds of the share are proportional to the share, so for mean gifts m_k (unconditional)
   log(m_k/m_0) = (sigma - 1) [log(1 + lam nu_k) - log(1 + lam nu_0)]  (homogeneous donors, W fixed).
Minimum-distance estimation over (sigma, lam) with an individual bootstrap (arms are randomized); profile over lam.
Output: data/karlan_list_arms.csv, data/karlan_list_profile.csv"""
import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_stata(os.path.join(HERE, "data_external", "karlan_list_2007_dataverse", "AER merged.dta"), convert_categoricals=False)
d["arm"] = np.where(d.control == 1, 0, d.ratio).astype(int)
nu = np.array([1.0, 2.0, 3.0, 4.0])
LG = np.concatenate([[0.0], np.logspace(-2, 2, 120)])

def stats(x):
    g = x.groupby("arm")
    return g.amount.mean().values, g.gave.mean().values, g.amount.apply(lambda a: a[a > 0].mean()).values, g.size().values

def fit(m):
    y = np.log(m[1:] / m[0]); best = (np.inf, None, None)
    for lam in LG:
        f = np.log(1 + lam * nu[1:]) - np.log(1 + lam * nu[0])
        den = (f ** 2).sum()
        if den < 1e-12: continue
        b = (f * y).sum() / den                      # b = sigma - 1
        ssr = ((y - b * f) ** 2).sum()
        if ssr < best[0]: best = (ssr, 1 + b, lam)
    return best

m, gv, cond, n = stats(d)
arms = pd.DataFrame(dict(arm=["control", "1:1", "2:1", "3:1"], n=n, response_rate=gv, mean_gift_unconditional=m, mean_gift_given_gift=cond))
rng = np.random.default_rng(5); B = 500
ratios, sig, lam_ = [], [], []
byarm = [d[d.arm == k] for k in range(4)]
for _ in range(B):
    mb = np.array([a.amount.values[rng.integers(0, len(a), len(a))].mean() for a in byarm])
    ratios.append(mb / mb[0]); s = fit(mb); sig.append(s[1]); lam_.append(s[2])
ratios = np.array(ratios)
for k, nm in enumerate(["control", "1:1", "2:1", "3:1"]):
    arms.loc[k, "ratio_to_control"] = m[k] / m[0]
    arms.loc[k, "ratio_lo90"], arms.loc[k, "ratio_hi90"] = np.percentile(ratios[:, k], [5, 95])
est = fit(m)
print(arms.round(4).to_string())
print("min-distance: sigma_hat = %.3f, lambda_hat = %.3f, SSR = %.5f" % (est[1], est[2], est[0]))
print("bootstrap sigma_hat 5/50/95: ", np.percentile(sig, [5, 50, 95]).round(3), " share sigma_hat<1:", np.mean(np.array(sig) < 1).round(3))
print("bootstrap lambda_hat 5/50/95:", np.percentile(lam_, [5, 50, 95]).round(3))
# sign prediction: own gross gift under a match falls iff sigma < 1 (lam > 0); test on the extensive and intensive margins
print("difference of means vs control (t):")
for k in (1, 2, 3):
    a, c = byarm[k].amount, byarm[0].amount
    t = (a.mean() - c.mean()) / np.sqrt(a.var() / len(a) + c.var() / len(c)); print(" ratio %d:1 mean gift diff %.3f (t=%.2f)" % (k, a.mean() - c.mean(), t))
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
arms.to_csv(os.path.join(HERE, "data", "karlan_list_arms.csv"), index=False)
pd.DataFrame(dict(sigma=sig, lam=lam_)).describe().to_csv(os.path.join(HERE, "data", "karlan_list_profile.csv"))
