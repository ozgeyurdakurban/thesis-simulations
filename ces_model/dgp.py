"""Data-generating process for the Monte Carlo, power and estimation studies.
Groups of three: star (centre = node 0) and K3.  Types B_i ~ Beta(mean, sd),
i.i.d.; equilibrium of the CES game; observed choices = equilibrium + N(0, tau^2)
tremble, clipped to [0, e].  Shares are computed from observed choices:
    share_i = theta_i c_i / W_i,  W_i = e + (beta/d_i) * (sum of neighbours' c).
"""
import numpy as np
from ces_net import *

E, BETA, N = 15.0, 1.5, 3
GS, GK = adjacency("star", N), adjacency("complete", N)

def draw_types(rng, R, mean, sd):
    a, b = beta_params(mean, sd)
    return np.clip(rng.beta(a, b, size=(R, N)), 1e-4, 1 - 1e-4)

def group_data(rng, G, R, sigma, lam, mean, sd, tau, beta=BETA, e=E):
    B = draw_types(rng, R, mean, sd)
    c = equilibrium(G, B, beta, sigma, lam, e)
    obs = np.clip(c + rng.normal(0, tau, c.shape), 0, e) if tau > 0 else c
    d, th, Lam = structure(G, beta)
    W = e + (beta / d) * (obs @ G.T) + 0 * obs
    W = e + (beta / d) * (obs @ G.T)
    shr = th * obs / W
    return dict(B=B, c=c, obs=obs, share=shr, W=W)

def pool(rng, sigma, lam, mean=.10, sd=.04, tau=1.5, R=20000):
    return group_data(rng, GS, R, sigma, lam, mean, sd, tau), group_data(rng, GK, R, sigma, lam, mean, sd, tau)
