"""
ces_net.py -- core library for the Simulations chapter.

Model (Chapter 2): local-average public good on a network, CES preferences over
the material component X_i and the moral component Y_i = Lambda~_i c_i.

    X_i      = e - c_i + (beta/d_i) * sum_{j in N[i]} c_j = W_i - theta_i c_i
    Lambda_i = 1 + sum_{j in G_i} beta/d_j,   Lambda~_i = 1 + lam_i (Lambda_i - 1)
    p_i      = theta_i / Lambda~_i,           theta_i = 1 - beta/d_i
    odds(pi_i) = (B_i/(1-B_i))**sigma * p_i**(1-sigma)            (giving share)
    best response: c_i = min{ e, alpha_i + gamma_i * S_i },
                   alpha_i = pi_i e/theta_i, gamma_i = pi_i beta/(d_i - beta)

Equilibria are computed by monotone best-response iteration started from both
c = 0 and c = e.  The game is supermodular, so the two limits are the least and
the greatest Nash equilibrium; they coincide iff the equilibrium is unique.
All routines are vectorised over a leading 'replication' axis.
"""
import numpy as np

E_DEFAULT, BETA_DEFAULT = 15.0, 1.5


# ----------------------------------------------------------------- networks
def adjacency(kind, n):
    """Adjacency matrix (no loops).  'star': node 0 is the centre."""
    G = np.zeros((n, n))
    if kind == "complete":
        G = np.ones((n, n)) - np.eye(n)
    elif kind == "cycle":
        for i in range(n):
            G[i, (i + 1) % n] = G[i, (i - 1) % n] = 1
    elif kind == "star":
        G[0, 1:] = 1
        G[1:, 0] = 1
    else:
        raise ValueError(kind)
    return G


def structure(G, beta):
    """Return d (closed-neighbourhood size), theta, Lambda for adjacency G."""
    d = G.sum(1) + 1.0
    theta = 1.0 - beta / d
    Lam = 1.0 + G @ (beta / d)
    return d, theta, Lam


# ------------------------------------------------------------- preferences
def share(B, p, sigma):
    """Giving share pi = odds/(1+odds), odds = (B/(1-B))^sigma p^(1-sigma)."""
    B = np.asarray(B, float)
    lodds = sigma * (np.log(B) - np.log1p(-B)) + (1.0 - sigma) * np.log(p)
    return 1.0 / (1.0 + np.exp(-lodds))


def price(theta, Lam, lam):
    """p_i = theta_i / Lambda~_i, Lambda~_i = 1 + lam (Lambda_i - 1)."""
    return theta / (1.0 + lam * (Lam - 1.0))


# --------------------------------------------------------------- equilibrium
def equilibrium(G, B, beta=BETA_DEFAULT, sigma=1.0, lam=1.0, e=E_DEFAULT,
                tol=1e-12, maxit=200000, return_both=False):
    """Nash equilibrium contributions.  B has shape (..., n); sigma and lam may
    be scalars or broadcastable to B.  Returns c (..., n)."""
    B = np.asarray(B, float)
    d, theta, Lam = structure(G, beta)
    p = price(theta, Lam, lam)
    pi = share(B, p, sigma)
    alpha = pi * e / theta
    gamma = pi * beta / (d - beta)

    def iterate(c):
        for _ in range(maxit):
            new = np.minimum(e, alpha + gamma * (c @ G.T))
            if np.max(np.abs(new - c)) < tol:
                return new
            c = new
        return c

    lo = iterate(np.zeros_like(alpha))
    hi = iterate(np.full_like(alpha, e))
    if return_both:
        return lo, hi
    return 0.5 * (lo + hi)


def utility(c, S, d, theta, Lam, B, beta, sigma, lam, e):
    """CES utility of a single agent facing neighbours' total S."""
    X = e + beta / d * S - theta * c
    Y = (1.0 + lam * (Lam - 1.0)) * c
    if abs(sigma - 1.0) < 1e-12:
        return (1 - B) * np.log(X) + B * np.log(np.maximum(Y, 1e-300))
    rho = 1.0 - 1.0 / sigma
    Z = (1 - B) * X ** rho + B * np.maximum(Y, 1e-300) ** rho
    return Z / rho


# ------------------------------------------------------ closed-form (checks)
def cK_closed(n, B, beta, sigma, lam, e):
    """Symmetric K_n equilibrium (valid when interior)."""
    theta = 1 - beta / n
    Lt = 1 + lam * (n - 1) * beta / n
    pi = share(B, theta / Lt, sigma)
    return pi * n * e / (n - beta - pi * beta * (n - 1))


def star_closed(n, B, beta, sigma, lam, e):
    """Star centre and leaf contributions, common B (valid when interior)."""
    thC, thL = 1 - beta / n, 1 - beta / 2
    LC, LL = 1 + lam * (n - 1) * beta / 2, 1 + lam * beta / n
    piC, piL = share(B, thC / LC, sigma), share(B, thL / LL, sigma)
    S = (n - 1) * piL
    D = (n - beta) * (2 - beta) - piC * beta ** 2 * S
    cC = piC * e * (n * (2 - beta) + 2 * beta * S) / D
    cL = piL * e * (2 * (n - beta) + n * piC * beta) / D
    return cC, cL


def lam_bar(n, beta):
    """lambda at which p_C = p_L on the star (Prop. positional ordering)."""
    return 2 * n / ((n * n + n) * (2 - beta) - 2 * beta)


def sigma_star(n, beta, lam):
    """Elasticity above which the leaf-centre ordering reverses (lam>lam_bar)."""
    thC, thL = 1 - beta / n, 1 - beta / 2
    LC, LL = 1 + lam * (n - 1) * beta / 2, 1 + lam * beta / n
    pC, pL = thC / LC, thL / LL
    if pL <= pC:
        return np.inf
    return 1.0 + np.log(thC / thL) / np.log(pL / pC)


def beta_params(mean, sd):
    k = mean * (1 - mean) / sd ** 2 - 1.0
    if k <= 0:
        raise ValueError("sd too large for Beta with this mean")
    return mean * k, (1 - mean) * k
