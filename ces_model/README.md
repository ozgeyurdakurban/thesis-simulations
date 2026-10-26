# thesis-simulations

Numerical analysis for the **Simulations** section of Chapter 2 of the doctoral thesis
*Altruism in Networked Public Good Games*: the general CES model with impact weight
(`sigma`, `lambda`) on networks. The computations are deterministic.

Parameters of the design: `e = 15`, `n = 3`, `beta = 1.5`.

## Method

Best response `c_i = min{e, alpha_i + gamma_i S_i}` with giving share
`odds(pi_i) = (B_i/(1-B_i))^sigma p_i^(1-sigma)`, `p_i = theta_i / Lambda~_i`.
The equilibrium is the limit of monotone best-response iteration started from `c = 0`
(least equilibrium) and from `c = e` (greatest equilibrium); the game is supermodular
(Milgrom and Roberts, 1990), and uniqueness is verified wherever the two limits coincide.

## Files

| File | Content | Output |
|---|---|---|
| `ces_net.py` | library: networks, shares, prices, equilibrium solver, closed forms, `lam_bar`, `sigma_star` | -- |
| `sim01_validation.py` | best response vs direct utility maximization; K_n and S_n closed forms; least = greatest equilibrium | `data/validation.csv` |
| `sim02_positions.py` | contributions by position over `sigma` and `lambda` | `figures/fig10_*.png`, `data/positions_*.csv` |
| `sim03_regimes.py` | regime map in `(B, sigma)`; feasible set of the star design in `(B, beta)` | `figures/fig11_*.png`, `fig12_*.png`, `data/ceilings_beta15.csv` |
| `sim04_orderings.py` | interiority ceilings (table); leaf-centre ordering in `(lambda, sigma)` with analytic boundary | `figures/fig13_*.png`, `data/ceilings_table.csv` |

## Run

```bash
pip install numpy scipy matplotlib
python sim01_validation.py && python sim02_positions.py && python sim03_regimes.py && python sim04_orderings.py
```

Run time: about two minutes in total.
