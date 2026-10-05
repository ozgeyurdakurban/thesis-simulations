# thesis-simulations

Simulation study of the **interior-solution ranges** and **feasible parameter
regions** of the impact-based Cobb–Douglas networked public-goods model, for the
doctoral thesis *Altruism in Networked Public Good Games*. Reproduces every figure
and table in the **Simulations** chapter.

Equilibria are computed with a **clipped best-response solver** (project to `[0,e]`),
which returns the exact constrained equilibrium in every regime — including mixed
regimes where some positions are at the corner `c=e` while others are interior. On
the fully interior region the solver reproduces the closed forms of the
*Equilibrium Analysis* chapter (validated at startup).

Baseline parameters: `e = 15`, `n = 3`, `beta = 1.5`.

## Files

| File | What it does | Outputs |
|------|--------------|---------|
| `simulations_interior_solutions.py` | Phase diagrams in `(B, beta)`; comparative statics `c*(B)` and `c*(beta)`; interiority ceilings; heterogeneous-`B` contribution distributions. | `figures/fig1_phase_diagrams.png`, `fig2_cstar_vs_B.png`, `fig3_cstar_vs_beta.png`, `fig4_heterogeneous.png` |
| `simulations_grid_mc.py` | **Deterministic grid**: homogeneous `(B,beta)` ceilings + exhaustive asymmetric triples `(B1,B2,B3)`. **Monte Carlo**: Beta-distributed `B_i`, replications with MC standard errors, convergence diagnostic, corner probabilities, social efficiency. | `data/grid_ceilings.csv`, `data/grid_asymmetric_summary.csv`, `data/montecarlo_summary.csv`, `figures/fig5_montecarlo_convergence.png` |
| `simulations_feasibility.py` | **Feasible parameter region**: where the model's conditions (dilemma `1<beta<d_i`, social productivity `sigma_i>1`, interior equilibrium `0<c*<e`) hold jointly, per position and for the 3-person star design. | `figures/fig6_feasible_by_position.png`, `figures/fig7_feasible_experiment.png` |
| `simulations_ces_networks.py` | **CES robustness**: equilibrium contributions by position and interiority ceilings as functions of the elasticity of substitution `sigma = 1/(1-rho)`, impact-based vs act-based moral component; checks that `sigma = 1` reproduces the Cobb–Douglas closed forms. | `figures/fig8_ces_sigma_contributions.png`, `figures/fig9_ces_sigma_ceilings.png`, `data/ces_sigma_table.csv` |

`simulations_feasibility.py` imports the solver from `simulations_interior_solutions.py`,
so keep the two in the same directory.

## Requirements

```
python >= 3.10
numpy
matplotlib
```
```
pip install numpy matplotlib
```

## Run

```bash
python simulations_interior_solutions.py   # fig1–fig4 + validation
python simulations_grid_mc.py              # grid + Monte Carlo -> data/ and fig5
python simulations_feasibility.py          # fig6, fig7
python simulations_ces_networks.py         # fig8, fig9 + data/ces_sigma_table.csv
```
Figures are written to `figures/`, tabular output to `data/` (both created
automatically).

## Key results reproduced

- Interiority ceilings at `beta = 1.5` (homogeneous `B`): for the symmetric positions
  (regular member and star center, both `d = 3`) `B*(beta) = (d - beta)/(d + beta(d - 1)) = 0.25`;
  the star periphery (`d = 2`) is the binding position, with ceiling `1/6 ≈ 0.167`
  determined jointly with the center's best response (the own-degree formula would
  give the incorrect `1/7`).
- Higher `beta` narrows the interior region.
- Positional ordering periphery > center > regular under heterogeneity (Monte Carlo).
- Joint-feasible homogeneous-`B` interval for the star experiment at `beta=1.5`: `(0, 0.167)`.

## Layout

```
.
├── simulations_interior_solutions.py
├── simulations_grid_mc.py
├── simulations_feasibility.py
├── simulations_ces_networks.py
├── figures/      # generated PNGs
└── data/         # generated CSVs
```
