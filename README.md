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
| `simulations_grid_mc.py` | **Deterministic grid**: homogeneous `(B,beta)` ceilings + exhaustive asymmetric triples `(B1,B2,B3)`. **Monte Carlo**: Beta-distributed `B_i`, replications with MC standard errors, convergence diagnostic, corner probabilities, welfare. | `data/grid_ceilings.csv`, `data/grid_asymmetric_summary.csv`, `data/montecarlo_summary.csv`, `figures/fig5_montecarlo_convergence.png` |
| `simulations_feasibility.py` | **Feasible parameter region**: where the model's conditions (dilemma `1<beta<d_i`, social productivity `sigma_i>1`, interior equilibrium `0<c*<e`) hold jointly, per position and for the 3-person star design. | `figures/fig6_feasible_by_position.png`, `figures/fig7_feasible_experiment.png` |

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
```
Figures are written to `figures/`, tabular output to `data/` (both created
automatically).

## Key results reproduced

- Interiority ceiling `B*(beta) = (d_i - beta)/(d_i + beta(d_i - 1))`; regular member
  and star center share `d=3` and the same ceiling (`0.25` at `beta=1.5`), the star
  periphery (`d=2`) is the binding position (`0.167`).
- Higher `beta` narrows the interior region.
- Positional ordering periphery > center > regular under heterogeneity (Monte Carlo).
- Joint-feasible homogeneous-`B` interval for the star experiment at `beta=1.5`: `(0, 0.167)`.

## Layout

```
.
├── simulations_interior_solutions.py
├── simulations_grid_mc.py
├── simulations_feasibility.py
├── figures/      # generated PNGs
└── data/         # generated CSVs
```
