# Lake Modal Analysis Package

A Python package for modal analysis of internal waves in stratified lakes,
converted from the MATLAB code by Carlos Rozas Rojas (2010).

**Reference:**  
Rozas, C., de la Fuente, A., Ulloa, H. et al. *Quantifying the effect of wind
on internal wave resonance in Lake Villarrica, Chile.* Environ Fluid Mech 14,
849–871 (2014). https://doi.org/10.1007/s10652-013-9329-9

## Overview

This package performs eigenvalue analysis on a two-layer stratified system to
compute normal oscillation modes (basin-scale internal waves / seiches) in
lakes, including:

- Variable bathymetry
- Earth's rotation (Coriolis effect, constant or beta-plane)
- Two-layer density stratification
- Staggered grid (Arakawa C-grid) finite difference discretization

## Installation

```bash
cd lake_modal_analysis
pip install numpy scipy matplotlib --break-system-packages
```

## Quick Start

```python
import numpy as np
from lake_modal_analysis import ModalAnalysis

config = {
    'g': 9.81,              # Gravity (m/s²)
    'h1': 30,               # Upper layer thickness (m)
    'rho1': 998.4,          # Upper layer density (kg/m³)
    'rho2': 999.9,          # Lower layer density (kg/m³)
    'dx': 400,              # Grid spacing x (m)
    'dy': 400,              # Grid spacing y (m)
    'x_range': (500, 565),  # Domain extent x (km)
    'y_range': (117, 155),  # Domain extent y (km)
    'lat0': 46.46,          # Central latitude (degrees)
    'y0': 142.5,            # Central y-coordinate (km)
    'n_modes': 9,           # Number of modes
    'period_target': 16,    # Target period (hours)
}

analysis = ModalAnalysis(config)
results = analysis.run_analysis('shoreline.txt', 'bathymetry.xyz')

# Plot modes
fig = analysis.plot_modes(n_modes=9, save_path='modes.png')

# Show inline in Jupyter
from IPython.display import display
display(fig)
```

## Package Structure

```
lake_modal_analysis/
├── __init__.py           # Package initialization
├── modal_analysis.py     # Main orchestration class
├── mesh.py               # Mesh generation
├── bathymetry.py         # Bathymetry processing
├── coriolis.py           # Coriolis parameter calculation
├── matrices.py           # System matrix assembly (K and M)
├── eigensolve.py         # Eigenvalue problem solver
├── velocities.py         # Velocity field processing
├── visualization.py      # Plotting and animation
└── examples/
    └── rectangle/        # Synthetic rectangular basin test case
```

## Input Data Format

### Shoreline file
Two columns, coordinates in **metres**:
```
543210.5  5123456.7
543215.3  5123460.2
...
```

### Bathymetry file
Three columns (x, y, depth) in **metres**. Depth is negative below the
water surface:
```
543210.5  5123456.7  -45.3
543215.3  5123460.2  -47.8
...
```

## Configuration Parameters

| Key | Description | Units |
|-----|-------------|-------|
| `g` | Gravitational acceleration | m/s² |
| `h1` | Upper layer thickness | m |
| `rho1` | Upper layer density | kg/m³ |
| `rho2` | Lower layer density | kg/m³ |
| `dx`, `dy` | Grid spacing | m |
| `x_range`, `y_range` | Domain extent (min, max) | km |
| `lat0` | Central latitude | degrees |
| `y0` | Y-coordinate of central latitude | km |
| `beta0` | Beta parameter (`0` = constant f, `None` = auto) | 1/(m·s) |
| `n_modes` | Number of modes to compute | — |
| `period_target` | Target period for eigenvalue search | hours |

## Accessing Results

```python
results = analysis.run_analysis(...)

# Mode periods (hours), sorted longest first
periods = results['eigenvalues']['wshorasID'][:, 0]

# Interface displacement mode shapes — shape (ny, nx, n_modes)
eta2 = results['eigenvalues']['eta2tilde']

# Velocity fields (subsampled for plotting)
u1s = results['velocities']['u1s']
v1s = results['velocities']['v1s']
```

## Plotting in Jupyter

```python
import matplotlib.pyplot as plt
from IPython.display import display

# All modes on one figure
fig = analysis.plot_modes(n_modes=9, save_path='modes.png')
display(fig)
plt.close(fig)

# Single mode
fig = analysis.plot_single_mode(mode_idx=0)
display(fig)
plt.close(fig)
```

> **Note:** Do not use `plt.show(fig)` — `plt.show()` takes no arguments.
> Use `display(fig)` for inline rendering in Jupyter.

## Saving and Loading Results

```python
# Save
analysis.save_results('results.npz')

# Load
data = ModalAnalysis.load_results('results.npz')
```

## Scientific Background

The package solves the linearised shallow-water equations for a two-layer
system discretised on a staggered Arakawa C-grid.  After applying no-normal-
flow boundary conditions, the system reduces to a generalised eigenvalue
problem:

```
K · ξ = ω · M · ξ
```

where K is the (complex, non-Hermitian) stiffness matrix, M is the diagonal
mass matrix, ω are the eigenvalues (angular frequencies), and ξ are the
eigenvectors (mode shapes).  The solver uses shift-invert Arnoldi iteration
(`scipy.sparse.linalg.eigs`).

## Known Issues / Troubleshooting

**Eigenvalue solver did not converge**  
Adjust `period_target` to be closer to the expected dominant mode period.

**Modes appear distorted**  
Verify that `rho2 > rho1` and that `h1` is less than the total depth
everywhere in the domain.

**Plots not showing in Jupyter**  
Use `display(fig)` instead of `plt.show()`. Do not call
`matplotlib.use('Agg')` before importing the package.

**Wrong number of modes returned**  
Increase `n_modes` or move `period_target` closer to the frequency band of
interest.

## References

Original MATLAB code: Copyright © 2010 Carlos Rozas Rojas (crozas@ing.uchile.cl)  
License: GNU General Public License v3.0

## Version History

- **v1.1.0** (March 2026): Four bug fixes — Coriolis offsets, eigensolver,
  complex arrays, matplotlib backend. See `CONVERSION_SUMMARY.md`.
- **v1.0.0** (December 2025): Initial Python conversion.
