# Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration Parameters](#configuration-parameters)
4. [Running the Analysis](#running-the-analysis)
5. [Plotting in Jupyter](#plotting-in-jupyter)
6. [Saving and Loading](#saving-and-loading)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
cd lake_modal_analysis
pip install -r requirements.txt --break-system-packages
# or for development install:
pip install -e . --break-system-packages
```

---

## Quick Start

```python
import numpy as np
from lake_modal_analysis import ModalAnalysis
from IPython.display import display

config = {
    'g': 9.81, 'h1': 30, 'dx': 400, 'dy': 400,
    'rho1': 998.4, 'rho2': 999.9,
    'x_range': (500, 565), 'y_range': (117, 155),
    'lat0': 46.46, 'y0': 142.5,
    'n_modes': 9, 'period_target': 16,
}

analysis = ModalAnalysis(config)
results = analysis.run_analysis('shoreline.txt', 'bathymetry.xyz')

fig = analysis.plot_modes(n_modes=9, save_path='modes.png')
display(fig)
```

---

## Configuration Parameters

### Physical parameters
```python
'g'    : 9.81     # Gravitational acceleration (m/s²)
'h1'   : 30       # Upper layer thickness (m)
'rho1' : 998.4    # Upper layer density (kg/m³)
'rho2' : 999.9    # Lower layer density (kg/m³)
```

### Grid parameters
```python
'dx'      : 400           # Grid spacing in x (m)
'dy'      : 400           # Grid spacing in y (m)
'x_range' : (500, 565)    # Domain x extent (km)
'y_range' : (117, 155)    # Domain y extent (km)
```

### Rotation parameters
```python
'lat0'  : 46.46   # Central latitude (degrees)
'y0'    : 142.5   # Y-coordinate of central latitude (km)
'Rt'    : 6371    # Earth radius (km)
'beta0' : None    # None = auto-calculate; 0 = constant f-plane
```

### Modal analysis parameters
```python
'n_modes'       : 9    # Number of modes to compute
'period_target' : 16   # Target period (hours) — modes near this are returned
```

#### Choosing `period_target`
The solver returns `n_modes` eigenvalues closest to the target frequency.
Set it to roughly the period of the modes you are most interested in:

```python
'period_target': 6    # modes near 6 h
'period_target': 12   # modes near 12 h (inertial period at mid-latitudes)
'period_target': 24   # modes near 24 h
'period_target': 72   # modes near 72 h (long basin seiches)
```

---

## Running the Analysis

```python
results = analysis.run_analysis(
    shoreline_file='data/shoreline.txt',
    bathymetry_file='data/bathymetry.xyz',
    verbose=True,
)
```

### Accessing results

```python
# Mode periods (hours), sorted longest first
periods = results['eigenvalues']['wshorasID'][:, 0]

# Index mapping: wshorasID[:,1] gives the original eigensolver index
# Interface displacement — shape (ny, nx, n_modes), complex
eta2 = results['eigenvalues']['eta2tilde']

# Eigenvalues (complex angular frequencies, rad/s)
ws = results['eigenvalues']['ws']

# Modal energy
etilde = results['eigenvalues']['etilde']

# Velocity fields (subsampled, complex), shapes (ny_sub, nx_sub, n_modes)
u1s = results['velocities']['u1s']
v1s = results['velocities']['v1s']
xs  = results['velocities']['xs']   # coordinates of subsampled points
ys  = results['velocities']['ys']
```

---

## Plotting in Jupyter

### Correct pattern

```python
import matplotlib.pyplot as plt
from IPython.display import display

fig = analysis.plot_modes(n_modes=9, save_path='modes.png')
display(fig)       # renders inline in Jupyter
plt.close(fig)     # prevents duplicate rendering at end of cell
```

```python
# Single mode
fig = analysis.plot_single_mode(mode_idx=0)
display(fig)
plt.close(fig)
```

### Important notes

- **Do not** call `plt.show(fig)` — `plt.show()` takes no arguments and
  cannot target a specific figure.
- **Do not** call `matplotlib.use(...)` after importing the package —
  the backend is already set by the package.
- `display(fig)` is available automatically in Jupyter; if you get a
  `NameError` add `from IPython.display import display` at the top of
  the cell.
- Always call `plt.close(fig)` after `display(fig)` to release memory
  and avoid the figure appearing twice.

### Time evolution at a specific time

```python
import numpy as np

mode_idx = 0
omega = results['eigenvalues']['ws'][mode_idx]
eta2  = results['eigenvalues']['eta2tilde'][:, :, mode_idx]

t = 3600  # 1 hour in seconds
amp_t = np.exp(1j * omega * t)
eta_t = np.real(amp_t * eta2 + np.conj(amp_t) * np.conj(eta2))
```

---

## Saving and Loading

```python
# Save all key arrays to a compressed .npz file
analysis.save_results('results/my_results.npz')

# Load later (no need to re-run the analysis)
data = ModalAnalysis.load_results('results/my_results.npz')
xeta     = data['xeta']
eta2     = data['eta2tilde']
wshorasID = data['wshorasID']
```

---

## Advanced Topics

### Seasonal comparison

```python
config_summer = {**config, 'rho1': 997.0, 'rho2': 999.5}
config_winter = {**config, 'rho1': 999.0, 'rho2': 999.3}

for name, cfg in [('summer', config_summer), ('winter', config_winter)]:
    a = ModalAnalysis(cfg)
    r = a.run_analysis(shore, bathy, verbose=False)
    print(f"{name}  mode-1 period: {r['eigenvalues']['wshorasID'][0,0]:.2f} h")
```

### Animation

```python
anim = analysis.animate_modes(n_modes=9, n_frames=64, dt=600,
                               save_path='animation.gif')
```

### Parameter study

```python
for h1 in [10, 20, 30, 40]:
    cfg = {**config, 'h1': h1}
    a = ModalAnalysis(cfg)
    r = a.run_analysis(shore, bathy, verbose=False)
    print(f"h1={h1:3d} m   mode-1: {r['eigenvalues']['wshorasID'][0,0]:.2f} h")
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Plots not showing in Jupyter | `matplotlib.use('Agg')` called somewhere | Use `display(fig)` not `plt.show()` |
| `TypeError: show() takes 1 argument` | `plt.show(fig)` called | Use `display(fig)` instead |
| Eigenvalue solver does not converge | `period_target` far from actual modes | Move `period_target` closer to expected period |
| Mode shapes look like noise | Wrong eigensolver | Ensure `eigs` (not `eigsh`) is used — see `eigensolve.py` |
| Matrix size mismatch | Contour extraction failed | Check `h1` < max depth in bathymetry |
| Distorted or unrealistic modes | `rho2 ≤ rho1`, or `h1 ≥ total depth` | Check density values and layer thickness |
| Very long computation time | Grid too fine or domain too large | Increase `dx`/`dy` or reduce domain |
