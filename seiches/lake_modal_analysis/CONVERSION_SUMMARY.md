# MATLAB to Python Conversion Summary

## Package: Lake Modal Analysis

### Conversion Details

**Original Code:** 15 MATLAB files by Carlos Rozas Rojas (2010)  
**Python Version:** Complete package with 8 core modules + documentation  
**Initial Conversion:** December 2025  
**Bug-fix revision:** March 2026  
**License:** GNU GPL v3.0 (maintained from original)

---

## File Mapping

### MATLAB → Python Module Correspondence

| MATLAB File | Python Module | Description |
|-------------|---------------|-------------|
| `ModalAnalysis_Geneva.m` | `modal_analysis.py` | Main orchestration class |
| `generaMalla.m` | `mesh.py` | Mesh generation with staggered grid |
| `generaBatimetria.m` | `bathymetry.py` | Bathymetry processing |
| `fvariable.m` | `coriolis.py` | Coriolis parameter calculation |
| `generaMatrices.m` | `matrices.py` | System matrix assembly (K and M) |
| `valorespropios.m` | `eigensolve.py` | Eigenvalue problem solver |
| `Velocidades.m` | `velocities.py` | Velocity field processing |
| `Grafico_modos.m` | `visualization.py` | Mode shape plotting |
| `animacion_modos.m` | `visualization.py` | Animation functions |
| `Grafico_modos_mar.m` | `visualization.py` | Advanced plotting |
| `fases_modos.m` | (integrated) | Phase evolution (in visualizer) |
| `fases_modos_1plot.m` | (integrated) | Single plot phases |
| `animacion_modos_presentacion.m` | (integrated) | Presentation animation |
| `esquema_malla.m` | (optional) | Mesh schematic plotting |
| `plot_results.m` | `visualization.py` | Results plotting |

---

## Package Structure

```
lake_modal_analysis/
├── __init__.py                  # Package initialization
├── modal_analysis.py            # Main class (replaces ModalAnalysis_Geneva.m)
├── mesh.py                      # Mesh generation (replaces generaMalla.m)
├── bathymetry.py               # Bathymetry (replaces generaBatimetria.m)
├── coriolis.py                 # Coriolis (replaces fvariable.m)
├── matrices.py                 # Matrices (replaces generaMatrices.m)
├── eigensolve.py               # Eigenvalues (replaces valorespropios.m)
├── velocities.py               # Velocities (replaces Velocidades.m)
├── visualization.py            # All plotting (replaces multiple .m files)
├── examples/
│   ├── rectangle/
│   │   ├── lake_1km_shoreline.txt
│   │   └── lake_1km_bathymetry.xyz
│   └── example_geneva.py       # Lake Geneva example
├── README.md
├── CONVERSION_SUMMARY.md
├── USAGE.md
├── requirements.txt
└── setup.py
```

---

## Bug Fixes (March 2026)

Four bugs were identified and corrected by systematic side-by-side comparison
with the original MATLAB code.

### Bug 1 — `matrices.py`: Wrong Coriolis column offsets (critical)

**Root cause:** The MATLAB code fills the full matrix K first and removes
boundary-condition rows/columns at the end.  The Python translation
incorrectly used *post-removal* column offsets (`iniu1`, `iniv1`, `iniu2`,
`iniv2`) when filling the Coriolis rotation terms, even though column removal
had not yet happened at that point.

**Effect:** All Coriolis coupling terms were written to completely wrong
positions in K, corrupting every mode shape and period.

| Block | Wrong offset | Correct offset |
|-------|-------------|----------------|
| u1 rotation → v1 columns | `iniv1 + idx` | `iv1 + idx` |
| v1 rotation → u1 columns | `iniu1 + idx` | `iu1 + idx` |
| u2 rotation → v2 columns | `iniv2 + idx` | `iv2 + idx` |
| v2 rotation → u2 columns | `iniv2 + idx` | `iu2 + idx` |

### Bug 2 — `eigensolve.py`: Wrong eigensolver (critical)

**Root cause:** `scipy.sparse.linalg.eigsh` was used instead of `eigs`.
`eigsh` is for symmetric/Hermitian matrices only.  K is complex and
non-Hermitian because the Coriolis terms are antisymmetric.

**Effect:** The solver either failed to converge or returned physically
meaningless eigenvalues.

**Fix:** Replaced `eigsh` with `eigs` throughout.  After the fix, imaginary
parts of all eigenvalues are at machine epsilon (~1e-16), confirming the
system is correctly modelled as conservative.

### Bug 3 — `velocities.py`: ComplexWarning (minor)

**Root cause:** `u1_plot` and `v1_plot` arrays were initialised with
`dtype=float` (via `np.full(..., np.nan)`) but then assigned complex values
from `u1tilde`/`v1tilde`.

**Fix:** Arrays now initialised with `dtype=complex`.

### Bug 4 — `mesh.py`: `matplotlib.use('Agg')` poisoning the session

**Root cause:** `matplotlib.use('Agg')` was called at module import time to
provide a headless backend for the contour extraction step.  This call is
global and permanent — once executed it switches the entire Python session
to a non-display backend, silently preventing all subsequent Jupyter inline
plots from rendering.

**Fix:** Removed `matplotlib.use('Agg')`.  The h1-depth contour is now
extracted using `matplotlib.figure.Figure()` directly (a pure object-level
figure, completely detached from the display system), so the global pyplot
state is never modified.

---

## Key Differences from MATLAB Version

### 1. Object-Oriented Design
- MATLAB: Script-based with global variables passed through the workspace
- Python: Modular classes with explicit interfaces

### 2. Indexing
- MATLAB: 1-based indexing
- Python: 0-based indexing
- All node index arrays (`IDeta1`, `IDu1`, etc.) store 0-based indices.
  The `-1` sentinel value indicates "no node" (equivalent to MATLAB's `0`
  in 1-based node numbering).

### 3. Sparse Matrices
- MATLAB: `sparse()`, `eigs(K, M, n, sigma)`
- Python: `scipy.sparse`, `scipy.sparse.linalg.eigs(K, k=n, M=M, sigma=sigma)`

### 4. Complex Numbers
- MATLAB: `1i`
- Python: `1j`

### 5. Plotting in Jupyter
- Do **not** use `plt.show(fig)` — `plt.show()` takes no arguments.
- Use `display(fig)` to show a specific figure inline in Jupyter.
- Use `plt.close(fig)` after display to prevent duplicate rendering.

---

## Technical Translation Notes

### Matrix Assembly Strategy
The MATLAB code fills the complete K matrix (including boundary rows/columns)
and then strips them at the end with two index operations:
```matlab
K([u1del,v1del,u2del,v2del],:)=[];   % remove rows
K(:,[u1del,v1del,u2del,v2del])=[];   % remove columns
```
The Python code replicates this exactly: rows are removed after filling eta
equations, columns are removed after filling all velocity equations.  Column
offsets used while filling velocity equations must refer to the
**pre-removal** layout (`iu1`, `iv1`, `iu2`, `iv2`).

### Eigensolver
`scipy.sparse.linalg.eigs` is the direct equivalent of MATLAB's `eigs`.
Both use shift-invert Arnoldi iteration.  The call signature is:
```python
omega_vals, xi = eigs(K, k=n_modes, M=M, sigma=freq_target, which='LM')
```

---

## Validation

### Rectangular basin test
A 1 km × 0.5 km flat-bottom basin (depth 20 m, h1 = 10 m, dx = dy = 25 m)
was run to verify the fixed code.  Results:

- Matrix size: 4330 × 4330 (after BC removal)
- All `imag(ω)` < 1e-15 (machine epsilon) ✓
- Mode shapes show expected longitudinal and transverse seiche patterns ✓

### Lake Geneva
Run with actual Geneva shoreline and bathymetry to compare against original
MATLAB output.

---

## Dependencies

### Required
- Python ≥ 3.7
- NumPy ≥ 1.18.0
- SciPy ≥ 1.4.0
- Matplotlib ≥ 3.1.0

---

## Version History

**v1.1.0 (March 2026)**
- Fixed Coriolis column offsets in `matrices.py` (critical)
- Fixed eigensolver: `eigsh` → `eigs` in `eigensolve.py` (critical)
- Fixed complex array initialisation in `velocities.py` (minor)
- Fixed matplotlib backend poisoning in `mesh.py` (usability)

**v1.0.0 (December 2025)**
- Initial Python conversion from MATLAB
