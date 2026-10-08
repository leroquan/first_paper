To detect eddies after removing all saved modes, set these fields in the SWIRL JSON config:

```json
{
  "remove_iw": true,
  "lake_name": "zug",
  "lake_config_path": "../../../../modal_analysis/horizontal_mode_analysis/lake_config.yaml"
}
```

`lake_config_path` is relative to the JSON config file. The selected YAML lake's
`modes_file` identifies its modal-analysis directory; shapes and coefficients
are read from `results/modal_decomposition/mode_results_YYYY-MM-DD.npz` beneath it.
Relative `modes_file` paths are resolved relative to the YAML file.

The NPZ files are produced by
`python/modal_analysis/horizontal_mode_analysis/02-modal_decomposition_loop.py`.
The matching `YYYY-MM-DD_KE.csv` supplies timestamps in the saved coefficient
order. Reconstruction uses `2 * real(sum(atilde * shape, axis=mode))` with every
mode stored in each file, without an additional period or energy filter.
The producer itself selects which modes to save. Layer 1 applies at `Z >= -h1`
and layer 2 below, using `h1` from `modes_file`. The unused outer C-grid face
is removed from each saved velocity shape. Overlapping decomposition windows
at requested timestamps raise an error to avoid ambiguous reconstructions.
The loader retains raw MITgcm data for lake-wide statistics. Before worker tasks
are built, a separate dataset replaces UVEL/VVEL with residuals for SWIRL
detection and eddy diagnostics, including eddy kinetic energy. Lake kinetic
energy is computed from the raw velocities. Optional NetCDF exports retain
the modal-residual velocities when removal is enabled.
Temperature, vertical velocity, grid metadata, and timestamp order are retained.
The transformation remains lazy. Missing coefficients, missing mode files, and undefined modal cells retain
the original MITgcm velocities. Snapshot order and count are preserved.

Saved mode shapes must match the loaded horizontal grid, ; coarsened mode shapes cannot be used on a finer SWIRL grid.
The input must expose the native C-grid dimensions and hFacW/hFacS masks.
IW removal is applied only where decomposition data exist. When false
or absent, `remove_iw` leaves the existing pipeline unchanged.
