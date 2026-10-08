"""Optional saved-mode removal before SWIRL workers receive velocity fields."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _reconstruct_window(
    ds, path, records, available, time_chunk, coarse_factor
):
    """Reconstruct saved modes on the full staggered MITgcm grid."""
    layers = {}

    def source_coordinates(dim, n_coarse):
        """Recreate coordinates for a trimmed, coarsened staggered axis.

        The saved modal fields are coarsened by ``coarse_factor`` relative to the
        MITgcm/SWIRL grid, so we must reconstruct the modal coordinates on the
        coarse grid and only then interpolate back to the full staggered grid.
        """
        target = ds[dim]

        if target.ndim != 1:
            raise ValueError(f"{dim} must be a one-dimensional coordinate")

        n_before_coarsening = n_coarse * coarse_factor
        n_extra = n_before_coarsening - target.size

        if n_extra <= -coarse_factor or n_extra > coarse_factor:
            raise ValueError(
                f"{path}: cannot map {n_coarse} coarse {dim} cells "
                f"to {target.size} target coordinates with factor "
                f"{coarse_factor}. (n_extra={n_extra})"
            )

        original = np.asarray(target.values)

        if n_extra < 0:
            # boundary="trim" drops an incomplete trailing block.
            original = original[:n_before_coarsening]
        elif n_extra > 0:
            # Saved staggered fields may include an extra outer face. Extend
            # the coordinate axis to include it before forming coarse blocks.
            if original.size < 2:
                raise ValueError(f"{path}: insufficient coordinates to reconstruct {dim}")
            step = original[-1] - original[-2]
            extras = original[-1] + step * np.arange(1, n_extra + 1)
            original = np.append(original, extras)

        return original.reshape(-1, coarse_factor).mean(axis=1)

    with np.load(path) as saved:
        coefficients = saved["atilde"]

        if (
            coefficients.ndim != 2
            or coefficients.shape[1] != len(records)
        ):
            raise ValueError(
                f"{path}: atilde must have shape "
                f"(mode, {len(records)})"
            )

        coefficients = coefficients[
            :, records.get_indexer(available)
        ]

        if not np.isfinite(coefficients).all():
            raise ValueError(
                f"Non-finite atilde coefficients in {path}"
            )

        atilde = xr.DataArray(
            coefficients,
            dims=("mode", "time"),
            coords={"time": available},
        ).chunk({"time": time_chunk})

        for key in ("u1", "v1", "u2", "v2"):
            dims = (
                ("YC", "XG") if key.startswith("u")
                else ("YG", "XC")
            )
            values = saved[key]

            if values.ndim != 3 or values.shape[-1] != coefficients.shape[0]:
                raise ValueError(
                    f"{path}: {key} has shape {values.shape}; "
                    f"expected (*, *, {coefficients.shape[0]})"
                )

            # Saved modal fields live on the coarse, staggered mode grid. Match
            # the original grid first and then interpolate the reconstruction onto
            # the full MITgcm/SWIRL grid used by the worker tasks.
            coordinates = {
                dim: source_coordinates(dim, values.shape[axis])
                for axis, dim in enumerate(dims)
            }

            mode_shape = xr.DataArray(
                values,
                dims=(*dims, "mode"),
                coords=coordinates,
            )

            # Retain NaNs in the modal reconstruction. Reconstruct on the
            # coarse grid first, then interpolate to the full grid.
            reconstructed = (
                2 * (atilde * mode_shape)
                .sum("mode", skipna=False)
                .real
            )

            targets = {dim: ds[dim] for dim in dims}
            if coarse_factor == 1:
                # Already on the native grid: avoid interpolation spreading
                # NaNs from adjacent land cells into valid modal cells.
                layers[key] = reconstructed.sel(targets)
            else:
                layers[key] = reconstructed.interp(
                    targets,
                    method="linear",
                    kwargs={"fill_value": "extrapolate"},
                )

    return xr.Dataset(layers)

def apply_modal_residuals(input_data, config):
    """Replace horizontal velocities lazily; retain all other SWIRL inputs."""
    enabled = config.get("remove_iw", False)
    if not isinstance(enabled, bool):
        raise ValueError("remove_iw must be a JSON boolean")
    if not enabled:
        return input_data

    import yaml

    lake_name = config.get("lake_name")
    if not lake_name or not config.get("lake_config_path"):
        raise ValueError("remove_iw requires lake_name and lake_config_path")
    path = Path(config["lake_config_path"]).expanduser()
    if not path.is_absolute():
        path = Path(config.get("_config_dir", Path.cwd())) / path
    path = path.resolve()
    with path.open() as stream:
        lake = yaml.safe_load(stream)[lake_name]
    coarse_factor = lake["coarsening_factor"]
    modes_file = Path(lake["modes_file"]).expanduser()
    if not modes_file.is_absolute():
        modes_file = path.parent / modes_file
    root = modes_file.parent / "results"
    ds = input_data.ds_mitgcm
    times = pd.DatetimeIndex(ds.time.values)
    if times.empty or not times.is_unique:
        raise ValueError("Modal removal requires nonempty, unique snapshot timestamps")
    if not modes_file.is_file():
        return input_data
    parameters = pd.read_csv(modes_file, parse_dates=["date"])
    if not parameters["date"].is_unique:
        raise ValueError(f"Duplicate dates in {modes_file}")
    windows = []
    covered = pd.DatetimeIndex([])
    time_chunk = config.get("time_chunk_size", 1)
    if time_chunk == "auto":
        time_chunk = 1
    for parameter in parameters.itertuples(index=False):
        date = parameter.date
        shape = root / "modal_decomposition" / f"mode_results_{date:%Y-%m-%d}.npz"
        csv = root / "modal_decomposition" / f"{date:%Y-%m-%d}_KE.csv"
        if not shape.is_file() or not csv.is_file():
            continue
        # CSV row order is the time axis of atilde in the producer script.
        records = pd.DatetimeIndex(pd.read_csv(csv, usecols=["time"], parse_dates=["time"])["time"])
        if not records.is_unique or records.hasnans:
            raise ValueError(f"Invalid or duplicate timestamps in {csv}")
        available = times.intersection(records, sort=False)
        if available.empty:
            continue
        if not covered.intersection(available).empty:
            raise ValueError(f"Overlapping decomposition windows at {date:%Y-%m-%d}")
        if not np.isfinite(parameter.h1) or parameter.h1 <= 0:
            raise ValueError(f"Invalid h1 for {date:%Y-%m-%d}")
        layers = _reconstruct_window(
            ds, shape, records, available, time_chunk, coarse_factor
        )
        upper = ds.Z >= -parameter.h1
        windows.append(xr.Dataset({
            "u_modal": xr.where(upper, layers.u1, layers.u2),
            "v_modal": xr.where(upper, layers.v1, layers.v2),
        }))
        covered = covered.union(available)
    if not windows:
        return input_data
    modal = xr.concat(windows, dim="time").reindex(time=ds.time)
    result = ds.copy()
    # Keep the existing coordinates and auxiliary grid metadata exactly as loaded.
    for target, source, mask in (("UVEL", "u_modal", "hFacW"),
                                 ("VVEL", "v_modal", "hFacS")):
        velocity = (ds[target] - modal[source]).where(ds[mask] > 0).transpose(*ds[target].dims)
        # Missing times and undefined modal cells retain the original values.
        velocity = velocity.where(np.isfinite(velocity), ds[target])
        result[target] = ds[target].copy(data=velocity.data)
        result[target].attrs = dict(ds[target].attrs,
                                    long_name=f"{target} minus all saved modes where available")
    result.attrs.update(remove_iw=True, lake_config_path=str(path),
                        velocity_source="MITgcm minus all saved modes where available; otherwise MITgcm")
    return input_data._replace(ds_mitgcm=result)
