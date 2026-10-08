from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def repair_cartesian_coordinates(
    ds: xr.Dataset,
    grid_spacing: float = 100.0,
) -> xr.Dataset:
    """
    Assign regular Cartesian coordinates to an MITgcm C-grid dataset.

    Coordinates are assigned by position only. No interpolation or
    reindexing of the velocity fields is performed.

    Parameters
    ----------
    ds : xr.Dataset
        MITgcm dataset containing XC, YC, XG and YG dimensions.
    grid_spacing : float, default 100
        Horizontal grid spacing in metres.

    Returns
    -------
    xr.Dataset
        Dataset with repaired XC, YC, XG and YG coordinates.
    """
    ds = ds.copy()

    for dim, offset in (
        ("XC", 0.5),
        ("YC", 0.5),
        ("XG", 0.0),
        ("YG", 0.0),
    ):
        if dim not in ds.dims:
            raise ValueError(f"MITgcm dataset is missing dimension {dim!r}")

        axis = dim[0]

        attrs = {
            "units": "m",
            "long_name": f"Grid {axis.lower()}",
            "axis": axis,
        }

        if dim.endswith("G"):
            attrs["c_grid_axis_shift"] = -0.5

        ds = ds.assign_coords(
            {
                dim: xr.DataArray(
                    (np.arange(ds.sizes[dim]) + offset) * grid_spacing,
                    dims=dim,
                    attrs=attrs,
                )
            }
        )

    return ds


def load_coefficients(
    decomposition_dir: str | Path,
    time_start=None,
    time_end=None,
    coefficient_file: str = "KE_V1H1.csv",
) -> pd.DataFrame:
    """
    Load modal coefficients.

    Parameters
    ----------
    decomposition_dir : Path-like
        Directory containing the decomposition CSV files.
    time_start, time_end : optional
        Time limits understood by pandas.
    coefficient_file : str
        Combined coefficient filename.

    Returns
    -------
    pd.DataFrame
        Coefficients indexed by time.
    """
    decomposition_dir = Path(decomposition_dir)

    path = decomposition_dir / coefficient_file

    if not path.is_file():
        raise FileNotFoundError(path)

    coefficients = (
        pd.read_csv(
            path,
            parse_dates=["time"],
            index_col="time",
        )
        .sort_index()
    )

    if not coefficients.index.is_unique:
        raise ValueError(f"Duplicate timestamps in {path}")

    required = {"amplitude", "phase_rad"}
    missing = required - set(coefficients.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    coefficients = coefficients.loc[time_start:time_end].copy()

    if coefficients.empty:
        raise ValueError("No coefficients in requested time range")

    if not np.isfinite(
        coefficients[["amplitude", "phase_rad"]].to_numpy()
    ).all():
        raise ValueError("Non-finite amplitude or phase")

    return coefficients


def match_coefficients_to_modes(
    coefficients: pd.DataFrame,
    parameters: pd.DataFrame,
    decomposition_dir: str | Path,
) -> pd.DataFrame:
    """
    Determine exactly which saved modal shape belongs to each coefficient.

    This checks the original per-date decomposition files rather than
    assigning the nearest modal-analysis date.

    Parameters
    ----------
    coefficients : pd.DataFrame
        Combined coefficients indexed by time.
    parameters : pd.DataFrame
        Modal parameters indexed by modal-analysis date.
    decomposition_dir : Path-like
        Directory containing YYYY-MM-DD_KE.csv files.

    Returns
    -------
    pd.DataFrame
        Copy of coefficients containing an additional ``mode_date`` column.
    """
    decomposition_dir = Path(decomposition_dir)

    coefficients = coefficients.copy()
    coefficients["mode_date"] = pd.NaT

    if not parameters.index.is_unique:
        raise ValueError("Duplicate dates in mode parameters")

    for date in parameters.index:
        csv_path = decomposition_dir / f"{date:%Y-%m-%d}_KE.csv"

        if not csv_path.is_file():
            continue

        original = pd.read_csv(
            csv_path,
            parse_dates=["time"],
            index_col="time",
        )

        if not original.index.is_unique:
            raise ValueError(f"Duplicate timestamps in {csv_path}")

        times = coefficients.index.intersection(original.index)

        if len(times) == 0:
            continue

        if coefficients.loc[times, "mode_date"].notna().any():
            raise ValueError(
                f"Overlapping decomposition windows at {date:%Y-%m-%d}"
            )

        np.testing.assert_allclose(
            coefficients.loc[
                times, ["amplitude", "phase_rad"]
            ].to_numpy(),
            original.loc[
                times, ["amplitude", "phase_rad"]
            ].to_numpy(),
            err_msg=(
                f"Combined coefficients differ from "
                f"{csv_path.name}"
            ),
        )

        coefficients.loc[times, "mode_date"] = date

    missing = coefficients["mode_date"].isna()

    if missing.any():
        bad_times = coefficients.index[missing]

        raise ValueError(
            "Some coefficients cannot be matched to a saved "
            f"decomposition window. First unmatched time: {bad_times[0]}"
        )

    return coefficients


def load_mode_shapes(
    modes_dir: str | Path,
    mode_dates: pd.DatetimeIndex,
    ny: int,
    nx: int,
) -> dict[str, np.ndarray]:
    """
    Load modal velocity shapes from NPZ files.

    Saved NPZ velocity fields contain the outer C-grid face:
        u : (ny, nx + 1)
        v : (ny + 1, nx)

    MITgcm output used here contains:
        u : (ny, nx)
        v : (ny, nx)

    The unused outer face is therefore removed.

    Returns
    -------
    dict
        Arrays with dimensions:

        u1, u2 : (mode_date, YC, XG)
        v1, v2 : (mode_date, YG, XC)
    """
    modes_dir = Path(modes_dir)

    shapes = {
        key: []
        for key in ("u1", "v1", "u2", "v2")
    }

    for date in mode_dates:
        path = modes_dir / f"mode_results_{date:%Y-%m-%d}.npz"

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path) as saved:
            for key in shapes:
                if key not in saved:
                    raise KeyError(
                        f"{key!r} missing from {path}"
                    )

                if key.startswith("u"):
                    expected = (ny, nx + 1)
                else:
                    expected = (ny + 1, nx)

                if saved[key].shape != expected:
                    raise ValueError(
                        f"Unexpected shape for {date:%Y-%m-%d} "
                        f"{key}: {saved[key].shape}; "
                        f"expected {expected}"
                    )

                if key.startswith("u"):
                    field = saved[key][:, :-1]
                else:
                    field = saved[key][:-1, :]

                shapes[key].append(field)

    return {
        key: np.stack(values)
        for key, values in shapes.items()
    }


def reconstruct_layer_velocities(
    ds_mitgcm: xr.Dataset,
    coefficients: pd.DataFrame,
    shapes: dict[str, np.ndarray],
    mode_dates: pd.DatetimeIndex,
    time_chunk: int | None = 24,
) -> xr.Dataset:
    """
    Reconstruct V1H1 layer velocities from complex mode shapes.

    Reconstruction is

        velocity(t) =
            2 * Re[
                amplitude(t)
                * exp(i * phase(t))
                * mode_shape
            ]

    Parameters
    ----------
    ds_mitgcm : xr.Dataset
        MITgcm dataset selected at coefficient timestamps.
    coefficients : pd.DataFrame
        Must contain amplitude, phase_rad and mode_date.
    shapes : dict
        Output from ``load_mode_shapes``.
    mode_dates : pd.DatetimeIndex
        Dates corresponding to the first dimension of ``shapes``.
    time_chunk : int or None
        Dask chunk size along time. If None, no explicit chunking is
        performed.

    Returns
    -------
    xr.Dataset
        Variables u1, v1, u2 and v2.
    """
    mode_index_values = mode_dates.get_indexer(
        pd.DatetimeIndex(coefficients["mode_date"])
    )

    if np.any(mode_index_values < 0):
        raise ValueError(
            "At least one coefficient references an unknown mode date"
        )

    mode_index = xr.DataArray(
        mode_index_values,
        dims="time",
        coords={"time": ds_mitgcm.time},
    )

    atilde = xr.DataArray(
        coefficients["amplitude"].to_numpy()
        * np.exp(
            1j
            * coefficients["phase_rad"].to_numpy()
        ),
        dims="time",
        coords={"time": ds_mitgcm.time},
        name="atilde",
    )

    if time_chunk is not None:
        atilde = atilde.chunk({"time": time_chunk})

    layer_velocities = {}

    for key, values in shapes.items():

        if key.startswith("u"):
            dims = ("YC", "XG")
        else:
            dims = ("YG", "XC")

        shape = xr.DataArray(
            values,
            dims=("mode", *dims),
            coords={
                dim: ds_mitgcm[dim]
                for dim in dims
            },
            name=f"{key}_shape",
        )

        # Loading one mode-date at a time is generally convenient
        # because typically only a small number of mode dates exist.
        shape = shape.chunk({"mode": 1})

        selected_shape = shape.isel(mode=mode_index)

        reconstructed = (
            2 * (atilde * selected_shape).real
        )

        reconstructed.name = key
        reconstructed.attrs.update(
            units="m s-1",
            long_name=f"Reconstructed modal {key} velocity",
        )

        layer_velocities[key] = reconstructed

    return xr.Dataset(layer_velocities)


def extract_residuals(
    ds_mitgcm: xr.Dataset,
    *,
    modes_file: str | Path,
    decomposition_dir: str | Path,
    modes_dir: str | Path | None = None,
    time_start=None,
    time_end=None,
    time_index=None,
    mode_name: str = "V1H1",
    grid_spacing: float = 100.0,
    time_chunk: int | None = 24,
    repair_coordinates: bool = True,
) -> xr.Dataset:
    """
    Extract MITgcm residual velocities after removing a modal reconstruction.

    The MITgcm dataset is supplied directly; this function does not open
    MITgcm MDS files.

    Parameters
    ----------
    ds_mitgcm : xr.Dataset
        Dataset containing at least:

            UVEL(time, Z, YC, XG)
            VVEL(time, Z, YG, XC)
            hFacW
            hFacS

    modes_file : Path-like
        CSV containing mode parameters, including ``date`` and ``h1``.

    decomposition_dir : Path-like
        Directory containing:

            KE_V1H1.csv
            YYYY-MM-DD_KE.csv

    modes_dir : Path-like, optional
        Directory containing:

            mode_results_YYYY-MM-DD.npz

        Defaults to sibling ``modes`` directory relative to
        ``decomposition_dir``.

    time_start, time_end : optional
        Optional time selection.

    time_index : optional sequence of timestamps
        Exact requested snapshots, in output order. Missing coefficients raise
        an error. Useful for sparse batches supplied by postprocessing.

    mode_name : str, default "V1H1"
        Name stored in dataset metadata.

    grid_spacing : float, default 100
        Horizontal grid spacing in metres.

    time_chunk : int or None, default 24
        Time chunk for reconstructed modal arrays.

    repair_coordinates : bool, default True
        Replace XC/YC/XG/YG by regular Cartesian coordinates.

    Returns
    -------
    xr.Dataset
        Dataset containing:

            u_mitgcm
            v_mitgcm
            u_v1h1
            v_v1h1
            u_residual
            v_residual
            h1
            mode_date

    Notes
    -----
    The vertical mapping is:

        layer 1 : Z >= -h1
        layer 2 : Z <  -h1

    Residuals are:

        u_residual = u_mitgcm - u_v1h1
        v_residual = v_mitgcm - v_v1h1

    Undefined/land values remain NaN.
    """
    modes_file = Path(modes_file)
    decomposition_dir = Path(decomposition_dir)

    if modes_dir is None:
        modes_dir = decomposition_dir.parent / "modes"

    modes_dir = Path(modes_dir)

    # ------------------------------------------------------------
    # Validate MITgcm dataset
    # ------------------------------------------------------------

    required_variables = {
        "UVEL",
        "VVEL",
        "hFacW",
        "hFacS",
    }

    missing = required_variables - set(ds_mitgcm.variables)

    if missing:
        raise ValueError(
            "MITgcm dataset is missing required variables: "
            f"{sorted(missing)}"
        )

    required_dims = {
        "time",
        "Z",
        "XC",
        "YC",
        "XG",
        "YG",
    }

    missing_dims = required_dims - set(ds_mitgcm.dims)

    if missing_dims:
        raise ValueError(
            "MITgcm dataset is missing required dimensions: "
            f"{sorted(missing_dims)}"
        )

    ds_mitgcm = ds_mitgcm.copy()

    if repair_coordinates:
        ds_mitgcm = repair_cartesian_coordinates(
            ds_mitgcm,
            grid_spacing=grid_spacing,
        )

    # ------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------

    coefficient_filename = f"KE_{mode_name}.csv"

    coefficients = load_coefficients(
        decomposition_dir,
        time_start=time_start,
        time_end=time_end,
        coefficient_file=coefficient_filename,
    )

    if time_index is not None:
        requested = pd.DatetimeIndex(time_index)
        missing = requested.difference(coefficients.index)
        if len(missing):
            raise ValueError(
                f"Missing V1H1 coefficients for {len(missing)} snapshots; "
                f"first missing timestamp: {missing[0]}"
            )
        coefficients = coefficients.loc[requested].copy()

    # ------------------------------------------------------------
    # Mode parameters
    # ------------------------------------------------------------

    if not modes_file.is_file():
        raise FileNotFoundError(modes_file)

    parameters = (
        pd.read_csv(
            modes_file,
            parse_dates=["date"],
        )
        .set_index("date")
        .sort_index()
    )

    if not parameters.index.is_unique:
        raise ValueError(
            f"Duplicate dates in {modes_file}"
        )

    if "h1" not in parameters.columns:
        raise ValueError(
            f"{modes_file} does not contain an h1 column"
        )

    # ------------------------------------------------------------
    # Exact mode-date matching
    # ------------------------------------------------------------

    coefficients = match_coefficients_to_modes(
        coefficients,
        parameters,
        decomposition_dir,
    )

    mode_dates = pd.DatetimeIndex(
        sorted(
            pd.DatetimeIndex(
                coefficients["mode_date"]
            ).unique()
        )
    )

    # ------------------------------------------------------------
    # Exact MITgcm time matching
    # ------------------------------------------------------------

    requested_times = coefficients.index.to_numpy()

    available_times = pd.DatetimeIndex(
        ds_mitgcm.time.values
    )

    missing_times = coefficients.index.difference(
        available_times
    )

    if len(missing_times):
        raise ValueError(
            f"{len(missing_times)} coefficient timestamps "
            "are missing from MITgcm results. "
            f"First missing timestamp: {missing_times[0]}"
        )

    ds_mitgcm = ds_mitgcm.sel(
        time=requested_times
    )

    np.testing.assert_array_equal(
        ds_mitgcm.time.values,
        requested_times,
    )

    # ------------------------------------------------------------
    # Validate grid
    # ------------------------------------------------------------

    ny = ds_mitgcm.sizes["YC"]
    nx = ds_mitgcm.sizes["XC"]

    if ds_mitgcm.sizes["YG"] != ny:
        raise ValueError(
            f"YG={ds_mitgcm.sizes['YG']} but YC={ny}"
        )

    if ds_mitgcm.sizes["XG"] != nx:
        raise ValueError(
            f"XG={ds_mitgcm.sizes['XG']} but XC={nx}"
        )

    # ------------------------------------------------------------
    # Mode shapes
    # ------------------------------------------------------------

    shapes = load_mode_shapes(
        modes_dir=modes_dir,
        mode_dates=mode_dates,
        ny=ny,
        nx=nx,
    )

    # ------------------------------------------------------------
    # Reconstruct layer velocities
    # ------------------------------------------------------------

    ds_layers = reconstruct_layer_velocities(
        ds_mitgcm,
        coefficients,
        shapes,
        mode_dates,
        time_chunk=time_chunk,
    )

    # ------------------------------------------------------------
    # h1(t)
    # ------------------------------------------------------------

    h1_values = parameters.loc[
        pd.DatetimeIndex(
            coefficients["mode_date"]
        ),
        "h1",
    ].to_numpy()

    h1 = xr.DataArray(
        h1_values,
        dims="time",
        coords={"time": ds_mitgcm.time},
        name="h1",
    )

    if not np.isfinite(h1.values).all():
        raise ValueError(
            "Non-finite h1 values"
        )

    if not (h1.values > 0).all():
        raise ValueError(
            "h1 must be strictly positive"
        )

    # ------------------------------------------------------------
    # Map two-layer modes onto MITgcm vertical levels
    # ------------------------------------------------------------

    upper_layer = ds_mitgcm.Z >= -h1

    u_modal = xr.where(
        upper_layer,
        ds_layers["u1"],
        ds_layers["u2"],
    ).transpose(
        "time",
        "Z",
        "YC",
        "XG",
    )

    v_modal = xr.where(
        upper_layer,
        ds_layers["v1"],
        ds_layers["v2"],
    ).transpose(
        "time",
        "Z",
        "YG",
        "XC",
    )

    # ------------------------------------------------------------
    # Apply wet-face masks
    # ------------------------------------------------------------

    u_mitgcm = ds_mitgcm.UVEL.where(
        ds_mitgcm.hFacW > 0
    )

    v_mitgcm = ds_mitgcm.VVEL.where(
        ds_mitgcm.hFacS > 0
    )

    u_modal = u_modal.where(
        ds_mitgcm.hFacW > 0
    )

    v_modal = v_modal.where(
        ds_mitgcm.hFacS > 0
    )

    # ------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------

    with xr.set_options(arithmetic_join="exact"):

        ds_residuals = xr.Dataset(
            {
                "u_mitgcm": u_mitgcm,
                "v_mitgcm": v_mitgcm,

                "u_v1h1": u_modal,
                "v_v1h1": v_modal,

                "u_residual": u_mitgcm - u_modal,
                "v_residual": v_mitgcm - v_modal,

                "h1": h1,

                "mode_date": (
                    "time",
                    coefficients[
                        "mode_date"
                    ].to_numpy(),
                ),
            }
        )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    for name in ds_residuals.data_vars:
        if name.startswith(("u_", "v_")):
            ds_residuals[name].attrs[
                "units"
            ] = "m s-1"

    ds_residuals["h1"].attrs.update(
        units="m",
        long_name="Upper-layer thickness",
    )

    ds_residuals.attrs.update(
        mode=mode_name,
        reconstruction=(
            "2 * real("
            "amplitude * exp(1j * phase_rad) "
            "* saved complex mode shape)"
        ),
        vertical_mapping=(
            "Layer 1 where Z >= -h1; "
            "layer 2 below; "
            "native staggered C-grid"
        ),
        coefficients_file=str(
            decomposition_dir
            / coefficient_filename
        ),
        mode_shapes_directory=str(
            modes_dir
        ),
        modes_file=str(
            modes_file
        ),
    )

    # ------------------------------------------------------------
    # Coordinate consistency checks
    # ------------------------------------------------------------

    if repair_coordinates:

        for dim, offset in (
            ("XC", 0.5),
            ("YC", 0.5),
            ("XG", 0.0),
            ("YG", 0.0),
        ):
            expected = (
                np.arange(
                    ds_residuals.sizes[dim]
                )
                + offset
            ) * grid_spacing

            np.testing.assert_array_equal(
                ds_residuals[
                    dim
                ].values,
                expected,
            )

            if not ds_residuals.indexes[
                dim
            ].is_unique:
                raise ValueError(
                    f"{dim} coordinate is not unique"
                )

            if (
                ds_residuals[dim]
                .attrs.get("units")
                != "m"
            ):
                raise ValueError(
                    f"{dim} does not have metre units"
                )

    return ds_residuals