import xarray as xr
import numpy as np
import dask

from open_files_functions import get_mitgcm_grid


def center_mitgcm_results(ds):
    u_shift = (ds['UVEL']
               .rolling(XG=2, center=False).mean()
               .shift(XG=-1)  # shift forward so it’s (i, i+1) instead of (i-1, i)
               .rename({"XG": "XC"})
               )
    u_shift['XC'] = ds['XC']

    v_shift = (ds['VVEL']
               .rolling(YG=2, center=False).mean()
               .shift(YG=-1)  # shift forward so it’s (i, i+1) instead of (i-1, i)
               .rename({"YG": "YC"}))
    v_shift['YC'] = ds['YC']

    w_shift = (ds['WVEL']
               .shift(Zl=1).fillna(0)  # shift to get a 0 at the surface
               .rolling(Zl=2, center=False).mean()
               .shift(Zl=-1)  # shift back
               .rename({"Zl": "Z"}))
    w_shift['Z'] = ds['Z']

    return u_shift, v_shift, w_shift


def reformat_and_save_mitgcm_results(ds, times, depths, grid_folder, output_path, nodata=-999.0):
    grid = get_mitgcm_grid(grid_folder)

    u_shift, v_shift, w_shift = center_mitgcm_results(ds)

    # Optional rotation
    if "rotation" in grid.parameters:
        print(f"Rotating u,v by {-grid.parameters['rotation']}°")
        theta_rad = np.deg2rad(-grid.parameters["rotation"])
        u = u_shift * np.cos(theta_rad) - v_shift * np.sin(theta_rad)
        v = u_shift * np.sin(theta_rad) + v_shift * np.cos(theta_rad)
        w = w_shift
    else:
        u, v, w = u_shift, v_shift, w_shift

    # Apply nodata mask
    print(f"Applying nodata mask")
    theta = ds['THETA']
    mask = (theta == 0.0) | np.isnan(theta)
    for arr in [theta, w, u, v]:
        arr = arr.where(~mask, nodata)

    # Build dataset
    print(f"Building dataset")
    theta.attrs = {"units": "°C", "long_name": "Temperature"}
    u.attrs = {"units": "m/s", "long_name": "Eastward velocity"}
    v.attrs = {"units": "m/s", "long_name": "Northward velocity"}
    w.attrs = {"units": "m/s", "long_name": "Vertical velocity"}

    #theta = theta.chunk({"time": 10, "Z": 5, "YC": 100, "XC": 100})
    #u = u.chunk(theta.chunks)
    #v = v.chunk(theta.chunks)
    #w = w.chunk(theta.chunks)

    # Build dataset
    ds = xr.Dataset(
        {
            "t": (("time", "depth", "Y", "X"), theta.data, {"units": "°C", "long_name": "Temperature"}),
            "u": (("time", "depth", "Y", "X"), u.data, {"units": "m/s", "long_name": "Eastward velocity"}),
            "v": (("time", "depth", "Y", "X"), v.data, {"units": "m/s", "long_name": "Northward velocity"}),
            "w": (("time", "depth", "Y", "X"), w.data, {"units": "m/s", "long_name": "Vertical velocity"}),
        },
        coords={
            "time": ("time", times, {"long_name": "time"}),
            "depth": ("depth", depths, {"units": "m", "long_name": "Depth below surface"}),
            "lat": (("Y", "X"), grid.lat_grid, {"long_name": "Latitude"}),
            "lng": (("Y", "X"), grid.lon_grid, {"long_name": "Longitude"}),
        },
        attrs={"MITgcm_version": "MITgcm-checkpoint67z"},
    )

    print(f"Saving dataset")
    with dask.config.set(scheduler="single-threaded"):
        # Prepare encoding with proper chunksizes
        encoding = {}
        for var in ds.data_vars:
            if hasattr(ds[var].data, 'chunks'):  # Dask-backed array
                # Take the first chunk along each dimension
                encoding[var] = {"chunksizes": tuple(c[0] for c in ds[var].data.chunks)}
            else:
                encoding[var] = {}  # in-memory array, no chunking needed

        # Write dataset chunk by chunk
        ds.to_netcdf(
            output_path,
            format="NETCDF4",
            engine="netcdf4",
            compute=True,
            encoding=encoding
        )