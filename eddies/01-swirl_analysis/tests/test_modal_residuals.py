import importlib.util
from pathlib import Path
from collections import namedtuple
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

spec = importlib.util.spec_from_file_location(
    "modal_residuals", Path(__file__).resolve().parents[1] / "functions/modal_residuals.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Input = namedtuple("Input", "ds_mitgcm dx")


class ModalResidualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        modes = root / "results/modes"
        decomposition = root / "results/modal_decomposition"
        modes.mkdir(parents=True)
        decomposition.mkdir()
        self.times = pd.date_range("2025-07-01", periods=3, freq="h")
        pd.DataFrame({"time": self.times, "KE_mode_MJ": [1., 2., 3.]}).to_csv(
            decomposition / "2025-07-07_KE.csv", index=False)
        pd.DataFrame({"date": ["2025-07-07"], "h1": [5.]}).to_csv(root / "modes.csv", index=False)
        fields = {}
        for key, shape, value in (("u1", (2, 3), .1), ("v1", (3, 2), .2),
                                  ("u2", (2, 3), .3), ("v2", (3, 2), .4)):
            fields[key] = np.stack([np.full(shape, value), np.full(shape, 1j * value)], axis=-1)
        np.savez(decomposition / "mode_results_2025-07-07.npz",
                 atilde=np.array([[.5, 9., .5], [-.5j, 8j, -.5j]]),
                 periods=[24., 12.], **fields)
        (root / "lake.yaml").write_text('zug:\n  modes_file: modes.csv\n  coarsening_factor: 1\n')
        self.config = dict(remove_iw=True, lake_name="zug", lake_config_path="lake.yaml", _config_dir=str(root))
        coords = dict(time=self.times[[0, 2]], Z=[-1., -10.], YC=[50., 150.], XC=[50., 150.], XG=[0., 100.], YG=[0., 100.])
        self.ds = xr.Dataset({
            "UVEL": (("time", "Z", "YC", "XG"), np.ones((2, 2, 2, 2))),
            "VVEL": (("time", "Z", "YG", "XC"), np.ones((2, 2, 2, 2))),
            "hFacW": (("Z", "YC", "XG"), np.ones((2, 2, 2))),
            "hFacS": (("Z", "YG", "XC"), np.ones((2, 2, 2))),
            "THETA": (("time", "Z", "YC", "XC"), np.full((2, 2, 2, 2), 12.)),
        }, coords=coords).chunk({"time": 1})

    def test_disabled_is_unchanged(self):
        data = Input(self.ds, 100.)
        self.assertIs(module.apply_modal_residuals(data, {}), data)

    def test_sparse_snapshots_use_residuals_and_preserve_other_data(self):
        result = module.apply_modal_residuals(Input(self.ds, 100.), self.config).ds_mitgcm
        self.assertTrue(hasattr(result.UVEL.data, "dask"))
        np.testing.assert_allclose(result.UVEL.isel(Z=0), .8)
        np.testing.assert_allclose(result.UVEL.isel(Z=1), .4)
        np.testing.assert_allclose(result.VVEL.isel(Z=0), .6)
        np.testing.assert_allclose(result.VVEL.isel(Z=1), .2)
        xr.testing.assert_identical(result.THETA, self.ds.THETA)
        xr.testing.assert_identical(result.time, self.ds.time)
        np.testing.assert_allclose(self.ds.UVEL, 1.)

    def test_missing_coefficients_keep_original(self):
        ds = self.ds.assign_coords(time=[self.times[0], self.times[2] + pd.Timedelta(minutes=30)])
        result = module.apply_modal_residuals(Input(ds, 100.), self.config).ds_mitgcm
        np.testing.assert_allclose(result.UVEL.isel(time=0, Z=0), .8)
        np.testing.assert_allclose(result.UVEL.isel(time=1), 1.)
        xr.testing.assert_identical(result.time, ds.time)

    def test_no_mode_files_keeps_original(self):
        for path in Path(self.temp.name).glob("results/modal_decomposition/*.npz"):
            path.unlink()
        data = Input(self.ds, 100.)
        self.assertIs(module.apply_modal_residuals(data, self.config), data)

    def test_undefined_mode_cells_keep_original(self):
        path = next(Path(self.temp.name).glob("results/modal_decomposition/*.npz"))
        with np.load(path) as saved:
            fields = {key: saved[key].copy() for key in saved.files}
        fields["u1"][0, 0] = np.nan
        np.savez(path, **fields)
        result = module.apply_modal_residuals(Input(self.ds, 100.), self.config).ds_mitgcm
        np.testing.assert_allclose(result.UVEL.isel(Z=0, YC=0, XG=0), 1.)
        np.testing.assert_allclose(result.UVEL.isel(Z=0, YC=1, XG=1), .8)

    def test_coefficient_length_mismatch_raises(self):
        path = next(Path(self.temp.name).glob("results/modal_decomposition/*.npz"))
        with np.load(path) as saved:
            fields = dict(saved)
        fields["atilde"] = fields["atilde"][:, :2]
        np.savez(path, **fields)
        with self.assertRaisesRegex(ValueError, "atilde must have shape"):
            module.apply_modal_residuals(Input(self.ds, 100.), self.config)

    def test_dry_faces_keep_original(self):
        self.ds["hFacW"] = xr.zeros_like(self.ds.hFacW)
        result = module.apply_modal_residuals(Input(self.ds, 100.), self.config).ds_mitgcm
        np.testing.assert_allclose(result.UVEL, 1.)
        np.testing.assert_allclose(result.VVEL.isel(Z=0), .6)

    def test_trimmed_coarse_grid_reconstructs_linear_velocity(self):
        # Geneva's 245 fine YC cells leave one cell after 61 blocks of four.
        ds = xr.Dataset(coords={
            "YC": (np.arange(245) + .5) * 100,
            "YG": np.arange(245) * 100,
            "XC": (np.arange(10) + .5) * 100,
            "XG": np.arange(10) * 100,
        })
        fields = {}
        for key in ("u1", "u2", "v1", "v2"):
            if key.startswith("u"):
                y = (np.arange(61) * 4 + 2) * 100
                x = (np.arange(3) * 4 + 1.5) * 100
            else:
                y = (np.arange(62) * 4 + 1.5) * 100
                x = (np.arange(2) * 4 + 2) * 100
            fields[key] = (y[:, None] + 2 * x[None, :])[..., None]
        path = Path(self.temp.name) / "coarse.npz"
        np.savez(path, atilde=np.array([[.5]]), **fields)
        times = self.times[:1]
        layers = module._reconstruct_window(ds, path, times, times, 1, 4)
        for key in fields:
            ydim, xdim = ("YC", "XG") if key.startswith("u") else ("YG", "XC")
            expected = ds[ydim].values[:, None] + 2 * ds[xdim].values[None, :]
            np.testing.assert_allclose(layers[key].isel(time=0).compute(), expected)

    def test_incompatible_coarse_grid_raises(self):
        path = next(Path(self.temp.name).glob("results/modal_decomposition/*.npz"))
        ds = self.ds.reindex(YC=np.arange(20))
        with self.assertRaisesRegex(ValueError, "cannot map"):
            module._reconstruct_window(ds, path, self.times, self.times[:1], 1, 4)


if __name__ == "__main__":
    unittest.main()
