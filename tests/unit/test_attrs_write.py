# Technically not our code we're testing, but want to have advanced warning if this functionality breaks

import tempfile
import pathlib

import xarray as xr


def test_attrs_write():
    orig_ds = xr.Dataset()
    orig_ds.attrs["nested_attr"] = {"key": "value"}
    orig_ds.attrs["none_attr"] = None

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = pathlib.Path(tmpdir) / "nested.zarr"
        orig_ds.to_zarr(zarr_path)
        opened_ds = xr.open_zarr(zarr_path)
        assert opened_ds.nested_attr["key"] == "value"
        assert opened_ds.none_attr is None
