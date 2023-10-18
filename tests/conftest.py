import pytest
import json

import pandas as pd
import numpy as np
import xarray as xr

from examples.managers.chirps import CHIRPSFinal25
#
# Pytest fixtures, automatic fixtures, and plugins that will load automatically for the entire suite of tests.
#
# See the pytest_addoption notes for how to avoid option name conflicts.
#


def pytest_addoption(parser):
    """
    Automatically run by pytest at invocation. These options can be passed on the command line and will be forwarded
    to all functions that include `requests` as a parameter and then can be accessed as `requests.config.option.[NAME]`.

    Options are global whether they are defined in the root tests/ directory or in a subdirectory like tests/era5.
    Therefore, option names in era5/conftest.py and prism_zarr/conftest.py cannot be the same.

    For example, if you wanted an option like "time_chunk", it would either have to be defined here once and apply to
    both ERA5 and PRISM or would have to have to be defined with a different name in each, like "era5_time_chunk" and
    "prism_time_chunk".

    Per subdirectory addoptions are not supported:
    https://github.com/pytest-dev/pytest/issues/7974

    Neither are per subdirectory INI files:
    https://github.com/pytest-dev/pytest/discussions/7732
    """
    # Remove pass statement and add global command line flags here
    pass


@pytest.fixture
def create_heads_file_for_testing(heads_path):
    """
    Create the heads file only if it doesn't exist
    """
    if not heads_path.exists():
        with open(heads_path, "w") as heads:
            json.dump({}, heads)
        print(f"Created empty heads JSON at {heads_path}")
    else:
        print(f"Found existing heads JSON at {heads_path}")

@pytest.fixture()
def fake_dataset():
    time = xr.DataArray(np.arange(5), dims="time", coords={"time": np.arange(5)})
    lat = xr.DataArray(np.arange(10, 50, 10), dims="lat",
                        coords={"lat": np.arange(10, 50, 10)})
    lon = xr.DataArray(np.arange(100, 140, 10), dims="lon",
                        coords={"lon": np.arange(100, 140, 10)})
    data = xr.DataArray(np.random.randn(5, 4, 4), dims=("time", "lat", "lon"),
                        coords=(time, lat, lon))

    fake_dataset = xr.Dataset({"data_var": data})
    return fake_dataset

@pytest.fixture()
def fake_forecast_dataset():
    forecast_reference_time = xr.DataArray(data=pd.date_range("2021-05-05", periods=1), dims="forecast_reference_time", 
                                            coords={"forecast_reference_time": pd.date_range("2021-05-05", periods=1)})
    # we add one forecast 3 hours ahead to allow testing of infill behavior (via reindex)
    step = xr.DataArray(data=np.append(np.array(np.arange(3600000000000, 18000000000000, 3600000000000),
                                                dtype='timedelta64[ns]'), 3600000000000 *2 + 18000000000000), dims="step",
                        coords={"step": np.append(np.array(np.arange(3600000000000, 18000000000000, 3600000000000),
                                                            dtype='timedelta64[ns]'), 3600000000000 * 2 + 18000000000000)})
    lat = xr.DataArray(np.arange(10, 50, 10), dims="lat",
                        coords={"lat": np.arange(10, 50, 10)})
    lon = xr.DataArray(np.arange(100, 140, 10), dims="lon",
                        coords={"lon": np.arange(100, 140, 10)})
    data = xr.DataArray(np.random.randn(1, 5, 4, 4), dims=("forecast_reference_time", "step", "lat", "lon"),
                        coords=(forecast_reference_time, step, lat, lon))

    fake_dataset = xr.Dataset({"data_var": data})
    return fake_dataset

@pytest.fixture()
def example_zarr_json():
    example_json = "inputs/chirps_example_zarr.json"
    zarr_json = json.loads(open(example_json).read())
    return zarr_json


@pytest.fixture
def manager_class():
    """
    etls.managers.CHIRPSFinal25 child to run tests with
    """
    return CHIRPSFinal25

