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

original_times = np.array([
       '2021-09-16', '2021-09-17',
       '2021-09-18', '2021-09-19',
       '2021-09-20', '2021-09-21',
       '2021-09-22', '2021-09-23',
       '2021-09-24', '2021-09-25',
       '2021-09-26', '2021-09-27',
       '2021-09-28', '2021-09-29',
       '2021-09-30', '2021-10-01',
       '2021-10-02', '2021-10-03',
       '2021-10-04', '2021-10-05',
       '2021-10-06', '2021-10-07',
       '2021-10-08', '2021-10-09',
       '2021-10-10', '2021-10-11',
       '2021-10-12', '2021-10-13',
       '2021-10-14', '2021-10-15',
       '2021-10-16', '2021-10-17',
       '2021-10-18', '2021-10-19',
       '2021-10-20', '2021-10-21',
       '2021-10-22', '2021-10-23',
       '2021-10-24', '2021-10-25',
       '2021-10-26', '2021-10-27',
       '2021-10-28', '2021-10-29',
       '2021-10-30', '2021-10-31',
       '2021-11-01', '2021-11-02',
       '2021-11-03', '2021-11-04',
       '2021-11-05', '2021-11-06',
       '2021-11-07', '2021-11-08',
       '2021-11-09', '2021-11-10',
       '2021-11-11', '2021-11-12',
       '2021-11-13', '2021-11-14',
       '2021-11-15', '2021-11-16',
       '2021-11-17', '2021-11-18',
       '2021-11-19', '2021-11-20',
       '2021-11-21', '2021-11-22',
       '2021-11-23', '2021-11-24',
       '2021-11-25', '2021-11-26',
       '2021-11-27', '2021-11-28',
       '2021-11-29', '2021-11-30',
       '2021-12-01', '2021-12-02',
       '2021-12-03', '2021-12-04',
       '2021-12-05', '2021-12-06',
       '2021-12-07', '2021-12-08',
       '2021-12-09', '2021-12-10',
       '2021-12-11', '2021-12-12',
       '2021-12-13', '2021-12-14',
       '2021-12-15', '2021-12-16',
       '2021-12-17', '2021-12-18',
       '2021-12-19', '2021-12-20',
       '2021-12-21', '2021-12-22',
       '2021-12-23', '2021-12-24',
       '2021-12-25', '2021-12-26',
       '2021-12-27', '2021-12-28',
       '2021-12-29', '2021-12-30',
       '2021-12-31', '2022-01-01',
       '2022-01-02', '2022-01-03',
       '2022-01-04', '2022-01-05',
       '2022-01-06', '2022-01-07',
       '2022-01-08', '2022-01-09',
       '2022-01-10', '2022-01-11',
       '2022-01-12', '2022-01-13',
       '2022-01-14', '2022-01-15',
       '2022-01-16', '2022-01-17',
       '2022-01-18', '2022-01-19',
       '2022-01-20', '2022-01-21',
       '2022-01-22', '2022-01-23',
       '2022-01-24', '2022-01-25',
       '2022-01-26', '2022-01-27',
       '2022-01-28', '2022-01-29',
       '2022-01-30', '2022-01-31'
       ],
      dtype='datetime64[ns]')


complex_update_times = np.array([
        "2021-10-10",
        "2021-10-16",
        "2021-10-17",
        "2021-10-18",
        "2021-10-19",
        "2021-10-20",
        "2021-10-21",
        "2021-10-22",
        "2021-10-23",
        "2021-11-11",
        "2021-12-11",
        "2021-12-25",
        "2021-12-26",
        "2021-12-27",
        "2021-12-28",
        "2021-12-29",
        "2021-12-30",
        "2021-12-31",
        "2022-01-01",
        "2022-01-02",
        "2022-01-03",
        "2022-01-04",
        "2022-01-05",
        "2022-01-14",
        "2022-02-01",
        "2022-02-02",
        "2022-02-03",
        "2022-02-04",
        "2022-02-05",
        "2022-02-06",
        "2022-02-07",
        "2022-02-08",
        "2022-02-09",
        "2022-02-10",
        "2022-02-11",
        "2022-02-12",
        "2022-02-13",
        "2022-02-14",
        "2022-02-15",
        "2022-02-16",
        "2022-02-17",
        "2022-02-18",
        "2022-02-19",
        "2022-02-20",
        "2022-02-21",
        "2022-02-22",
        "2022-02-23",
        "2022-02-24",
        "2022-02-25",
        "2022-02-26",
        "2022-02-27",
        "2022-02-28",
        "2022-03-01",
        "2022-03-02",
        "2022-03-03",
        "2022-03-04",
        "2022-03-05",
        "2022-03-06",
        "2022-03-07",
        "2022-03-08",
    ],
      dtype='datetime64[ns]')


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

    return xr.Dataset({"data_var": data})


@pytest.fixture()
def fake_original_dataset():
    time = xr.DataArray(np.array(original_times), dims="time", coords={"time": np.arange(138)})
    lat = xr.DataArray(np.arange(10, 50, 10), dims="lat",
                        coords={"lat": np.arange(10, 50, 10)})
    lon = xr.DataArray(np.arange(100, 140, 10), dims="lon",
                        coords={"lon": np.arange(100, 140, 10)})
    data = xr.DataArray(np.random.randn(138, 4, 4), dims=("time", "lat", "lon"),
                        coords=(time, lat, lon))

    return xr.Dataset({"data_var": data})


@pytest.fixture()
def fake_complex_update_dataset():
    time = xr.DataArray(np.array(complex_update_times), dims="time", coords={"time": np.arange(60)})
    lat = xr.DataArray(np.arange(10, 50, 10), dims="lat",
                        coords={"lat": np.arange(10, 50, 10)})
    lon = xr.DataArray(np.arange(100, 140, 10), dims="lon",
                        coords={"lon": np.arange(100, 140, 10)})
    data = xr.DataArray(np.random.randn(60, 4, 4), dims=("time", "lat", "lon"),
                        coords=(time, lat, lon))

    return xr.Dataset({"data_var": data})


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

    return xr.Dataset({"data_var": data})


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

