import datetime
import json
import pathlib

import numpy as np
import pytest
import xarray as xr

from gridded_etl_tools import dataset_manager

HERE = pathlib.Path(__file__).parent
INPUTS = HERE / "inputs"


@pytest.fixture
def example_zarr_json():
    example_json = INPUTS / "chirps_example_zarr.json"
    zarr_json = json.loads(open(example_json).read())
    return zarr_json


@pytest.fixture
def fake_original_dataset():
    time = xr.DataArray(np.array(original_times), dims="time", coords={"time": np.arange(138)})
    latitude = xr.DataArray(np.arange(10, 50, 10), dims="latitude", coords={"latitude": np.arange(10, 50, 10)})
    longitude = xr.DataArray(np.arange(100, 140, 10), dims="longitude", coords={"longitude": np.arange(100, 140, 10)})
    data = xr.DataArray(
        np.random.randn(138, 4, 4),
        dims=("time", "latitude", "longitude"),
        coords=(time, latitude, longitude),
    )

    ds = xr.Dataset({"data": data})
    ds["data"] = ds["data"].astype("<f4")
    return ds


@pytest.fixture
def fake_large_dataset():
    time = xr.DataArray(np.array(original_times), dims="time", coords={"time": np.arange(138)})
    latitude = xr.DataArray(np.arange(1, 1001), dims="latitude", coords={"latitude": np.arange(1, 1001)})
    longitude = xr.DataArray(np.arange(1, 1001), dims="longitude", coords={"longitude": np.arange(1, 1001)})
    data = xr.DataArray(
        np.random.randn(138, 1000, 1000),
        dims=("time", "latitude", "longitude"),
        coords=(time, latitude, longitude),
    )

    ds = xr.Dataset({"data": data})
    ds["data"] = ds["data"].astype("<f4")
    return ds


@pytest.fixture
def forecast_dataset():
    time = xr.DataArray(
        np.array(original_times), dims="forecast_reference_time", coords={"forecast_reference_time": np.arange(138)}
    )
    step = xr.DataArray(np.arange(2, 10, 2))
    latitude = xr.DataArray(np.arange(10, 50, 10), dims="latitude", coords={"latitude": np.arange(10, 50, 10)})
    longitude = xr.DataArray(np.arange(100, 140, 10), dims="longitude", coords={"longitude": np.arange(100, 140, 10)})
    data = xr.DataArray(
        np.random.randn(138, 4, 4, 4),
        dims=("forecast_reference_time", "latitude", "longitude", "step"),
        coords=(time, latitude, longitude, step),
    )

    ds = xr.Dataset({"data": data})
    ds["data"] = ds["data"].astype("<f4")
    return ds


@pytest.fixture
def hindcast_dataset():
    hrt = xr.DataArray(
        np.array([np.datetime64("2021-10-16T00:00:00.000000000")]),
        dims="hindcast_reference_time",
        coords={"hindcast_reference_time": np.arange(1)},
    )
    step = xr.DataArray([np.timedelta64(3600000000000, "[ns]")], dims="step", coords={"step": np.arange(1)})
    ensembles = xr.DataArray([np.array(1)], dims="ensemble", coords={"ensemble": np.arange(1)})
    fro = xr.DataArray([np.array(np.timedelta64(1, "[D]"))], dims="ensemble", coords={"ensemble": np.arange(1)})
    latitude = xr.DataArray(np.arange(10, 50, 10), dims="latitude", coords={"latitude": np.arange(10, 50, 10)})
    longitude = xr.DataArray(np.arange(100, 140, 10), dims="longitude", coords={"longitude": np.arange(100, 140, 10)})
    data = xr.DataArray(
        np.random.randn(1, 1, 1, 1, 4, 4),
        dims=(
            "hindcast_reference_time",
            "step",
            "ensemble",
            "forecast_reference_offset",
            "latitude",
            "longitude",
        ),
        coords=(hrt, step, ensembles, fro, latitude, longitude),
    )

    ds = xr.Dataset({"data": data})
    ds["data"] = ds["data"].astype("<f4")
    return ds


@pytest.fixture
def fake_complex_update_dataset():
    time = xr.DataArray(np.array(complex_update_times), dims="time", coords={"time": np.arange(60)})
    latitude = xr.DataArray(np.arange(10, 50, 10), dims="latitude", coords={"latitude": np.arange(10, 50, 10)})
    longitude = xr.DataArray(np.arange(100, 140, 10), dims="longitude", coords={"longitude": np.arange(100, 140, 10)})
    data = xr.DataArray(
        np.random.randn(60, 4, 4), dims=("time", "latitude", "longitude"), coords=(time, latitude, longitude)
    )

    ds = xr.Dataset({"data": data})
    ds.chunk(None)
    ds["data"] = ds["data"].astype("<f4")
    return ds


@pytest.fixture
def single_time_instant_dataset():
    return _single_time_instant_dataset(original_times[:1])


def _single_time_instant_dataset(times):
    time = xr.DataArray(np.array(times), dims="time", coords={"time": np.arange(1)})
    latitude = xr.DataArray(np.arange(10, 50, 10), dims="latitude", coords={"latitude": np.arange(10, 50, 10)})
    longitude = xr.DataArray(np.arange(100, 140, 10), dims="longitude", coords={"longitude": np.arange(100, 140, 10)})
    data = xr.DataArray(
        np.random.randn(1, 4, 4), dims=("time", "latitude", "longitude"), coords=(time, latitude, longitude)
    )

    ds = xr.Dataset({"data": data})
    ds["data"] = ds["data"].astype("<f4")
    return ds


@pytest.fixture
def base_class():
    return DummyManagerBase


@pytest.fixture
def manager_class():
    return DummyManager


def unimplemented(*args, **kwargs):  # pragma NO COVER
    raise NotImplementedError


def noop(*args, **kwargs):
    """Do nothing"""


class DummyManagerBase(dataset_manager.DatasetManager):
    prepare_input_files = noop

    unit_of_measurement = "parsecs"
    requested_zarr_chunks = {}
    encryption_key = None
    fill_value = ""

    def __init__(self, requested_dask_chunks=None, requested_zarr_chunks=None, set_key_dims=True, *args, **kwargs):
        if requested_dask_chunks is None:
            requested_dask_chunks = {}

        if requested_zarr_chunks is None:
            requested_zarr_chunks = {}

        self._static_metadata = kwargs.pop("static_metadata", {})
        super().__init__(requested_dask_chunks, requested_zarr_chunks, *args, **kwargs)
        if set_key_dims:
            self.set_key_dims()

    data_var = "data"

    @property
    def data_var_dtype(self):
        return "<f4"

    def extract(self, date_range=None):
        return super().extract(date_range=date_range)

    @property
    def dataset_start_date(self):
        return datetime.datetime(1975, 7, 7, 0, 0, 0)

    @property
    def static_metadata(self):
        return self._static_metadata


class DummyManager(DummyManagerBase):
    collection_name = "Vintage Guitars"
    concat_dimensions = ["z", "zz"]
    dataset_name = "DummyManager"
    identical_dimensions = ["x", "y"]
    protocol = "handshake"
    time_resolution = dataset_manager.DatasetManager.SPAN_DAILY
    final_lag_in_days = 3
    expected_nan_frequency = 0.2


# Set up overcomplicated mro for testing get_subclass(es)
class John(DummyManager):
    dataset_name = "John"


class Paul(DummyManager):
    dataset_name = "Paul"


class George(John, Paul):
    dataset_name = "George"


class Ringo(George):
    dataset_name = "Ringo"
    time_resolution = dataset_manager.DatasetManager.SPAN_HOURLY


class RingoDaily(DummyManager):
    dataset_name = "Ringo"
    time_resolution = dataset_manager.DatasetManager.SPAN_DAILY


original_times = np.array(
    [
        "2021-09-16",
        "2021-09-17",
        "2021-09-18",
        "2021-09-19",
        "2021-09-20",
        "2021-09-21",
        "2021-09-22",
        "2021-09-23",
        "2021-09-24",
        "2021-09-25",
        "2021-09-26",
        "2021-09-27",
        "2021-09-28",
        "2021-09-29",
        "2021-09-30",
        "2021-10-01",
        "2021-10-02",
        "2021-10-03",
        "2021-10-04",
        "2021-10-05",
        "2021-10-06",
        "2021-10-07",
        "2021-10-08",
        "2021-10-09",
        "2021-10-10",
        "2021-10-11",
        "2021-10-12",
        "2021-10-13",
        "2021-10-14",
        "2021-10-15",
        "2021-10-16",
        "2021-10-17",
        "2021-10-18",
        "2021-10-19",
        "2021-10-20",
        "2021-10-21",
        "2021-10-22",
        "2021-10-23",
        "2021-10-24",
        "2021-10-25",
        "2021-10-26",
        "2021-10-27",
        "2021-10-28",
        "2021-10-29",
        "2021-10-30",
        "2021-10-31",
        "2021-11-01",
        "2021-11-02",
        "2021-11-03",
        "2021-11-04",
        "2021-11-05",
        "2021-11-06",
        "2021-11-07",
        "2021-11-08",
        "2021-11-09",
        "2021-11-10",
        "2021-11-11",
        "2021-11-12",
        "2021-11-13",
        "2021-11-14",
        "2021-11-15",
        "2021-11-16",
        "2021-11-17",
        "2021-11-18",
        "2021-11-19",
        "2021-11-20",
        "2021-11-21",
        "2021-11-22",
        "2021-11-23",
        "2021-11-24",
        "2021-11-25",
        "2021-11-26",
        "2021-11-27",
        "2021-11-28",
        "2021-11-29",
        "2021-11-30",
        "2021-12-01",
        "2021-12-02",
        "2021-12-03",
        "2021-12-04",
        "2021-12-05",
        "2021-12-06",
        "2021-12-07",
        "2021-12-08",
        "2021-12-09",
        "2021-12-10",
        "2021-12-11",
        "2021-12-12",
        "2021-12-13",
        "2021-12-14",
        "2021-12-15",
        "2021-12-16",
        "2021-12-17",
        "2021-12-18",
        "2021-12-19",
        "2021-12-20",
        "2021-12-21",
        "2021-12-22",
        "2021-12-23",
        "2021-12-24",
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
        "2022-01-06",
        "2022-01-07",
        "2022-01-08",
        "2022-01-09",
        "2022-01-10",
        "2022-01-11",
        "2022-01-12",
        "2022-01-13",
        "2022-01-14",
        "2022-01-15",
        "2022-01-16",
        "2022-01-17",
        "2022-01-18",
        "2022-01-19",
        "2022-01-20",
        "2022-01-21",
        "2022-01-22",
        "2022-01-23",
        "2022-01-24",
        "2022-01-25",
        "2022-01-26",
        "2022-01-27",
        "2022-01-28",
        "2022-01-29",
        "2022-01-30",
        "2022-01-31",
    ],
    dtype="datetime64[ns]",
)


complex_update_times = np.array(
    [
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
    dtype="datetime64[ns]",
)


# V2 and V3 metadata fixtures
@pytest.fixture
def v3_zarr_json():
    """Simplified v3 zarr.json metadata for testing."""
    return {
        "attributes": {
            "Conventions": "CF-1.0",
            "title": "Test Dataset",
            "coordinate reference system": "EPSG:4326",
            "spatial resolution": 0.5,
            "temporal resolution": "daily",
            "update cadence": "daily",
            "publisher": "Test Publisher",
            "unit of measurement": "test_unit",
            "missing value": -9.99e36,
            "created": "2023-01-01T12:00:00.00000Z",
            "update_previous_end_date": "2023-06-30",
            "updated": "2023-07-15T12:00:00.00000Z",
            "date range": ["2023-01-01", "2023-07-15"],
            "update_date_range": ["2023-07-01", "2023-07-15"],
            "bbox": [-180.0, -90.0, 180.0, 90.0],
            "update_is_append_only": True,
        },
        "zarr_format": 3,
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {
                "longitude": {
                    "shape": [720],
                    "data_type": "float32",
                    "chunk_grid": {"name": "regular", "configuration": {"chunks": [720]}},
                    "zarr_format": 3,
                    "node_type": "array",
                },
                "latitude": {
                    "shape": [360],
                    "data_type": "float32",
                    "chunk_grid": {"name": "regular", "configuration": {"chunks": [360]}},
                    "zarr_format": 3,
                    "node_type": "array",
                },
                "time": {
                    "shape": [200],
                    "data_type": "float64",
                    "chunk_grid": {"name": "regular", "configuration": {"chunks": [100]}},
                    "zarr_format": 3,
                    "node_type": "array",
                },
                "data": {
                    "shape": [200, 360, 720],
                    "data_type": "float32",
                    "chunk_grid": {"name": "regular", "configuration": {"chunks": [50, 90, 90]}},
                    "zarr_format": 3,
                    "node_type": "array",
                },
            },
        },
    }


@pytest.fixture
def v2_zattrs():
    """Simplified v2 .zattrs metadata for testing."""
    return {
        "Conventions": "CF-1.0",
        "title": "Test Dataset",
        "coordinate reference system": "EPSG:4326",
        "spatial resolution": 0.5,
        "temporal resolution": "daily",
        "update cadence": "daily",
        "publisher": "Test Publisher",
        "unit of measurement": "test_unit",
        "missing value": -9.99e36,
        "created": "2023-01-01T12:00:00.00000Z",
        "update_previous_end_date": "2023-06-30",
        "updated": "2023-07-15T12:00:00.00000Z",
        "date range": ["2023-01-01", "2023-07-15"],
        "update_date_range": ["2023-07-01", "2023-07-15"],
        "bbox": [-180.0, -90.0, 180.0, 90.0],
        "update_is_append_only": True,
    }


@pytest.fixture
def v2_zmetadata():
    """Simplified v2 .zmetadata metadata for testing."""
    return {
        "metadata": {
            ".zattrs": {
                "Conventions": "CF-1.0",
                "title": "Test Dataset",
                "coordinate reference system": "EPSG:4326",
                "spatial resolution": 0.5,
                "temporal resolution": "daily",
                "update cadence": "daily",
                "publisher": "Test Publisher",
                "unit of measurement": "test_unit",
                "missing value": -9.99e36,
                "created": "2023-01-01T12:00:00.00000Z",
                "update_previous_end_date": "2023-06-30",
                "updated": "2023-07-15T12:00:00.00000Z",
                "date range": ["2023-01-01", "2023-07-15"],
                "update_date_range": ["2023-07-01", "2023-07-15"],
                "bbox": [-180.0, -90.0, 180.0, 90.0],
                "update_is_append_only": True,
            },
            ".zgroup": {"zarr_format": 2},
            "time/.zarray": {
                "chunks": [100],
                "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
                "dtype": "<f8",
                "fill_value": -9.99e36,
                "filters": None,
                "order": "C",
                "shape": [200],
                "zarr_format": 2,
            },
            "data/.zarray": {
                "chunks": [50, 90, 90],
                "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
                "dtype": "<f4",
                "fill_value": -9.99e36,
                "filters": None,
                "order": "C",
                "shape": [200, 360, 720],
                "zarr_format": 2,
            },
        },
        "zarr_consolidated_format": 1,
    }


@pytest.fixture
def updated_attrs():
    """Fixture with sample updated attribute values for testing metadata update functions."""
    return {
        "update_previous_end_date": "2023-07-15",
        "updated": "2023-07-31T12:00:00.00000Z",
        "update_date_range": ["2023-07-16", "2023-07-31"],
        "date range": ["2023-01-01", "2023-07-31"],
    }


@pytest.fixture
def updated_arrays():
    """Fixture with sample updated array information for testing array update functions."""
    return {"time": {"chunks": [100], "shape": [220]}, "data": {"chunks": [50, 90, 90], "shape": [220, 360, 720]}}


@pytest.fixture
def v2_time_zarray():
    """V2 .zarray file content for time dimension."""
    return {
        "chunks": [100],
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
        "dtype": "<f8",
        "fill_value": -9.99e36,
        "filters": None,
        "order": "C",
        "shape": [200],
        "zarr_format": 2,
    }


@pytest.fixture
def v2_data_var_zarray():
    """V2 .zarray file content for data variable."""
    return {
        "chunks": [50, 90, 90],
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
        "dtype": "<f4",
        "fill_value": -9.99e36,
        "filters": None,
        "order": "C",
        "shape": [200, 360, 720],
        "zarr_format": 2,
    }
