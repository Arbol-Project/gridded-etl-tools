import pandas as pd
import numpy as np
import xarray as xr
import pytest
import os
import json

from gridded_etl_tools.dataset_manager import DatasetManager
from .common import get_manager
from .conftest import fake_dataset, fake_forecast_dataset, example_zarr_json

def mocked_jz_to_file(zarr_json: dict):
    zarr_json['refs']['precip/.zattrs']

def mocked_missing_value_indicator():
    return -8888


def test_standard_dims(manager_class, mocker):
    """
    Test that standard dimensions are correctly instantiated for regular, forecast, and ensemble datasets
    """
    dm = get_manager(manager_class)
    # Test normal standard dims
    dm.set_key_dims()
    assert dm.standard_dims == ["time", "latitude", "longitude"]
    assert dm.time_dim == "time"
    # Forecast standard dims
    mocker.patch("gridded_etl_tools.utils.attributes.Attributes.forecast", return_value=True)
    dm.set_key_dims()
    assert dm.standard_dims == ["forecast_reference_time", "step", "latitude", "longitude"]
    assert dm.time_dim == "forecast_reference_time"
    # Ensemble standard dims
    mocker.patch("gridded_etl_tools.utils.attributes.Attributes.ensemble", return_value=True)
    dm.set_key_dims()
    assert dm.standard_dims == ["forecast_reference_time", "step", "ensemble", "latitude", "longitude"]
    assert dm.time_dim == "forecast_reference_time"


def test_export_zarr_json_in_memory(manager_class, mocker):
    dm = get_manager(manager_class)
    local_file_path = "output_zarr_json.json"
    json_str = str(example_zarr_json)
    dm.zarr_json_in_memory_to_file(json_str, local_file_path=local_file_path)
    assert os.path.exists(local_file_path)
    os.remove(local_file_path)


def test_preprocess_kerchunk(manager_class, mocker, example_zarr_json):
    """
    Test that the preprocess_kerchunk method successfully changes the _FillValue attribute of all arrays
    """
    orig_fill_value = json.loads(example_zarr_json["refs"]["latitude/.zarray"])["fill_value"]
    # prepare a dataset manager and preprocess a Zarr JSON
    dm = get_manager(manager_class)
    mocker.patch("examples.managers.chirps.CHIRPSFinal25.missing_value_indicator", return_value=-8888)
    pp_zarr_json = dm.preprocess_kerchunk(example_zarr_json["refs"])
    # populate before/after fill value variables
    modified_fill_value = json.loads(pp_zarr_json["latitude/.zarray"])["fill_value"]
    # test that None != -8888
    assert orig_fill_value != modified_fill_value

