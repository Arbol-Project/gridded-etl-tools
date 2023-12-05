import os
import json

import numpy as np
import xarray as xr

from copy import deepcopy
from gridded_etl_tools.dataset_manager import DatasetManager
from gridded_etl_tools.utils import store
from ..common import get_manager

from unittest.mock import call, Mock, MagicMock


def test_standard_dims(mocker, manager_class: DatasetManager):
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
    assert dm.standard_dims == [
        "forecast_reference_time",
        "step",
        "latitude",
        "longitude",
    ]
    assert dm.time_dim == "forecast_reference_time"
    # Ensemble standard dims
    mocker.patch("gridded_etl_tools.utils.attributes.Attributes.ensemble", return_value=True)
    dm.set_key_dims()
    assert dm.standard_dims == [
        "forecast_reference_time",
        "step",
        "ensemble",
        "latitude",
        "longitude",
    ]
    assert dm.time_dim == "forecast_reference_time"
    # Ensemble standard dims
    mocker.patch("gridded_etl_tools.utils.attributes.Attributes.hindcast", return_value=True)
    dm.set_key_dims()
    assert dm.standard_dims == [
        "hindcast_reference_time",
        "forecast_reference_offset",
        "step",
        "ensemble",
        "latitude",
        "longitude",
    ]
    assert dm.time_dim == "hindcast_reference_time"


def test_export_zarr_json_in_memory(manager_class: DatasetManager, example_zarr_json):
    dm = get_manager(manager_class)
    local_file_path = "output_zarr_json.json"
    json_str = str(example_zarr_json)
    dm.zarr_json_in_memory_to_file(json_str, local_file_path=local_file_path)
    assert os.path.exists(local_file_path)
    os.remove(local_file_path)


def test_preprocess_kerchunk(mocker, manager_class: DatasetManager, example_zarr_json: dict):
    """
    Test that the preprocess_kerchunk method successfully changes the _FillValue attribute of all arrays
    """
    orig_fill_value = json.loads(example_zarr_json["refs"]["latitude/.zarray"])["fill_value"]

    # prepare a dataset manager and preprocess a Zarr JSON
    class MyManagerClass(manager_class):
        missing_value = -8888

    dm = get_manager(MyManagerClass)

    pp_zarr_json = dm.preprocess_kerchunk(example_zarr_json["refs"])
    # populate before/after fill value variables
    modified_fill_value = int(json.loads(pp_zarr_json["latitude/.zarray"])["fill_value"])
    # test that None != -8888
    assert orig_fill_value != modified_fill_value
    assert modified_fill_value == -8888


def test_calculate_update_time_ranges(
    manager_class: DatasetManager,
    fake_original_dataset: xr.Dataset,
    fake_complex_update_dataset: xr.Dataset,
):
    """
    Test that the calculate_date_ranges function correctly prepares insert and append date ranges as anticipated
    """
    # prepare a dataset manager
    dm = get_manager(manager_class)
    dm.set_key_dims()
    datetime_ranges, regions_indices = dm.calculate_update_time_ranges(
        fake_original_dataset, fake_complex_update_dataset
    )
    # Test that 7 distinct updates -- 6 inserts and 1 append -- have been prepared
    assert len(regions_indices) == 7
    # Test that all of the updates are of the expected sizes
    insert_range_sizes = []
    for region in regions_indices:
        index_range = region[1] - region[0]
        insert_range_sizes.append(index_range)
    assert insert_range_sizes == [1, 8, 1, 1, 12, 1, 1]
    # Test that the append is of the expected size
    append_update = datetime_ranges[-1]
    append_size = (append_update[-1] - append_update[0]).astype("timedelta64[D]")
    assert append_size == np.timedelta64(35, "D")

def test_to_zarr(
        mocker,
        manager_class: DatasetManager,
        fake_original_dataset: xr.Dataset
):
    """
    Test that calls to `to_zarr` correctly run three times, updating relevant metadata fields to show a parse is underway.

    Test that metadata fields for date ranges, etc. are only populated to a datset *after* a successful parse
    """
    dm = get_manager(manager_class, store='local')
    dm.set_key_dims()
    dm.update_attributes = ["date range", "update_previous_end_date", "another attribute"]
    # Mock update components
    dataset = deepcopy(fake_original_dataset)
    dataset.attrs.update(**{"date range": ("2000010100", "2020123123"),
                          "update_date_range" : ("202012293", "2020123123"),
                          "update_previous_end_date" : "2020123023",
                          "update_in_progress" : False, 
                          "attribute relevant to updates" : 1,
                          "another attribute" : True})
    update_dataset = deepcopy(fake_original_dataset)
    update_dataset.attrs.update(**{"date range": ("2000010100", "2021010523"),
                                 "update_date_range" : ("2021010123", "2021010523"),
                                 "update_previous_end_date" : "2020123123",
                                 "another attribute" : True})

    mocker.patch("gridded_etl_tools.utils.zarr_methods.xr.core.dataset.Dataset.to_zarr", autospec=True, return_value=None)
    dm.pre_parse_quality_check = Mock()
    dm.store = Mock(has_existing=True, mapper=Mock(refresh=True, return_value=1), dataset=Mock(return_value=dataset), spec=store.Local)
    # Corresponding update dict
    update_dict = {"date range" : ("2000010100", "2021010523"),
                   "update_previous_end_date" : "2020123123",
                   "attribute relevant to updates" : 1,
                   "update_in_progress" : False}
    # Arguments to check
    to_zarr_calls = [call("empty_dataset", dm.store.mapper(refresh=True), append_dim=dm.time_dim),
                     call(dataset, dm.store.mapper, [], {}),
                     call("empty_dataset", dm.store.mapper(), append_dim=dm.time_dim),]
    # Run to_zarr
    dm.to_zarr(update_dataset, dm.store.mapper, append_dim=dm.time_dim)
    # Assertions
    # dm.pre_parse_quality_check.assert_called_once_with()
    assert xr.core.dataset.Dataset.to_zarr.call_count == 3
    import ipdb; ipdb.set_trace(context=4)
    xr.core.dataset.Dataset.to_zarr.assert_has_calls(to_zarr_calls, any_order=True)  # TODO turn any order to True
    # dm.store.dataset.attrs.update.assert_called_once_with(**update_dict)
    dm.to_zarr.empty_dataset.attrs.update.assert_called_once_with(**update_dict)
    assert dm.to_zarr.update_dict == update_dict