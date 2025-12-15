from collections import UserDict
import copy
import functools
import operator
import pathlib
import re
import datetime

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from gridded_etl_tools.utils import publish, store
from gridded_etl_tools.utils.publish import _is_infish, calculate_time_dim_chunks, ZarrOutputError
from gridded_etl_tools.utils.errors import NanFrequencyMismatchError


def generate_partial_nan_array(shape: tuple[float], percent_nan: float):
    # Calculate the number of NaNs and floats
    total_elements = np.prod(shape)
    num_nans = int(total_elements * percent_nan)
    num_floats = total_elements - num_nans

    # Generate them
    random_floats = np.random.random(num_floats)
    nans = np.full(num_nans, np.nan)

    # Combine them and shuffle them around
    combined_array = np.concatenate((random_floats, nans))
    np.random.shuffle(combined_array)

    # Reshape the array to the desired shape
    final_array = combined_array.reshape(shape)

    return final_array


class fake_vmem(dict):
    """
    Fake a vmem object with 16gb total memory using a dict
    """

    def __init__(self):
        self.total = 2**34


class TestPublish:
    @staticmethod
    def test_parse_first_time(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=False)
        dm.update_zarr = mock.Mock()
        dm.write_initial_zarr = mock.Mock()
        publish_dataset = mock.Mock()

        dm.parse(publish_dataset)

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_not_called()
        dm.write_initial_zarr.assert_called_once_with(publish_dataset)

        Client.assert_called_once_with(cluster)

    @staticmethod
    def test_parse_update(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dask_configuration = mock.Mock()

        # Mock that the store has a v2 Zarr
        dm.store = mock.Mock(
            spec=store.StoreInterface, has_existing=True, path="/home/vladimir_putin/", has_v3_metadata=False
        )

        dm.update_zarr = mock.Mock()
        dm.write_initial_zarr = mock.Mock()
        publish_dataset = mock.Mock()

        dm.parse(publish_dataset)

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_called_once_with(publish_dataset)
        dm.write_initial_zarr.assert_not_called()

        Client.assert_called_once_with(cluster)

    @staticmethod
    def test_parse_rebuild(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=True, allow_overwrite=True)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        dm.update_zarr = mock.Mock()
        dm.write_initial_zarr = mock.Mock()
        publish_dataset = mock.Mock()

        dm.parse(publish_dataset)

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_not_called()
        dm.write_initial_zarr.assert_called_once_with(publish_dataset)

        Client.assert_called_once_with(cluster)

    @staticmethod
    def test_parse_rebuild_but_overwrite_not_allowed(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=True, allow_overwrite=False)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        dm.update_zarr = mock.Mock()
        dm.write_initial_zarr = mock.Mock()
        publish_dataset = mock.Mock()

        with pytest.raises(RuntimeError):
            dm.parse(publish_dataset)

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_not_called()
        dm.write_initial_zarr.assert_not_called()

        Client.assert_called_once_with(cluster)

    @staticmethod
    def test_parse_update_ctrl_c(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dask_configuration = mock.Mock()

        # Mock that the store has a v2 Zarr
        dm.store = mock.Mock(
            spec=store.StoreInterface, has_existing=True, path="/home/vladimir_putin/", has_v3_metadata=False
        )

        dm.update_zarr = mock.Mock(side_effect=KeyboardInterrupt)
        dm.write_initial_zarr = mock.Mock()
        publish_dataset = mock.Mock()

        dm.parse(publish_dataset)

        LocalCluster.assert_called_once_with(
            processes=False,
            dashboard_address="127.0.0.1:8787",
            protocol="inproc://",
            threads_per_worker=2,
            n_workers=1,
        )

        dm.dask_configuration.assert_called_once_with()
        dm.update_zarr.assert_called_once_with(publish_dataset)
        dm.write_initial_zarr.assert_not_called()

        Client.assert_called_once_with(cluster)

    @staticmethod
    def test_parse_zarr_version_match(manager_class, mocker):
        mocker.patch("gridded_etl_tools.utils.publish.Client")
        dm = manager_class(rebuild_requested=False)
        dm.dask_configuration = mock.Mock()
        dm.update_zarr = mock.Mock()

        # Mock that the store has a v3 Zarr
        dm.store = mock.Mock(
            spec=store.StoreInterface, has_existing=True, path="/home/vladimir_putin/", has_v3_metadata=True
        )

        # The default of dm.output_zarr3 should be false, so this should fail
        with pytest.raises(RuntimeError, match="Existing data is Zarr v3, but output_zarr3 is not set."):
            dm.parse(mock.Mock())

        # With dm.output_zarr3 set, it should pass instead
        dm.output_zarr3 = True
        dm.parse(mock.Mock())
        dm.update_zarr.assert_called_once()

        # Mock that the store has a v2 Zarr
        dm.store.has_v3_metadata = False

        # dm.output_zarr3 unset should pass
        dm.output_zarr3 = False
        dm.parse(mock.Mock())
        assert dm.update_zarr.call_count == 2

        # dm.output_zarr3 set should fail
        with pytest.raises(RuntimeError, match="Existing data is not Zarr v3, but output_zarr3 is set."):
            dm.output_zarr3 = True
            dm.parse(mock.Mock())

    @staticmethod
    def test_publish_metadata(manager_class):
        dm = manager_class()
        dm.metadata = {"hi": "mom!"}
        dm.time_dims = ["what", "is", "time", "really?"]
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.populate_metadata = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.create_root_stac_catalog = mock.Mock()
        dm.create_stac_collection = mock.Mock()
        dm.create_stac_item = mock.Mock()
        current_zarr = dm.store.dataset.return_value

        dm.publish_metadata()

        dm.populate_metadata.assert_not_called()
        dm.set_key_dims.assert_not_called()
        dm.create_root_stac_catalog.assert_called_once_with()
        dm.create_stac_collection.assert_called_once_with(current_zarr)
        dm.create_stac_item.assert_called_once_with(current_zarr)

    @staticmethod
    def test_publish_metadata_no_current_zarr(manager_class):
        dm = manager_class()
        dm.metadata = {"hi": "mom!"}
        dm.time_dims = ["what", "is", "time", "really?"]
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.populate_metadata = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.create_root_stac_catalog = mock.Mock()
        dm.create_stac_collection = mock.Mock()
        dm.create_stac_item = mock.Mock()
        dm.store.dataset.return_value = None

        with pytest.raises(RuntimeError):
            dm.publish_metadata()

        dm.populate_metadata.assert_not_called()
        dm.set_key_dims.assert_not_called()
        dm.create_root_stac_catalog.assert_not_called()
        dm.create_stac_collection.assert_not_called()
        dm.create_stac_item.assert_not_called()

    @staticmethod
    def test_publish_metadata_populate_metadata(manager_class):
        dm = manager_class()
        dm.time_dims = ["what", "is", "time", "really?"]
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.populate_metadata = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.create_root_stac_catalog = mock.Mock()
        dm.create_stac_collection = mock.Mock()
        dm.create_stac_item = mock.Mock()
        current_zarr = dm.store.dataset.return_value

        dm.publish_metadata()

        dm.populate_metadata.assert_called_once_with()
        dm.set_key_dims.assert_not_called()
        dm.create_root_stac_catalog.assert_called_once_with()
        dm.create_stac_collection.assert_called_once_with(current_zarr)
        dm.create_stac_item.assert_called_once_with(current_zarr)

    @staticmethod
    def test_publish_metadata_set_key_dims(manager_class):
        dm = manager_class()
        dm.metadata = {"hi": "mom!"}
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.populate_metadata = mock.Mock()
        dm.set_key_dims = mock.Mock()
        dm.create_root_stac_catalog = mock.Mock()
        dm.create_stac_collection = mock.Mock()
        dm.create_stac_item = mock.Mock()
        current_zarr = dm.store.dataset.return_value

        dm.publish_metadata()

        dm.populate_metadata.assert_not_called()
        dm.set_key_dims.assert_called_once_with()
        dm.create_root_stac_catalog.assert_called_once_with()
        dm.create_stac_collection.assert_called_once_with(current_zarr)
        dm.create_stac_item.assert_called_once_with(current_zarr)

    @staticmethod
    def test_to_zarr_dry_run(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.dry_run = True

        dataset = mock.Mock()
        dm.to_zarr(dataset, "foo", bar="baz")

        dataset.to_zarr.assert_not_called()
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only.assert_not_called()

    @staticmethod
    def test_to_zarr(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)

        dataset = mock.Mock()
        dataset.get.return_value = "is it?"
        dm.to_zarr(dataset, "foo", bar="baz")

        # The behavior when DM is default and to_zarr is default is Zarr format is explicitly set to 2
        dataset.to_zarr.assert_called_once_with("foo", zarr_format=2, bar="baz")
        dataset.get.assert_called_once_with("update_is_append_only")
        dm.pre_parse_quality_check.assert_called_once_with(dataset)

        # Metadata v2 function must be called
        dm.store.write_metadata_only_v2.assert_has_calls(
            [
                mock.call(
                    update_attrs={
                        "update_in_progress": True,
                        "update_is_append_only": "is it?",
                        "initial_parse": False,
                    }
                ),
                mock.call(update_attrs={"update_in_progress": False}),
            ]
        )

        # Check that DatasetManager.output_zarr3 is applied
        dm.output_zarr3 = True
        dm.to_zarr(dataset)
        dataset.to_zarr.assert_called_with(zarr_format=3)
        dm.store.write_metadata_only.assert_called()

        # Check that setting zarr_format causes ValueError
        with pytest.raises(ValueError):
            dm.to_zarr(dataset, zarr_format=3)

        # Test what happens when an unknown exception happens during writing
        dataset.reset_mock()
        dataset.to_zarr.side_effect = RuntimeError("Nuclear meltdown")
        with pytest.raises(ZarrOutputError):
            dm.to_zarr(dataset)
        # Metadata must be reset even when an error happens during writing
        dm.store.write_metadata_only_v2.assert_has_calls(
            [
                mock.call(
                    update_attrs={
                        "update_in_progress": True,
                        "update_is_append_only": "is it?",
                        "initial_parse": False,
                    }
                ),
                mock.call(update_attrs={"update_in_progress": False}),
            ]
        )

        # Test different combinations of storage options kwargs
        dataset.to_zarr = mock.Mock()
        dm.store = mock.Mock(spec=store.S3)
        dm.store.endpoint_url = "https://waluigi.world"
        dm.to_zarr(dataset)
        dataset.to_zarr.assert_called_with(zarr_format=3, storage_options={"endpoint_url": "https://waluigi.world"})
        dm.to_zarr(dataset, storage_options={"endpoint_url": "https://waluigi.universe"})
        dataset.to_zarr.assert_called_with(zarr_format=3, storage_options={"endpoint_url": "https://waluigi.universe"})
        dm.to_zarr(dataset, storage_options={"SADDLE UP": "SALLY FORTH"})
        dataset.to_zarr.assert_called_with(
            zarr_format=3, storage_options={"SADDLE UP": "SALLY FORTH", "endpoint_url": "https://waluigi.world"}
        )

    @staticmethod
    def test_to_zarr_initial(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=False)

        dataset = mock.Mock()
        dataset.get.return_value = "is it?"
        dm.to_zarr(dataset, "foo", bar="baz")

        # Zarr format is explicitly set to 2.0
        dataset.to_zarr.assert_called_once_with("foo", bar="baz", zarr_format=2)
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only_v2.assert_has_calls([mock.call(update_attrs={"update_in_progress": False})])

    @staticmethod
    def test_to_zarr_integration(manager_class, fake_original_dataset, tmpdir):
        """
        Integration test that calls to `to_zarr` correctly run three times, updating relevant metadata fields to show a
        parse is underway.

        Test that metadata fields for date ranges, etc. are only populated to a datset *after* a successful parse
        """
        dm = manager_class()
        dm.update_attributes = ["date range", "update_previous_end_date", "another attribute"]
        pre_update_dict = {
            "date range": ["2000010100", "2020123123"],
            "update_date_range": ["202012293", "2020123123"],
            "update_previous_end_date": "2020123023",
            "update_in_progress": False,
            "attribute relevant to updates": 1,
            "another attribute": True,
        }
        post_update_dict = {
            "date range": ["2000010100", "2021010523"],
            "update_previous_end_date": "2020123123",
            "update_in_progress": False,
            "another attribute": True,
            "initial_parse": False,
        }

        # Mock datasets
        dataset = copy.deepcopy(fake_original_dataset)
        dataset.attrs.update(**pre_update_dict)
        dm.custom_output_path = pathlib.Path(tmpdir)
        dataset.to_zarr(dm.store.path, zarr_format=2)  # write out local file to test updates on

        # Mock functions
        dm.pre_parse_quality_check = mock.Mock()

        # Tests
        for key in pre_update_dict.keys():
            assert dm.store.dataset().attrs[key] == pre_update_dict[key]

        dataset.attrs.update(**post_update_dict)
        dm.to_zarr(dataset, store=dm.store.path, append_dim=dm.time_dim)

        for key in post_update_dict.keys():
            assert dm.store.dataset().attrs[key] == post_update_dict[key]

        dm.pre_parse_quality_check.assert_called_once_with(dataset)

    @staticmethod
    def test_to_zarr_integration_initial(manager_class, fake_original_dataset, tmpdir):
        """
        Integration test that calls to `to_zarr` correctly run three times, updating relevant metadata fields to show a
        parse is underway.

        Test that metadata fields for date ranges, etc. are only populated to a datset *after* a successful parse
        """
        dm = manager_class()
        dm.update_attributes = ["date range", "update_previous_end_date", "another attribute"]
        pre_update_dict = {
            "date range": ["2000010100", "2020123123"],
            "update_in_progress": False,
            "attribute relevant to updates": 1,
            "another attribute": True,
            "initial_parse": True,
        }
        post_update_dict = {
            "date range": ["2000010100", "2021010523"],
            "update_previous_end_date": "2020123123",
            "update_in_progress": False,
            "another attribute": True,
            "initial_parse": False,
        }

        # Mock datasets
        dataset = copy.deepcopy(fake_original_dataset)
        dataset.attrs.update(**pre_update_dict)
        dm.custom_output_path = pathlib.Path(tmpdir)
        dataset.to_zarr(store=dm.store.path, zarr_format=2)  # write out local file to test updates on

        # Mock functions
        dm.pre_parse_quality_check = mock.Mock()

        # Tests
        for key in pre_update_dict.keys():
            assert dm.store.dataset().attrs[key] == pre_update_dict[key]

        dataset.attrs.update(**post_update_dict)
        dm.to_zarr(dataset, store=dm.store.path, append_dim=dm.time_dim)

        for key in post_update_dict.keys():
            assert dm.store.dataset().attrs[key] == post_update_dict[key]

        dm.pre_parse_quality_check.assert_called_once_with(dataset)

    @staticmethod
    def test_dask_configuration(manager_class, mocker):
        dask_config = {}

        def dask_config_set(config):
            dask_config.update(config)

        dask = mocker.patch("gridded_etl_tools.utils.publish.dask")
        dask.config.set = dask_config_set

        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)

        dm.dask_configuration()

        assert dask_config == {
            "distributed.scheduler.worker-saturation": dm.dask_scheduler_worker_saturation,
            "distributed.scheduler.worker-ttl": None,
            "distributed.worker.memory.target": dm.dask_worker_mem_target,
            "distributed.worker.memory.spill": dm.dask_worker_mem_spill,
            "distributed.worker.memory.pause": dm.dask_worker_mem_pause,
            "distributed.worker.memory.terminate": dm.dask_worker_mem_terminate,
        }

    @staticmethod
    def test_write_initial_zarr(manager_class):

        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.to_zarr = mock.Mock()
        publish_dataset = mock.Mock()
        publish_dataset.chunk.return_value = publish_dataset_rechunked = mock.Mock()

        dm.store.path = mock.Mock()

        dm.write_initial_zarr(publish_dataset)

        # dm.store.path.assert_called_once_with()
        dm.to_zarr.assert_called_once_with(publish_dataset_rechunked, store=dm.store.path, mode="w")

    @staticmethod
    def test_prepare_update_times(manager_class, fake_original_dataset, fake_complex_update_dataset):
        dm = manager_class()
        insert_times, update_times = dm.prepare_update_times(fake_original_dataset, fake_complex_update_dataset)
        assert insert_times == [
            np.datetime64("2021-10-10T00:00:00.000000000"),
        ] + list(
            np.arange(
                np.datetime64("2021-10-16T00:00:00.000000000"),
                np.datetime64("2021-10-24T00:00:00.000000000"),
                np.timedelta64(1, "[D]"),
            )
        ) + [
            np.datetime64("2021-11-11T00:00:00.000000000"),
            np.datetime64("2021-12-11T00:00:00.000000000"),
        ] + list(
            np.arange(
                np.datetime64("2021-12-25T00:00:00.000000000"),
                np.datetime64("2022-01-06T00:00:00.000000000"),
                np.timedelta64(1, "[D]"),
            )
        ) + [
            np.datetime64("2022-01-14T00:00:00.000000000"),
        ]
        assert update_times == list(
            np.arange(
                np.datetime64("2022-02-01T00:00:00.000000000"),
                np.datetime64("2022-03-09T00:00:00.000000000"),
                np.timedelta64(1, "[D]"),
            )
        )

    @staticmethod
    def test_prepare_update_times_no_time_dimension(manager_class, fake_original_dataset, fake_complex_update_dataset):
        update_dataset = fake_complex_update_dataset.sel(time=[np.datetime64("2021-10-10T00:00:00.000000000")])
        update_dataset = update_dataset.squeeze()
        assert "time" not in update_dataset.dims
        dm = manager_class()
        insert_times, update_times = dm.prepare_update_times(fake_original_dataset, update_dataset)
        assert insert_times == [
            np.datetime64("2021-10-10T00:00:00.000000000"),
        ]
        assert update_times == []

    # TODO
    @staticmethod
    def test_update_zarr(manager_class, fake_original_dataset):
        publish_dataset = object()
        dm = manager_class()
        insert_times, append_times = [], [object()]
        dm.prepare_update_times = mock.Mock(return_value=(insert_times, append_times))
        dm.update_quality_check = mock.Mock()
        dm.insert_into_dataset = mock.Mock()
        dm.append_to_dataset = mock.Mock()

        dm.update_zarr(publish_dataset)

        dm.update_quality_check.assert_called_once_with(fake_original_dataset, insert_times, append_times)
        dm.insert_into_dataset.assert_not_called()
        dm.append_to_dataset.assert_called_once_with(publish_dataset, append_times)

    @staticmethod
    def test_update_zarr_insert_but_overwrite_not_allowed(manager_class, fake_original_dataset):
        publish_dataset = object()

        dm = manager_class()
        insert_times, append_times = [object()], [object()]
        dm.prepare_update_times = mock.Mock(return_value=(insert_times, append_times))
        dm.update_quality_check = mock.Mock()
        dm.insert_into_dataset = mock.Mock()
        dm.append_to_dataset = mock.Mock()

        dm.update_zarr(publish_dataset)

        dm.update_quality_check.assert_called_once_with(fake_original_dataset, insert_times, append_times)
        dm.insert_into_dataset.assert_not_called()
        dm.append_to_dataset.assert_called_once_with(publish_dataset, append_times)

    @staticmethod
    def test_update_zarr_insert(manager_class, fake_original_dataset):
        publish_dataset = object()
        dm = manager_class()
        dm.allow_overwrite = True
        insert_times, append_times = [object()], []
        dm.prepare_update_times = mock.Mock(return_value=(insert_times, append_times))
        dm.update_quality_check = mock.Mock()
        dm.insert_into_dataset = mock.Mock()
        dm.append_to_dataset = mock.Mock()
        dm.requested_dask_chunks = {"time": 5, "latitude": 4, "longitude": 4}
        dm.complete_insert_slice = mock.Mock()

        dm.update_zarr(publish_dataset)

        dm.update_quality_check.assert_called_once_with(fake_original_dataset, insert_times, append_times)
        dm.insert_into_dataset.assert_called_once_with(fake_original_dataset, publish_dataset, insert_times)
        dm.append_to_dataset.assert_not_called()

    @staticmethod
    def test_insert_into_dataset_dry_run(manager_class):
        # Default DM without chunk alignment
        dm = manager_class()
        dm.dry_run = True
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.calculate_update_time_ranges.return_value = (
            (("breakfast", "second breakfast"), ("dusk", "dawn")),
            (("the shire", "mordor"), ("vegas", "atlantic city")),
        )
        dm.to_zarr = mock.Mock()

        # Mock original and update datasets
        original_dataset = mock.Mock()
        update_dataset = object()
        insert_times = object()

        slice1 = mock.Mock()
        slice2 = mock.Mock()
        insert_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        insert_dataset.attrs = {}
        insert_dataset.sel.side_effect = [slice1, slice2]

        slice1 = mock.Mock()
        slice2 = mock.Mock()
        insert_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        insert_dataset.attrs = {}
        insert_dataset.sel.side_effect = [slice1, slice2]

        dm.insert_into_dataset(original_dataset, update_dataset, insert_times)

        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times)
        dm.calculate_update_time_ranges.assert_called_once_with(original_dataset, insert_dataset)

        insert_dataset.sel.assert_has_calls(
            [mock.call(time=slice("breakfast", "second breakfast")), mock.call(time=slice("dusk", "dawn"))]
        )
        dm.to_zarr.assert_has_calls(
            [
                mock.call(
                    slice1.drop_vars.return_value, store=dm.store.path, region={"time": slice("the shire", "mordor")}
                ),
                mock.call(
                    slice2.drop_vars.return_value,
                    store=dm.store.path,
                    region={"time": slice("vegas", "atlantic city")},
                ),
            ]
        )
        assert insert_dataset.attrs == {"update_is_append_only": False}

        # DM with chunk alignment
        dm = manager_class()
        dm.dry_run = True
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.calculate_update_time_ranges.return_value = (
            (("breakfast", "second breakfast"), ("dusk", "dawn")),
            (("the shire", "mordor"), ("vegas", "atlantic city")),
        )
        dm.to_zarr = mock.Mock()
        dm.requested_dask_chunks = {"time": 5, "latitude": 4, "longitude": 4}
        dm.align_update_chunks = True

        # New mocks for original and update datasets
        original_dataset = mock.Mock()
        update_dataset = object()
        insert_times = object()

        slice1 = mock.Mock()
        slice2 = mock.Mock()
        insert_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        insert_dataset.attrs = {}
        insert_dataset.sel.side_effect = [slice1, slice2]

        full_index_slice_1, full_chunks_region_1 = mock.Mock(), ("full dusk", "full dawn")
        full_index_slice_2, full_chunks_region_2 = mock.Mock(), ("full vegas", "full atalntic city")
        with mock.patch(
            "gridded_etl_tools.utils.publish.complete_insert_slice",
            side_effect=[(full_index_slice_1, full_chunks_region_1), (full_index_slice_2, full_chunks_region_2)],
        ) as patched_complete_slice:
            dm.insert_into_dataset(original_dataset, update_dataset, insert_times)
            patched_complete_slice.assert_has_calls(
                [
                    mock.call(slice1, original_dataset, ("the shire", "mordor"), 5, "time"),
                    mock.call(slice2, original_dataset, ("vegas", "atlantic city"), 5, "time"),
                ]
            )

        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times)
        dm.calculate_update_time_ranges.assert_called_once_with(original_dataset, insert_dataset)

        insert_dataset.sel.assert_has_calls(
            [mock.call(time=slice("breakfast", "second breakfast")), mock.call(time=slice("dusk", "dawn"))]
        )
        dm.to_zarr.assert_has_calls(
            [
                mock.call(
                    full_index_slice_1.drop_vars.return_value,
                    store=dm.store.path,
                    region={"time": slice(*full_chunks_region_1)},
                ),
                mock.call(
                    full_index_slice_2.drop_vars.return_value,
                    store=dm.store.path,
                    region={"time": slice(*full_chunks_region_2)},
                ),
            ]
        )

        assert insert_dataset.attrs == {"update_is_append_only": False}

    @staticmethod
    def test_append_to_dataset_dry_run(manager_class):
        dm = manager_class()
        dm.dry_run = True
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        update_dataset = object()
        insert_times = object()

        append_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        append_dataset.attrs = {}

        dm.append_to_dataset(update_dataset, insert_times)

        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times)

        dm.to_zarr.assert_called_once_with(append_dataset, store=dm.store.path, append_dim="time")

        assert append_dataset.attrs == {"update_is_append_only": True}

        # Test behavior with chunk alignment enabled
        dm.align_update_chunks = True
        dm.rechunk_append_dataset = mock.Mock()
        pre_rechunk_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        append_dataset = dm.rechunk_append_dataset.return_value = mock.MagicMock()
        dm.append_to_dataset(update_dataset, insert_times)
        dm.rechunk_append_dataset.assert_called_once_with(pre_rechunk_dataset)

    @staticmethod
    def test_prep_update_dataset(manager_class, fake_complex_update_dataset):
        # Give the transpose call in prep_update_dataset something to do
        dataset = fake_complex_update_dataset.transpose("longitude", "latitude", "time")
        assert dataset.data.dims == ("longitude", "latitude", "time")
        time_values = np.arange(
            np.datetime64("2022-02-01T00:00:00.000000000"),
            np.datetime64("2022-03-09T00:00:00.000000000"),
            np.timedelta64(1, "[D]"),
        )
        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        dm.set_zarr_metadata = lambda x: x
        dm.requested_dask_chunks = {"time": 5, "latitude": 4, "longitude": 4}

        assert len(dataset.time) > len(time_values)

        dataset = dm.prep_update_dataset(dataset, time_values)
        dm.encode_ds(dataset)

        assert np.array_equal(dataset.time, time_values)
        assert dataset.chunks == {}
        assert dataset.data.dims == ("time", "latitude", "longitude")

    @staticmethod
    def test_prep_update_dataset_no_time_dimension(manager_class, fake_complex_update_dataset):
        # Give the transpose call in prep_update_dataset something to do
        dataset = fake_complex_update_dataset.transpose("longitude", "latitude", "time")
        assert dataset.data.dims == ("longitude", "latitude", "time")
        dataset = dataset.sel(time=[np.datetime64("2022-02-01T00:00:00.000000000")]).squeeze()
        time_values = np.arange(
            np.datetime64("2022-02-01T00:00:00.000000000"),
            np.datetime64("2022-03-09T00:00:00.000000000"),
            np.timedelta64(1, "[D]"),
        )
        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        dm.set_zarr_metadata = lambda x: x
        dm.requested_dask_chunks = {"time": 5, "latitude": 4, "longitude": 4}

        assert "time" not in dataset.dims

        dataset = dm.prep_update_dataset(dataset, time_values)
        dm.encode_ds(dataset)

        assert np.array_equal(dataset.time, time_values[:1])
        assert dataset.chunks == {}
        assert dataset.data.dims == ("time", "latitude", "longitude")

    @staticmethod
    def test_calculate_update_time_ranges(
        manager_class,
        fake_original_dataset,
        fake_complex_update_dataset,
    ):
        """
        Test that the calculate_date_ranges function correctly prepares insert and append date ranges as anticipated
        """
        # prepare a dataset manager
        dm = manager_class()
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

    @staticmethod
    def test_preparse_quality_check(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_ds(fake_original_dataset)
        dm.check_nan_frequency = mock.Mock()
        dm.store = mock.Mock(spec=store.Local, has_existing=True)

        dm.pre_parse_quality_check(fake_original_dataset)

        dm.check_random_values.assert_called_once_with(fake_original_dataset)
        dm.check_nan_frequency.assert_called_once()

    @staticmethod
    def test_preparse_quality_check_short_dataset(manager_class, single_time_instant_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_ds(single_time_instant_dataset)
        dm.check_nan_frequency = mock.Mock()
        dm.store = mock.Mock(spec=store.Local, has_existing=True)

        dm.pre_parse_quality_check(single_time_instant_dataset)

        dm.check_random_values.assert_called_once_with(single_time_instant_dataset)
        dm.check_nan_frequency.assert_called_once()

    @staticmethod
    def test_preparse_quality_check_noncontiguous_time(manager_class, fake_original_dataset):
        drop_times = fake_original_dataset.time[5:10]
        dataset = fake_original_dataset.drop_sel(time=drop_times)
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.check_nan_frequency = mock.Mock()
        dm.encode_ds(dataset)

        with pytest.raises(IndexError):
            dm.pre_parse_quality_check(dataset)

    @staticmethod
    def test_preparse_quality_check_bad_dtype(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_ds(fake_original_dataset)
        dm.check_nan_frequency = mock.Mock()
        fake_original_dataset.data.encoding["dtype"] = "thewrongtype"

        with pytest.raises(TypeError):
            dm.pre_parse_quality_check(fake_original_dataset)

    def test_preparse_quality_check_nan_binomial(mocker, manager_class, fake_large_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_ds(fake_large_dataset)
        dm.store = mock.Mock(spec=store.Local, has_existing=True)

        # patch sample size to 16, size of input dataset
        fake_large_dataset.attrs["expected_nan_frequency"] = 0.2
        dm.store.dataset = mock.Mock(return_value=fake_large_dataset)
        data_shape = np.shape(fake_large_dataset.data)

        # Check that it catches all NaNs
        fake_large_dataset.data[:] = np.nan
        dm.pre_chunk_dataset = fake_large_dataset
        with pytest.raises(NanFrequencyMismatchError):
            dm.pre_parse_quality_check(fake_large_dataset)

        # Check that it catches some NaNs
        partial_nan_array = generate_partial_nan_array(data_shape, 0.5)
        fake_large_dataset.data[:] = partial_nan_array
        dm.pre_chunk_dataset = fake_large_dataset
        with pytest.raises(NanFrequencyMismatchError):
            dm.pre_parse_quality_check(fake_large_dataset)

        # Check that it passes NaNs at or near the threeshold
        partial_nan_array = generate_partial_nan_array(data_shape, 0.2)
        fake_large_dataset.data[:] = partial_nan_array
        dm.pre_chunk_dataset = fake_large_dataset
        dm.pre_parse_quality_check(fake_large_dataset)

        # # Check that it passes NaNs below the threshold
        partial_nan_array = generate_partial_nan_array(data_shape, 0)
        fake_large_dataset.data[:] = partial_nan_array
        dm.pre_chunk_dataset = fake_large_dataset
        with pytest.raises(NanFrequencyMismatchError):
            dm.pre_parse_quality_check(fake_large_dataset)

    @staticmethod
    def test_preparse_quality_check_nan_binomial_small_array(mocker, manager_class, fake_original_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_ds(fake_original_dataset)
        dm.store = mock.Mock(spec=store.Local, has_existing=True)

        # patch sample size to 16, size of input dataset
        fake_original_dataset.attrs["expected_nan_frequency"] = 1
        fake_original_dataset.data[:] = np.nan
        dm.pre_chunk_dataset = fake_original_dataset
        dm.pre_parse_quality_check(fake_original_dataset)

    @staticmethod
    def test_preparse_quality_check_nan_binomial_no_existing(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_ds(fake_original_dataset)
        dm.check_nan_frequency = mock.Mock()
        dm.store = mock.Mock(spec=store.Local, has_existing=False)
        dm.pre_parse_quality_check(fake_original_dataset)

        dm.check_random_values.assert_called_once_with(fake_original_dataset)
        dm.check_nan_frequency.assert_not_called()

    @staticmethod
    def test_preparse_quality_check_nan_binomial_skip_check(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_ds(fake_original_dataset)
        dm.skip_pre_parse_nan_check = True
        dm.check_nan_frequency = mock.Mock()
        dm.store = mock.Mock(spec=store.Local, has_existing=True)

        dm.pre_parse_quality_check(fake_original_dataset)

        dm.check_random_values.assert_called_once_with(fake_original_dataset)
        dm.check_nan_frequency.assert_not_called()

    @staticmethod
    def test_preparse_quality_check_no_frequency_attribute(mocker, manager_class, fake_original_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_ds(fake_original_dataset)
        dm.store = mock.Mock(spec=store.Local, has_existing=True)

        # raise AttributeError if expected_nan_frequency attribute not present
        dm.pre_chunk_dataset = fake_original_dataset
        with pytest.raises(AttributeError):
            dm.pre_parse_quality_check(fake_original_dataset)

    @staticmethod
    def test_check_random_values_all_ok(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.EXTREME_VALUES_BY_UNIT = {"parsecs": (-10, 10)}
        dm.pre_chunk_dataset = fake_original_dataset
        dm.encode_ds(fake_original_dataset)
        dm.check_random_values(fake_original_dataset)

    @staticmethod
    def test_check_random_values_NaN(manager_class, fake_original_dataset):
        fake_original_dataset.data.values[:] = np.nan
        dm = manager_class()
        dm.pre_chunk_dataset = fake_original_dataset
        dm.encode_ds(fake_original_dataset)
        with pytest.raises(ValueError):
            dm.check_random_values(fake_original_dataset)

    @staticmethod
    def test_check_random_values_nonsense_value(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.pre_chunk_dataset = fake_original_dataset
        dm.encode_ds(fake_original_dataset)

        fake_original_dataset["data"].encoding["units"] = "K"
        fake_original_dataset.data.values[:] = 1_000_000  # hot

        with pytest.raises(ValueError):
            dm.check_random_values(fake_original_dataset)

    @staticmethod
    def test_check_random_values_time_dimension_only(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.pre_chunk_dataset = fake_original_dataset
        dataset = fake_original_dataset.drop_vars(dm._standard_dims_except(dm.time_dim))
        dm.encode_ds(dataset)
        dm.check_random_values(dataset)

    @staticmethod
    def test_check_random_values_NaNs_are_allowed(manager_class, fake_original_dataset):
        fake_original_dataset.data.values[:] = np.nan
        dm = manager_class()
        dm.pre_chunk_dataset = fake_original_dataset
        dm.encode_ds(fake_original_dataset)
        dm.has_nans = True
        dm.check_random_values(fake_original_dataset)

    @staticmethod
    def test_update_quality_check_everything_ok(manager_class, fake_original_dataset):
        time_dim = fake_original_dataset["time"]
        time_step = time_dim[1] - time_dim[0]

        insert_times = []
        append_times = [time_dim[-1] + time_step]

        dm = manager_class()
        dm.update_quality_check(fake_original_dataset, insert_times, append_times)

    @staticmethod
    def test_update_quality_check_early_append(manager_class, fake_original_dataset):
        time_dim = fake_original_dataset["time"]
        time_step = time_dim[1] - time_dim[0]

        insert_times = []
        append_times = [time_dim[0] - time_step]

        dm = manager_class()
        with pytest.raises(IndexError):
            dm.update_quality_check(fake_original_dataset, insert_times, append_times)

    @staticmethod
    def test_update_quality_check_early_insert(manager_class, fake_original_dataset):
        time_dim = fake_original_dataset["time"]
        time_step = time_dim[1] - time_dim[0]

        insert_times = [time_dim[0] - time_step]
        append_times = []

        dm = manager_class()
        with pytest.raises(IndexError):
            dm.update_quality_check(fake_original_dataset, insert_times, append_times)

    @staticmethod
    def test_update_quality_check_discontinuous_time(manager_class, fake_original_dataset):
        time_dim = fake_original_dataset["time"]
        time_step = time_dim[1] - time_dim[0]

        insert_times = []
        append_times = [time_dim[-1] + 2 * time_step]

        dm = manager_class()
        with pytest.raises(IndexError):
            dm.update_quality_check(fake_original_dataset, insert_times, append_times)

    @staticmethod
    def test_update_quality_check_nothing_to_do(manager_class, fake_original_dataset):
        dm = manager_class()
        with pytest.raises(ValueError):
            dm.update_quality_check(fake_original_dataset, [], [])

    @staticmethod
    def test_are_times_in_expected_order_regular_cadence_ok(manager_class):
        start = np.datetime64("2000-01-01T00:00:00")
        delta = np.datetime64("2000-01-01T01:00:00") - start
        times = [start + i * delta for i in range(10)]

        dm = manager_class()
        assert dm.are_times_in_expected_order(times, delta) is True

    @staticmethod
    def test_are_times_in_expected_order_regular_cadence_not_ok(manager_class):
        start = np.datetime64("2000-01-01T00:00:00")
        delta = np.datetime64("2000-01-01T01:00:00") - start
        times = [start + i * delta for i in range(10)] + [start + delta * 20]

        dm = manager_class()
        assert dm.are_times_in_expected_order(times, delta) is False

    @staticmethod
    def test_are_times_in_expected_order_irregular_cadence_ok(manager_class):
        start = np.datetime64("2000-01-01T00:00:00")
        delta = np.datetime64("2000-01-01T01:00:00") - start
        times = [start + i * delta * 1.05 for i in range(10)]

        class MyManager(manager_class):
            update_cadence_bounds = (delta / 2, delta * 2)

        dm = MyManager()
        assert dm.are_times_in_expected_order(times, delta) is True

    @staticmethod
    def test_are_times_in_expected_order_irregular_cadence_not_ok(manager_class):
        start = np.datetime64("2000-01-01T00:00:00")
        delta = np.datetime64("2000-01-01T01:00:00") - start
        times = [start + i * delta * 2.5 for i in range(10)]

        class MyManager(manager_class):
            update_cadence_bounds = (delta / 2, delta * 2)

        dm = MyManager()
        assert dm.are_times_in_expected_order(times, delta) is False

    @staticmethod
    def test_post_parse_quality_check(manager_class, mocker):
        dm = manager_class()
        dm.set_key_dims = mock.Mock()
        dm.get_random_coords = mock.Mock()
        dm.get_random_coords.return_value = ({"a": i} for i in range(1000))  # pragma NO COVER
        dm.raw_file_to_dataset = mock.Mock()
        dm.get_prod_update_ds = mock.Mock()
        dm.filter_search_space = mock.Mock()
        random = mocker.patch("gridded_etl_tools.utils.publish.random")

        orig_ds = dm.raw_file_to_dataset.return_value
        prod_ds = dm.get_prod_update_ds.return_value

        def check_written_value(dataset1, dataset2, threshold):
            assert dataset1 is orig_ds
            assert dataset2 is prod_ds
            assert threshold == 10e-5

        dm.check_written_value = check_written_value

        dm.post_parse_quality_check()

        # assert setup functions called once
        dm.set_key_dims.assert_called_once_with()
        dm.filter_search_space.assert_called_once_with(prod_ds)
        dm.get_prod_update_ds.assert_called_once_with()
        # assert functions called in loop
        assert random.choice.call_count == 100
        random.choice.assert_called_with(dm.filter_search_space())
        assert dm.raw_file_to_dataset.call_count == 100
        dm.raw_file_to_dataset.assert_called_with(random.choice(dm.filter_search_space(prod_ds)))

    @staticmethod
    def test_post_parse_quality_check_skip_it(manager_class, mocker):
        dm = manager_class()
        dm.skip_post_parse_qc = True
        dm.set_key_dims = mock.Mock()
        dm.raw_file_to_dataset = mock.Mock()
        dm.get_prod_update_ds = mock.Mock()
        dm.filter_search_space = mock.Mock()
        dm.check_written_value = mock.Mock()

        dm.post_parse_quality_check()

        dm.set_key_dims.assert_not_called()
        dm.get_prod_update_ds.assert_not_called()
        dm.filter_search_space.assert_not_called()
        dm.raw_file_to_dataset.assert_not_called()
        dm.check_written_value.assert_not_called()

    @staticmethod
    def test_post_parse_quality_check_timeout(manager_class, mocker):
        time = mocker.patch("gridded_etl_tools.utils.publish.time")
        time.perf_counter = mock.Mock(side_effect=[0, 1, 2, 5000, 5001])

        dm = manager_class()
        dm.set_key_dims = mock.Mock()
        dm.get_random_coords = mock.Mock()
        dm.get_random_coords.return_value = ({"a": i} for i in range(1000))  # pragma NO COVER
        dm.raw_file_to_dataset = mock.Mock()
        dm.get_prod_update_ds = mock.Mock()
        dm.filter_search_space = mock.Mock()
        random = mocker.patch("gridded_etl_tools.utils.publish.random")

        orig_ds = dm.raw_file_to_dataset.return_value
        prod_ds = dm.get_prod_update_ds.return_value

        def check_written_value(dataset1, dataset2, threshold):
            assert dataset1 is orig_ds
            assert dataset2 is prod_ds
            assert threshold == 10e-5

        dm.check_written_value = check_written_value

        dm.post_parse_quality_check()

        # assert setup functions called once
        dm.set_key_dims.assert_called_once_with()
        dm.filter_search_space.assert_called_once_with(prod_ds)
        dm.get_prod_update_ds.assert_called_once_with()
        # assert functions called in loop
        assert random.choice.call_count == 3
        random.choice.assert_called_with(dm.filter_search_space())
        assert dm.raw_file_to_dataset.call_count == 3
        dm.raw_file_to_dataset.assert_called_with(random.choice(dm.filter_search_space(prod_ds)))

    @staticmethod
    def test_get_prod_update_ds(manager_class, fake_original_dataset):
        fake_original_dataset.attrs["update_date_range"] = ("2021120100", "2022010100")

        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.store.dataset.return_value = fake_original_dataset

        dataset = dm.get_prod_update_ds()
        assert dataset["time"].values[0] == np.datetime64("2021-12-01T00:00:00.000000000")
        assert dataset["time"].values[-1] == np.datetime64("2022-01-01T00:00:00.000000000")

    @staticmethod
    def test_check_written_value(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy()

        dm.check_written_value(fake_original_dataset, prod_ds)

    @staticmethod
    def test_check_written_value_value_is_out_of_bounds(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(fake_original_dataset.dims, coord_indices)}
        dm.get_random_coords = mock.Mock(return_value=check_coords)

        prod_ds.data[coord_indices] += 10e-4
        with pytest.raises(ValueError):
            dm.check_written_value(fake_original_dataset, prod_ds)

    @staticmethod
    def test_check_written_value_override_threshold(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)

        prod_ds.data[coord_indices] += 10e-4
        dm.check_written_value(fake_original_dataset, prod_ds, threshold=10e-3)

    @staticmethod
    def test_check_written_value_value_one_infinity(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(fake_original_dataset.dims, coord_indices)}
        dm.get_random_coords = mock.Mock(return_value=check_coords)

        prod_ds.data[coord_indices] = np.inf
        with pytest.raises(ValueError):
            dm.check_written_value(fake_original_dataset, prod_ds, threshold=np.inf)

        prod_ds.data[coord_indices] = fake_original_dataset.data[coord_indices]
        fake_original_dataset.data[coord_indices] = np.inf
        with pytest.raises(ValueError):
            dm.check_written_value(fake_original_dataset, prod_ds, threshold=np.inf)

    @staticmethod
    def test_check_two_nans(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)

        fake_original_dataset.data[coord_indices] = np.nan
        prod_ds.data[coord_indices] = np.nan
        dm.check_written_value(fake_original_dataset, prod_ds)

    @staticmethod
    def test_check_written_value_value_one_missing_value_one_nan(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(fake_original_dataset.dims, coord_indices)}
        dm.get_random_coords = mock.Mock(return_value=check_coords)

        prod_ds.data[coord_indices] = np.nan
        fake_original_dataset.data[coord_indices] = dm.missing_value  # 42
        dm.check_written_value(fake_original_dataset, prod_ds, threshold=np.inf)

    @staticmethod
    def test_check_two_infinities_ish(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(fake_original_dataset.dims, coord_indices)}
        dm.get_random_coords = mock.Mock(return_value=check_coords)

        fake_original_dataset.data[coord_indices] = 5e100
        prod_ds.data[coord_indices] = np.inf
        dm.check_written_value(fake_original_dataset, prod_ds)

    @staticmethod
    def test_check_written_value_value_one_nan(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(fake_original_dataset.dims, coord_indices)}
        dm.get_random_coords = mock.Mock(return_value=check_coords)

        prod_ds.data[coord_indices] = np.nan
        with pytest.raises(ValueError):
            dm.check_written_value(fake_original_dataset, prod_ds, threshold=np.inf)

        prod_ds.data[coord_indices] = fake_original_dataset.data[coord_indices]
        fake_original_dataset.data[coord_indices] = np.nan
        with pytest.raises(ValueError):
            dm.check_written_value(fake_original_dataset, prod_ds, threshold=np.inf)

    @staticmethod
    def test_check_written_value_value_two_nans(manager_class, fake_original_dataset):
        dm = manager_class()
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(fake_original_dataset.dims, coord_indices)}
        dm.get_random_coords = mock.Mock(return_value=check_coords)

        prod_ds.data[coord_indices] = np.nan
        fake_original_dataset.data[coord_indices] = np.nan
        dm.check_written_value(fake_original_dataset, prod_ds, threshold=np.inf)

    @staticmethod
    def test_filter_search_space(manager_class, hindcast_dataset):
        timestamps = np.arange(
            np.datetime64("2021-10-16T00:00:00.000000000"),
            np.datetime64("2021-10-26T00:00:00.000000000"),
            np.timedelta64(1, "[D]"),
        )

        def fake_input_files(range_num: int):
            date_files = []
            for i in range(0, range_num):
                date_str = pd.Timestamp(timestamps[i]).to_pydatetime().date().isoformat()
                date_files.append(f"test_path_time-{date_str}")
            date_step_files = []
            for file in date_files:
                for i in range(1, range_num + 1):
                    date_step_files.append(file + f"_step-{i}")
            date_step_ensemble_files = []
            for file in date_step_files:
                for i in range(1, range_num + 1):
                    date_step_ensemble_files.append(file + f"_ensemble-{i}")
            date_step_ensemble_fro_files = []
            for file in date_step_ensemble_files:
                for i in range(1, range_num + 1):
                    date_step_ensemble_fro_files.append(file + f"_forecast_reference_offset-{i}")
            return date_step_ensemble_fro_files

        def raw_file_to_dataset(file_path):
            path_date = np.datetime64(re.search(r"time-(\d{4}-\d{2}-\d{2})_", file_path)[1])
            raw_ds = hindcast_dataset.assign_coords({"hindcast_reference_time": np.atleast_1d(path_date)})
            return raw_ds

        dm = manager_class()
        dm.dataset_category = "hindcast"
        dm.set_key_dims()
        dm.input_files = mock.Mock(return_value=fake_input_files(10))
        dm.raw_file_to_dataset = mock.Mock(side_effect=raw_file_to_dataset)

        hindcast_dataset.attrs["update_date_range"] = ("2021102400", "2021102500")

        assert len(dm.filter_search_space(hindcast_dataset)) == 2000

    @staticmethod
    def test_dataset_date_in_range(manager_class, fake_original_dataset):
        """Test the dataset_date_in_range method with various date ranges
        The method only takes the *first* date from the dataset as it's designed for
        processing single files in the post-parse QC"""
        dm = manager_class()
        dm.raw_file_to_dataset = mock.Mock(return_value=fake_original_dataset)

        # Test case 1: Date is within range
        date_range = (
            datetime.datetime(2021, 9, 15, 0, 0),  # One day before dataset start
            datetime.datetime(2022, 2, 1, 0, 0),  # One day after dataset end
        )
        assert dm.dataset_date_in_range(date_range, pathlib.Path("test_file.nc")) is True

        # Test case 2: DS dates start before range
        date_range = (
            datetime.datetime(2021, 9, 17, 0, 0),  # One day after dataset start
            datetime.datetime(2022, 2, 1, 0, 0),  # One day after dataset end
        )
        assert dm.dataset_date_in_range(date_range, pathlib.Path("test_file.nc")) is False

        # Test case 3: DS dates exceed range, but first date is within range
        date_range = (
            datetime.datetime(2021, 9, 15, 0, 0),  # One day before dataset start
            datetime.datetime(2022, 1, 30, 0, 0),  # One day before dataset end
        )
        assert dm.dataset_date_in_range(date_range, pathlib.Path("test_file.nc")) is True

        # Test case 4: Dates are exactly at range boundaries
        date_range = (
            datetime.datetime(2021, 9, 16, 0, 0),  # Exactly at dataset start
            datetime.datetime(2022, 1, 31, 0, 0),  # Exactly at dataset end
        )
        assert dm.dataset_date_in_range(date_range, pathlib.Path("test_file.nc")) is True

    @staticmethod
    def test_raw_file_to_dataset_bad_protocol(manager_class):
        dm = manager_class()
        dm.protocol = "nopenoway"
        with pytest.raises(ValueError):
            dm.raw_file_to_dataset("some/path")

    @staticmethod
    def test_raw_file_to_dataset_file(manager_class, mocker, fake_original_dataset):
        xr = mocker.patch("gridded_etl_tools.utils.publish.xr")
        dm = manager_class()
        dm.protocol = "file"
        dm.preprocess_zarr = mock.Mock()
        dm.postprocess_zarr = mock.Mock(return_value=fake_original_dataset)
        ds = dm.raw_file_to_dataset("some/path")
        assert ds == dm.reformat_orig_ds(fake_original_dataset, "some/path")
        xr.open_dataset.assert_called_once_with("some/path")

    # NOTE disabled due to regression in fsspec capabilities
    # @staticmethod
    # def test_raw_file_to_dataset_s3(manager_class, fake_original_dataset):
    #     dm = manager_class()
    #     dm.load_dataset_from_disk = mock.Mock()
    #     dm.load_dataset_from_disk.return_value = fake_original_dataset
    #     dm.protocol = "s3"
    #     dm.use_local_zarr_jsons = True

    #     ds = dm.raw_file_to_dataset(pathlib.PosixPath("some/path"))
    #     xr.testing.assert_equal(ds, dm.reformat_orig_ds(fake_original_dataset, pathlib.Path("some/path")))
    #     dm.load_dataset_from_disk.assert_called_once_with(zarr_json_path="some/path")

    @staticmethod
    def test_raw_file_to_dataset_s3_no_local_zarr_json(manager_class):
        dm = manager_class()
        dm.zarr_json_to_dataset = mock.Mock()
        dm.protocol = "s3"
        dm.use_local_zarr_jsons = False
        with pytest.raises(ValueError):
            dm.raw_file_to_dataset(pathlib.PosixPath("some/path"))

    @staticmethod
    def test_raw_file_to_dataset_protocol_handling(manager_class, mocker):
        """Test handling of different protocols in raw_file_to_dataset"""
        # Test file protocol
        dm = manager_class()
        dm.protocol = "file"
        dm.preprocess_zarr = mock.Mock()
        dm.postprocess_zarr = mock.Mock()
        dm.reformat_orig_ds = mock.Mock()

        xr = mocker.patch("gridded_etl_tools.utils.publish.xr")
        xr.open_dataset.return_value = mock.Mock()

        dm.raw_file_to_dataset("test/path")

        xr.open_dataset.assert_called_once_with("test/path")
        dm.preprocess_zarr.assert_called_once()
        dm.postprocess_zarr.assert_called_once()
        dm.reformat_orig_ds.assert_called_once()

        # Test S3 protocol with local zarr jsons enabled
        dm = manager_class()
        dm.protocol = "s3"
        dm.use_local_zarr_jsons = True
        dm.load_dataset_from_disk = mock.Mock()
        dm.reformat_orig_ds = mock.Mock()

        dm.raw_file_to_dataset("s3://test/path")

        dm.load_dataset_from_disk.assert_called_once_with(zarr_json_path="s3://test/path")
        dm.reformat_orig_ds.assert_called_once()

        # Test S3 protocol with local zarr jsons disabled
        dm = manager_class()
        dm.protocol = "s3"
        dm.use_local_zarr_jsons = False

        with pytest.raises(ValueError, match="ETL protocol is S3 but it was instantiated not to use local zarr JSONs"):
            dm.raw_file_to_dataset("s3://test/path")

    @staticmethod
    def test_reformat_orig_ds(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.rename_data_variable = mock.Mock(return_value=fake_original_dataset)
        dm.reformat_orig_ds(fake_original_dataset, "hi/mom.zarr")

        dm.rename_data_variable.assert_called_once_with(fake_original_dataset)

    @staticmethod
    def test_reformat_orig_ds_single_time_instant(manager_class, single_time_instant_dataset):
        dm = manager_class()
        dm.rename_data_variable = mock.Mock(return_value=single_time_instant_dataset)
        orig_dataset = single_time_instant_dataset.squeeze()
        dataset = dm.reformat_orig_ds(orig_dataset, "hi/mom.zarr")

        assert "time" in dataset.dims

        dm.rename_data_variable.assert_called_once_with(orig_dataset)

    @staticmethod
    def test_reformat_orig_ds_no_time_at_all(manager_class, single_time_instant_dataset):
        dm = manager_class()
        dm.rename_data_variable = mock.Mock(return_value=single_time_instant_dataset)
        orig_dataset = single_time_instant_dataset.squeeze().drop_vars("time")
        dataset = dm.reformat_orig_ds(orig_dataset, "hi/mom.zarr")

        assert "time" in dataset

        dm.rename_data_variable.assert_called_once_with(orig_dataset)

    @staticmethod
    def test_reformat_orig_ds_no_time_in_data_var(manager_class, single_time_instant_dataset):
        dm = manager_class()
        dm.rename_data_variable = mock.Mock(return_value=single_time_instant_dataset)
        orig_dataset = single_time_instant_dataset
        orig_dataset[dm.data_var] = orig_dataset[dm.data_var].squeeze()
        dataset = dm.reformat_orig_ds(orig_dataset, "hi/mom.zarr")

        assert "time" in dataset[dm.data_var].dims

        dm.rename_data_variable.assert_called_once_with(orig_dataset)

    @staticmethod
    def test_reformat_orig_ds_missing_step_dimension(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.rename_data_variable = mock.Mock(return_value=fake_original_dataset)
        dm.standard_dims += ["step"]
        dataset = dm.reformat_orig_ds(fake_original_dataset, "hi/mom-2022-07-04.zarr")

        assert "step" in dataset
        assert "step" in dataset.dims
        assert dataset.step[0] == np.datetime64("2022-07-04T00:00:00.000000000")

        dm.rename_data_variable.assert_called_once_with(dataset)


class DummyDataset(UserDict):
    def __init__(self, *dims: tuple[tuple[str, list[float]]]):
        self.dims = tuple((dim for dim, _ in dims))
        super().__init__(((dim, mock.Mock(values=values))) for dim, values in dims)

    def coords(self):
        n_elements = functools.reduce(operator.mul, (len(values.values) for values in self.values()))
        for i in range(n_elements):
            coords = []
            x = i
            for dim in self.dims:
                values = self[dim].values
                n_values = len(values)
                coords.append(values[x % n_values])
                x //= n_values

            yield {dim: coord for dim, coord in zip(self.dims, coords)}


def test_shuffled_coords():
    dataset = DummyDataset(
        ("a", list(range(10))),
        ("b", [i * 2 for i in range(10)]),
        ("c", [i * 1.5 for i in range(10)]),
    )
    unshuffled = list(dataset.coords())
    shuffled = list(publish.shuffled_coords(dataset))

    # infinitesimally small chance they match, so keep going until they don't, to make sure shuffling is going on
    while unshuffled == shuffled:  # pragma NO COVER
        shuffled = list(publish.shuffled_coords(dataset))

    # order should be different but set of values should be the same
    unshuffled_set = set((frozenset(coords.items()) for coords in unshuffled))
    shuffled_set = set((frozenset(coords.items()) for coords in shuffled))
    assert shuffled_set == unshuffled_set


def test_is_infish():
    # Test regular numbers
    assert not _is_infish(np.float64(42))
    assert not _is_infish(np.float32(42))

    # Test infinities
    assert _is_infish(np.float64("inf"))
    assert _is_infish(np.float32("inf"))
    assert _is_infish(-np.float64("inf"))
    assert _is_infish(-np.float32("inf"))

    # Test very large numbers
    assert _is_infish(np.float64(1e101))  # Beyond 1e100 threshold for float64
    assert not _is_infish(np.float64(1e99))  # Below 1e100 threshold for float64

    # Test float32 large numbers
    assert _is_infish(np.float32(1e39))  # Beyond 1e38 threshold for float32
    assert not _is_infish(np.float32(1e37))  # Below 1e38 threshold for float32


# calculate_time_dim_chunks tests


def test_calculate_time_dim_chunks_basic():
    """
    Test basic functionality with simple inputs
    """
    # Case: old dataset has 3 items in final chunk, desired chunk size is 5, update has 7 items
    # First chunk should be 5-3=2, then full chunk of 5, then remainder of 0
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=3, time_dim_chunk_size=5, append_time_length=7)
    expected = (2, 5)
    assert result == expected
    assert sum(result) == 7


def test_calculate_time_dim_chunks_perfect_alignment():
    """
    Test when update length exactly fills remaining space in old chunk
    """
    # Case: old dataset has 2 items in final chunk, desired chunk size is 5, update has 3 items
    # First chunk should be 5-2=3, which exactly matches update length
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=2, time_dim_chunk_size=5, append_time_length=3)
    expected = (3,)
    assert result == expected
    assert sum(result) == 3


def test_calculate_time_dim_chunks_no_remainder():
    """
    Test when chunks divide evenly after the first chunk
    """
    # Case: old dataset has 1 item in final chunk, desired chunk size is 4, update has 11 items
    # First chunk should be 4-1=3, then two full chunks of 4 each
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=1, time_dim_chunk_size=4, append_time_length=11)
    expected = (3, 4, 4)
    assert result == expected
    assert sum(result) == 11


def test_calculate_time_dim_chunks_with_remainder():
    """
    Test when there's a partial chunk at the end
    """
    # Case: old dataset has 2 items in final chunk, desired chunk size is 5, update has 10 items
    # First chunk should be 5-2=3, then full chunk of 5, then remainder of 2
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=2, time_dim_chunk_size=5, append_time_length=10)
    expected = (3, 5, 2)
    assert result == expected
    assert sum(result) == 10


def test_calculate_time_dim_chunks_small_update():
    """
    Test when update is smaller than needed to complete old chunk
    """
    # Case: old dataset has 2 items in final chunk, desired chunk size is 5, update has only 2 items
    # First chunk should be min(5-2, 2) = min(3, 2) = 2
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=2, time_dim_chunk_size=5, append_time_length=2)
    expected = (2,)
    assert result == expected
    assert sum(result) == 2


def test_calculate_time_dim_chunks_old_chunk_full():
    """
    Test when old dataset's final chunk is already at desired size
    """
    # Case: old dataset has 5 items in final chunk, desired chunk size is 5, update has 8 items
    # First chunk should be min(5-5, 8) = min(0, 8) = 0, so no first chunk, then full chunks
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=5, time_dim_chunk_size=5, append_time_length=8)
    expected = (5, 3)
    assert result == expected
    assert sum(result) == 8


def test_calculate_time_dim_chunks_zero_old_chunk():
    """
    Test when old dataset final chunk length is 0 (no existing data)
    """
    # Case: old dataset has 0 items in final chunk, desired chunk size is 4, update has 9 items
    # First chunk should be min(4-0, 9) = min(4, 9) = 4, then full chunk of 4, then remainder of 1
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=0, time_dim_chunk_size=4, append_time_length=9)
    expected = (4, 4, 1)
    assert result == expected
    assert sum(result) == 9


def test_calculate_time_dim_chunks_single_item_update():
    """
    Test with minimal update length of 1
    """
    # Case: old dataset has 3 items in final chunk, desired chunk size is 5, update has 1 item
    # First chunk should be min(5-3, 1) = min(2, 1) = 1
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=3, time_dim_chunk_size=5, append_time_length=1)
    expected = (1,)
    assert result == expected
    assert sum(result) == 1


def test_calculate_time_dim_chunks_large_update():
    """
    Test with large update that spans many chunks
    """
    # Case: old dataset has 1 item in final chunk, desired chunk size is 3, update has 20 items
    # First chunk should be 3-1=2, then 6 full chunks of 3, then remainder of 0
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=1, time_dim_chunk_size=3, append_time_length=22)
    expected = (2, 3, 3, 3, 3, 3, 3, 2)
    assert result == expected
    assert sum(result) == 22


def test_calculate_time_dim_chunks_edge_case_zero_first_chunk():
    """
    Test edge case where first chunk would be 0 (old chunk already full)
    """
    # Case: old dataset has 4 items in final chunk, desired chunk size is 4, update has 6 items
    # First chunk should be min(4-4, 6) = min(0, 6) = 0, so empty tuple for first, then chunks
    result = calculate_time_dim_chunks(old_dataset_final_chunk_length=4, time_dim_chunk_size=4, append_time_length=6)
    expected = (4, 2)
    assert result == expected
    assert sum(result) == 6
