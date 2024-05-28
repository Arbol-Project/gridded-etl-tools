from collections import UserDict
import copy
import functools
import operator
import pathlib

from unittest import mock

import numpy
import pandas as pd
import pytest

from gridded_etl_tools.utils import publish, store
from ...common import convert_raw_times_to_numpy_times


class fake_vmem(dict):
    """
    Fake a vmem object with 16gb total memory using a dict
    """

    def __init__(self):
        self.total = 2**34


class TestPublish:
    @staticmethod
    def test_parse_ipld_first_time(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.publish.nullcontext")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dataset_hash = "QmHiMom!"
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD, has_existing=False)
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

        Client.assert_not_called()
        nullcontext.assert_called_once_with()

    @staticmethod
    def test_parse_ipld_update(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.publish.nullcontext")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD, has_existing=True)
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

        Client.assert_not_called()
        nullcontext.assert_called_once_with()

    @staticmethod
    def test_parse_not_ipld_rebuild(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.publish.nullcontext")
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
        nullcontext.assert_not_called()

    @staticmethod
    def test_parse_not_ipld_rebuild_but_overwrite_not_allowed(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        cluster = LocalCluster.return_value.__enter__.return_value
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.publish.nullcontext")
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
        nullcontext.assert_not_called()

    @staticmethod
    def test_parse_ipld_update_ctrl_c(manager_class, mocker):
        LocalCluster = mocker.patch("gridded_etl_tools.utils.publish.LocalCluster")
        Client = mocker.patch("gridded_etl_tools.utils.publish.Client")
        nullcontext = mocker.patch("gridded_etl_tools.utils.publish.nullcontext")
        mocker.patch("psutil.virtual_memory", return_value=fake_vmem())

        dm = manager_class(rebuild_requested=False)
        dm.dask_configuration = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD, has_existing=True)
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

        Client.assert_not_called()
        nullcontext.assert_called_once_with()

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
    def test_to_zarr_ipld(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.move_post_parse_attrs_to_dict = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)

        dataset = mock.Mock()
        dm.to_zarr(dataset, "foo", bar="baz")

        dataset.to_zarr.assert_called_once_with("foo", bar="baz")
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only.assert_not_called()
        dm.move_post_parse_attrs_to_dict.assert_not_called()

    @staticmethod
    def test_to_zarr_ipld_dry_run(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.move_post_parse_attrs_to_dict = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)
        dm.dry_run = True

        dataset = mock.Mock()
        dm.to_zarr(dataset, "foo", bar="baz")

        dataset.to_zarr.assert_not_called()
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only.assert_not_called()
        dm.move_post_parse_attrs_to_dict.assert_not_called()

    @staticmethod
    def test_to_zarr_not_ipld(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.move_post_parse_attrs_to_dict = mock.Mock()
        dm.move_post_parse_attrs_to_dict.return_value = post_parse_attrs = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)

        dataset = mock.Mock()
        dataset.get.return_value = "is it?"
        dm.to_zarr(dataset, "foo", bar="baz")

        dataset.to_zarr.assert_called_once_with("foo", bar="baz")
        dataset.get.assert_called_once_with("update_is_append_only")
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only.assert_has_calls(
            [
                mock.call(
                    update_attrs={
                        "update_in_progress": True,
                        "update_is_append_only": "is it?",
                        "initial_parse": False,
                    }
                ),
                mock.call(update_attrs=post_parse_attrs),
            ]
        )
        dm.move_post_parse_attrs_to_dict.assert_called_once_with(dataset=dataset)

    @staticmethod
    def test_to_zarr_not_ipld_initial(manager_class, mocker):
        dm = manager_class()
        dm.pre_parse_quality_check = mock.Mock()
        dm.move_post_parse_attrs_to_dict = mock.Mock()
        dm.move_post_parse_attrs_to_dict.return_value = post_parse_attrs = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface, has_existing=False)

        dataset = mock.Mock()
        dataset.get.return_value = "is it?"
        dm.to_zarr(dataset, "foo", bar="baz")

        dataset.to_zarr.assert_called_once_with("foo", bar="baz")
        dm.pre_parse_quality_check.assert_called_once_with(dataset)
        dm.store.write_metadata_only.assert_has_calls([mock.call(update_attrs=post_parse_attrs)])
        dm.move_post_parse_attrs_to_dict.assert_called_once_with(dataset=dataset)

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
        dm.custom_output_path = tmpdir / "to_zarr_dataset.zarr"
        dataset.to_zarr(dm.custom_output_path)  # write out local file to test updates on

        # Mock functions
        dm.pre_parse_quality_check = mock.Mock()

        # Tests
        for key in pre_update_dict.keys():
            assert dm.store.dataset().attrs[key] == pre_update_dict[key]

        dataset.attrs.update(**post_update_dict)
        dm.to_zarr(dataset, dm.store.mapper(), append_dim=dm.time_dim)

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
        dm.custom_output_path = tmpdir / "to_zarr_dataset.zarr"
        dataset.to_zarr(dm.custom_output_path)  # write out local file to test updates on

        # Mock functions
        dm.pre_parse_quality_check = mock.Mock()

        # Tests
        for key in pre_update_dict.keys():
            assert dm.store.dataset().attrs[key] == pre_update_dict[key]

        dataset.attrs.update(**post_update_dict)
        dm.to_zarr(dataset, dm.store.mapper(), append_dim=dm.time_dim)

        for key in post_update_dict.keys():
            assert dm.store.dataset().attrs[key] == post_update_dict[key]

        dm.pre_parse_quality_check.assert_called_once_with(dataset)

    @staticmethod
    def test_move_post_parse_attrs_to_dict(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.update_attributes = dm.update_attributes + ["some relevant attribute", "some non-existant attribute"]
        fake_original_dataset.attrs = {
            "date range": ["2000010100", "2020123123"],
            "update_date_range": ["202012293", "2020123123"],
            "update_previous_end_date": "2020123023",
            "update_in_progress": False,
            "some relevant attribute": True,
            "some irrelevant attribute": False,
            "initial_parse": False,
        }

        update_attrs = dm.move_post_parse_attrs_to_dict(fake_original_dataset)
        assert update_attrs == {
            "update_in_progress": False,
            "date range": ["2000010100", "2020123123"],
            "update_previous_end_date": "2020123023",
            "initial_parse": False,
            "some relevant attribute": True,
        }

    @staticmethod
    def test_dask_configuration_ipld(manager_class, mocker):
        dask_config = {}

        def dask_config_set(config):
            dask_config.update(config)

        dask = mocker.patch("gridded_etl_tools.utils.publish.dask")
        dask.config.set = dask_config_set

        dm = manager_class()
        dm.store = mock.Mock(spec=store.IPLD)

        dm.dask_configuration()

        assert dask_config == {
            "distributed.scheduler.worker-saturation": dm.dask_scheduler_worker_saturation,
            "distributed.scheduler.worker-ttl": None,
            "distributed.worker.memory.target": dm.dask_worker_mem_target,
            "distributed.worker.memory.spill": dm.dask_worker_mem_spill,
            "distributed.worker.memory.pause": dm.dask_worker_mem_pause,
            "distributed.worker.memory.terminate": dm.dask_worker_mem_terminate,
            "scheduler": "threads",
        }

    @staticmethod
    def test_dask_configuration_not_ipld(manager_class, mocker):
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
    def test_write_initial_zarr_ipld(manager_class):
        class DummyHash:
            def __str__(self):
                return "QmHiMom"

        dm = manager_class()
        dm.store = mock.Mock(spec=store.IPLD)
        dm.to_zarr = mock.Mock()
        dm.dataset_hash = None
        publish_dataset = mock.Mock()
        publish_dataset.chunk.return_value = publish_dataset_rechunked = mock.Mock()

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = DummyHash()

        dm.write_initial_zarr(publish_dataset)

        dm.store.mapper.assert_called_once_with(set_root=False)
        dm.to_zarr.assert_called_once_with(publish_dataset_rechunked, mapper, consolidated=True, mode="w")
        assert dm.dataset_hash == "QmHiMom"

    @staticmethod
    def test_write_initial_zarr_not_ipld(manager_class):
        class DummyHash: ...

        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.to_zarr = mock.Mock()
        dm.dataset_hash = None
        publish_dataset = mock.Mock()
        publish_dataset.chunk.return_value = publish_dataset_rechunked = mock.Mock()

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = DummyHash()

        dm.write_initial_zarr(publish_dataset)

        dm.store.mapper.assert_called_once_with(set_root=False)
        dm.to_zarr.assert_called_once_with(publish_dataset_rechunked, mapper, consolidated=True, mode="w")
        assert dm.dataset_hash is None

    # @staticmethod
    # def test_update_zarr(manager_class):
    #     dm = manager_class()
    #     dm.store = mock.Mock(spec=store.StoreInterface)
    #     publish_dataset = mock.Mock()
    #     dm.prepare_update_times = mock.Mock()
    #     dm.update_quality_check = mock.Mock()
    #     dm.insert_into_dataset = mock.Mock()
    #     dm.append_to_dataset = mock.Mock()

    #     original_dataset = dm.store.dataset.return_value
    #     dm.prepare_update_times.return_value = (insert_times, append_times) = ([object()], [object()])
    #     dm.update_zarr(publish_dataset)

    #     dm.store.dataset.assert_called_once_with()
    #     dm.transformed_dataset.assert_called_once_with()
    #     dm.set_key_dims.assert_called_once_with()
    #     dm.prepare_update_times.assert_called_once_with(original_dataset, publish_dataset)
    #     dm.update_quality_check.assert_called_once_with(original_dataset, insert_times, append_times)

    @staticmethod
    def test_prepare_update_times(manager_class, fake_original_dataset, fake_complex_update_dataset):
        dm = manager_class()
        insert_times, update_times = dm.prepare_update_times(fake_original_dataset, fake_complex_update_dataset)
        assert insert_times == [
            numpy.datetime64("2021-10-10T00:00:00.000000000"),
        ] + list(
            numpy.arange(
                numpy.datetime64("2021-10-16T00:00:00.000000000"),
                numpy.datetime64("2021-10-24T00:00:00.000000000"),
                numpy.timedelta64(1, "[D]"),
            )
        ) + [
            numpy.datetime64("2021-11-11T00:00:00.000000000"),
            numpy.datetime64("2021-12-11T00:00:00.000000000"),
        ] + list(
            numpy.arange(
                numpy.datetime64("2021-12-25T00:00:00.000000000"),
                numpy.datetime64("2022-01-06T00:00:00.000000000"),
                numpy.timedelta64(1, "[D]"),
            )
        ) + [
            numpy.datetime64("2022-01-14T00:00:00.000000000"),
        ]
        assert update_times == list(
            numpy.arange(
                numpy.datetime64("2022-02-01T00:00:00.000000000"),
                numpy.datetime64("2022-03-09T00:00:00.000000000"),
                numpy.timedelta64(1, "[D]"),
            )
        )

    @staticmethod
    def test_prepare_update_times_no_time_dimension(manager_class, fake_original_dataset, fake_complex_update_dataset):
        update_dataset = fake_complex_update_dataset.sel(time=[numpy.datetime64("2021-10-10T00:00:00.000000000")])
        update_dataset = update_dataset.squeeze()
        assert "time" not in update_dataset.dims
        dm = manager_class()
        insert_times, update_times = dm.prepare_update_times(fake_original_dataset, update_dataset)
        assert insert_times == [
            numpy.datetime64("2021-10-10T00:00:00.000000000"),
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

        dm.update_zarr(publish_dataset)

        dm.update_quality_check.assert_called_once_with(fake_original_dataset, insert_times, append_times)
        dm.insert_into_dataset.assert_called_once_with(fake_original_dataset, publish_dataset, insert_times)
        dm.append_to_dataset.assert_not_called()

    @staticmethod
    def test_insert_into_dataset_ipld(manager_class):
        dm = manager_class()
        dm.dataset_hash = "browns"
        dm.store = mock.Mock(spec=store.IPLD)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        original_dataset = object()
        update_dataset = object()
        insert_times = object()

        slice1 = mock.Mock()
        slice2 = mock.Mock()
        insert_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        insert_dataset.attrs = {}
        insert_dataset.sel.side_effect = [slice1, slice2]

        dm.calculate_update_time_ranges.return_value = (
            (("breakfast", "second breakfast"), ("dusk", "dawn")),
            (("the shire", "mordor"), ("vegas", "atlantic city")),
        )

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = 42

        dm.insert_into_dataset(original_dataset, update_dataset, insert_times)

        dm.store.mapper.assert_called_once_with()
        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times)
        dm.calculate_update_time_ranges.assert_called_once_with(original_dataset, insert_dataset)

        insert_dataset.sel.assert_has_calls(
            [mock.call(time=slice("breakfast", "second breakfast")), mock.call(time=slice("dusk", "dawn"))]
        )
        dm.to_zarr.assert_has_calls(
            [
                mock.call(slice1.drop_vars.return_value, mapper, region={"time": slice("the shire", "mordor")}),
                mock.call(slice2.drop_vars.return_value, mapper, region={"time": slice("vegas", "atlantic city")}),
            ]
        )

        assert insert_dataset.attrs == {"update_is_append_only": False}
        assert dm.dataset_hash == "42"

    @staticmethod
    def test_insert_into_dataset_not_ipld_dry_run(manager_class):
        dm = manager_class()
        dm.dataset_hash = "browns"
        dm.dry_run = True
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        original_dataset = object()
        update_dataset = object()
        insert_times = object()

        slice1 = mock.Mock()
        slice2 = mock.Mock()
        insert_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        insert_dataset.attrs = {}
        insert_dataset.sel.side_effect = [slice1, slice2]

        dm.calculate_update_time_ranges.return_value = (
            (("breakfast", "second breakfast"), ("dusk", "dawn")),
            (("the shire", "mordor"), ("vegas", "atlantic city")),
        )

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = 42

        dm.insert_into_dataset(original_dataset, update_dataset, insert_times)

        dm.store.mapper.assert_called_once_with()
        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times)
        dm.calculate_update_time_ranges.assert_called_once_with(original_dataset, insert_dataset)

        insert_dataset.sel.assert_has_calls(
            [mock.call(time=slice("breakfast", "second breakfast")), mock.call(time=slice("dusk", "dawn"))]
        )
        dm.to_zarr.assert_has_calls(
            [
                mock.call(slice1.drop_vars.return_value, mapper, region={"time": slice("the shire", "mordor")}),
                mock.call(slice2.drop_vars.return_value, mapper, region={"time": slice("vegas", "atlantic city")}),
            ]
        )

        assert insert_dataset.attrs == {"update_is_append_only": False}
        assert dm.dataset_hash == "browns"

    @staticmethod
    def test_append_to_dataset_ipld(manager_class):
        dm = manager_class()
        dm.dataset_hash = "browns"
        dm.store = mock.Mock(spec=store.IPLD)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        update_dataset = object()
        insert_times = object()

        append_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        append_dataset.attrs = {}

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = 42

        dm.append_to_dataset(update_dataset, insert_times)

        dm.store.mapper.assert_called_once_with()
        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times)

        dm.to_zarr.assert_called_once_with(append_dataset, mapper, consolidated=True, append_dim="time")

        assert append_dataset.attrs == {"update_is_append_only": True}
        assert dm.dataset_hash == "42"

    @staticmethod
    def test_append_to_dataset_not_ipld_dry_run(manager_class):
        dm = manager_class()
        dm.dataset_hash = "browns"
        dm.dry_run = True
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.prep_update_dataset = mock.Mock()
        dm.calculate_update_time_ranges = mock.Mock()
        dm.to_zarr = mock.Mock()

        update_dataset = object()
        insert_times = object()

        append_dataset = dm.prep_update_dataset.return_value = mock.MagicMock()
        append_dataset.attrs = {}

        mapper = dm.store.mapper.return_value
        mapper.freeze.return_value = 42

        dm.append_to_dataset(update_dataset, insert_times)

        dm.store.mapper.assert_called_once_with()
        dm.prep_update_dataset.assert_called_once_with(update_dataset, insert_times)

        dm.to_zarr.assert_called_once_with(append_dataset, mapper, consolidated=True, append_dim="time")

        assert append_dataset.attrs == {"update_is_append_only": True}
        assert dm.dataset_hash == "browns"

    @staticmethod
    def test_prep_update_dataset(manager_class, fake_complex_update_dataset):
        # Give the transpose call in prep_update_dataset something to do
        dataset = fake_complex_update_dataset.transpose("longitude", "latitude", "time")
        assert dataset.data.dims == ("longitude", "latitude", "time")
        time_values = numpy.arange(
            numpy.datetime64("2022-02-01T00:00:00.000000000"),
            numpy.datetime64("2022-03-09T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        dm = manager_class()
        dm.set_zarr_metadata = lambda x: x
        dm.requested_dask_chunks = {"time": 5, "latitude": 4, "longitude": 4}

        assert len(dataset.time) > len(time_values)

        dataset = dm.prep_update_dataset(dataset, time_values)

        assert numpy.array_equal(dataset.time, time_values)
        dataset.chunks["time"][0] == 5
        dataset.chunks["latitude"][0] == 4
        dataset.chunks["longitude"][0] == 4
        assert dataset.data.dims == ("time", "latitude", "longitude")

    @staticmethod
    def test_prep_update_dataset_no_time_dimension(manager_class, fake_complex_update_dataset):
        # Give the transpose call in prep_update_dataset something to do
        dataset = fake_complex_update_dataset.transpose("longitude", "latitude", "time")
        assert dataset.data.dims == ("longitude", "latitude", "time")
        dataset = dataset.sel(time=[numpy.datetime64("2022-02-01T00:00:00.000000000")]).squeeze()
        time_values = numpy.arange(
            numpy.datetime64("2022-02-01T00:00:00.000000000"),
            numpy.datetime64("2022-03-09T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        dm = manager_class()
        dm.set_zarr_metadata = lambda x: x
        dm.requested_dask_chunks = {"time": 5, "latitude": 4, "longitude": 4}

        assert "time" not in dataset.dims

        dataset = dm.prep_update_dataset(dataset, time_values)

        assert numpy.array_equal(dataset.time, time_values[:1])
        dataset.chunks["time"][0] == 5
        dataset.chunks["latitude"][0] == 4
        dataset.chunks["longitude"][0] == 4
        assert dataset.data.dims == ("time", "latitude", "longitude")

    @staticmethod
    def test_prep_update_dataset_dask_chunks_not_full_size(manager_class, fake_complex_update_dataset):
        """
        Regression test to ensure that prep_update_datset always chunks the dataset to have full size
        full Dask chunks, preventing mismatches between Zarr and Dask chunks that cause Xarray errors
        """
        # Give the transpose call in prep_update_dataset something to do
        dataset = fake_complex_update_dataset.transpose("longitude", "latitude", "time")
        assert dataset.data.dims == ("longitude", "latitude", "time")
        dataset = dataset.chunk({"time": 3, "latitude": 1, "longitude": 1})
        time_values = numpy.arange(
            numpy.datetime64("2022-02-01T00:00:00.000000000"),
            numpy.datetime64("2022-03-09T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        dm = manager_class()
        dm.set_zarr_metadata = lambda x: x
        dm.requested_dask_chunks = {"time": 5, "latitude": 4, "longitude": 4}

        assert len(dataset.time) > len(time_values)
        dataset = dm.prep_update_dataset(dataset, time_values)

        assert numpy.array_equal(dataset.time, time_values)
        assert dataset.chunks["time"][0] == 5
        assert dataset.chunks["latitude"][0] == 4
        assert dataset.chunks["longitude"][0] == 4
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
        assert append_size == numpy.timedelta64(35, "D")

    @staticmethod
    def test_preparse_quality_check(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_vars(fake_original_dataset)
        dm.pre_parse_quality_check(fake_original_dataset)

        dm.check_random_values.assert_called_once_with(fake_original_dataset)

    @staticmethod
    def test_preparse_quality_check_short_dataset(manager_class, single_time_instant_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_vars(single_time_instant_dataset)
        dm.pre_parse_quality_check(single_time_instant_dataset)

        dm.check_random_values.assert_called_once_with(single_time_instant_dataset)

    @staticmethod
    def test_preparse_quality_check_noncontiguous_time(manager_class, fake_original_dataset):
        drop_times = fake_original_dataset.time[5:10]
        dataset = fake_original_dataset.drop_sel(time=drop_times)
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_vars(dataset)

        with pytest.raises(IndexError):
            dm.pre_parse_quality_check(dataset)

    @staticmethod
    def test_preparse_quality_check_bad_dtype(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.check_random_values = mock.Mock()
        dm.encode_vars(fake_original_dataset)
        fake_original_dataset.data.encoding["dtype"] = "thewrongtype"

        with pytest.raises(TypeError):
            dm.pre_parse_quality_check(fake_original_dataset)

    @staticmethod
    def test_check_random_values_all_ok(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.EXTREME_VALUES_BY_UNIT = {"parsecs": (-10, 10)}
        dm.pre_chunk_dataset = fake_original_dataset
        dm.encode_vars(fake_original_dataset)
        dm.check_random_values(fake_original_dataset)

    @staticmethod
    def test_check_random_values_NaN(manager_class, fake_original_dataset):
        fake_original_dataset.data.values[:] = numpy.nan
        dm = manager_class()
        dm.pre_chunk_dataset = fake_original_dataset
        dm.encode_vars(fake_original_dataset)
        with pytest.raises(ValueError):
            dm.check_random_values(fake_original_dataset)

    @staticmethod
    def test_check_random_values_nonsense_value(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.pre_chunk_dataset = fake_original_dataset
        dm.encode_vars(fake_original_dataset)

        fake_original_dataset["data"].encoding["units"] = "K"
        fake_original_dataset.data.values[:] = 1_000_000  # hot

        with pytest.raises(ValueError):
            dm.check_random_values(fake_original_dataset)

    @staticmethod
    def test_check_random_values_time_dimension_only(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.pre_chunk_dataset = fake_original_dataset
        dataset = fake_original_dataset.drop(dm._standard_dims_except(dm.time_dim))
        dm.encode_vars(dataset)
        dm.check_random_values(dataset)

    @staticmethod
    def test_check_random_values_NaNs_are_allowed(manager_class, fake_original_dataset):
        fake_original_dataset.data.values[:] = numpy.nan
        dm = manager_class()
        dm.pre_chunk_dataset = fake_original_dataset
        dm.encode_vars(fake_original_dataset)
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
        start = numpy.datetime64("2000-01-01T00:00:00")
        delta = numpy.datetime64("2000-01-01T01:00:00") - start
        times = [start + i * delta for i in range(10)]

        dm = manager_class()
        assert dm.are_times_in_expected_order(times, delta) is True

    @staticmethod
    def test_are_times_in_expected_order_regular_cadence_not_ok(manager_class):
        start = numpy.datetime64("2000-01-01T00:00:00")
        delta = numpy.datetime64("2000-01-01T01:00:00") - start
        times = [start + i * delta for i in range(10)] + [start + delta * 20]

        dm = manager_class()
        assert dm.are_times_in_expected_order(times, delta) is False

    @staticmethod
    def test_are_times_in_expected_order_irregular_cadence_ok(manager_class):
        start = numpy.datetime64("2000-01-01T00:00:00")
        delta = numpy.datetime64("2000-01-01T01:00:00") - start
        times = [start + i * delta * 1.05 for i in range(10)]

        class MyManager(manager_class):
            update_cadence_bounds = (delta / 2, delta * 2)

        dm = MyManager()
        assert dm.are_times_in_expected_order(times, delta) is True

    @staticmethod
    def test_are_times_in_expected_order_irregular_cadence_not_ok(manager_class):
        start = numpy.datetime64("2000-01-01T00:00:00")
        delta = numpy.datetime64("2000-01-01T01:00:00") - start
        times = [start + i * delta * 2.5 for i in range(10)]

        class MyManager(manager_class):
            update_cadence_bounds = (delta / 2, delta * 2)

        dm = MyManager()
        assert dm.are_times_in_expected_order(times, delta) is False

    @staticmethod
    def test_post_parse_quality_check(manager_class, mocker):
        shuffled_coords = mocker.patch("gridded_etl_tools.utils.publish.shuffled_coords")
        # Something about setting this return value is confusing coverage
        shuffled_coords.return_value = ({"a": i} for i in range(1000))  # pragma NO BRANCH

        dm = manager_class()
        dm.get_prod_update_ds = mock.Mock()
        prod_ds = dm.get_prod_update_ds.return_value

        def check_written_value(coords, dataset, threshold):
            assert dataset is prod_ds
            assert threshold == 10e-5

        dm.check_written_value = check_written_value

        dm.post_parse_quality_check()

        shuffled_coords.assert_called_once_with(prod_ds)
        dm.get_prod_update_ds.assert_called_once_with()

    @staticmethod
    def test_post_parse_quality_check_skip_it(manager_class, mocker):
        shuffled_coords = mocker.patch("gridded_etl_tools.utils.publish.shuffled_coords")
        # Something about setting this return value is confusing coverage
        shuffled_coords.return_value = ({"a": i} for i in range(1000))  # pragma NO BRANCH

        dm = manager_class()
        dm.get_prod_update_ds = mock.Mock()
        dm.skip_post_parse_qc = True
        dm.check_written_value = mock.Mock()

        dm.post_parse_quality_check()

        shuffled_coords.assert_not_called()
        dm.get_prod_update_ds.assert_not_called()
        dm.check_written_value.assert_not_called()

    @staticmethod
    def test_post_parse_quality_check_timeout(manager_class, mocker):
        shuffled_coords = mocker.patch("gridded_etl_tools.utils.publish.shuffled_coords")
        # Something about setting this return value is confusing coverage
        shuffled_coords.return_value = ({"a": i} for i in range(1000))  # pragma NO BRANCH
        time = mocker.patch("gridded_etl_tools.utils.publish.time")
        time.perf_counter = mock.Mock(side_effect=[0, 1, 2, 5000, 5001])

        dm = manager_class()
        dm.get_prod_update_ds = mock.Mock()
        prod_ds = dm.get_prod_update_ds.return_value

        def check_written_value(coords, dataset, threshold):
            assert dataset is prod_ds
            assert threshold == 10e-5
            return coords["a"] % 3 == 0

        dm.check_written_value = check_written_value

        dm.post_parse_quality_check()

        shuffled_coords.assert_called_once_with(prod_ds)
        dm.get_prod_update_ds.assert_called_once_with()

    @staticmethod
    def test_get_prod_update_ds(manager_class, fake_original_dataset):
        fake_original_dataset.attrs["update_date_range"] = ("2021120100", "2022010100")

        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.store.dataset.return_value = fake_original_dataset

        dataset = dm.get_prod_update_ds()
        assert dataset["time"].values[0] == numpy.datetime64("2021-12-01T00:00:00.000000000")
        assert dataset["time"].values[-1] == numpy.datetime64("2022-01-01T00:00:00.000000000")

    @staticmethod
    def test_check_written_value(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.get_original_ds = mock.Mock(return_value=fake_original_dataset)
        prod_ds = fake_original_dataset.copy()
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(prod_ds.dims, coord_indices)}

        dm.check_written_value(check_coords, prod_ds)

    @staticmethod
    def test_check_written_value_value_is_out_of_bounds(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.get_original_ds = mock.Mock(return_value=fake_original_dataset)
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(prod_ds.dims, coord_indices)}

        prod_ds.data[coord_indices] += 10e-4
        with pytest.raises(ValueError):
            dm.check_written_value(check_coords, prod_ds)

    @staticmethod
    def test_check_written_value_override_threshold(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.get_original_ds = mock.Mock(return_value=fake_original_dataset)
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(prod_ds.dims, coord_indices)}

        prod_ds.data[coord_indices] += 10e-4
        dm.check_written_value(check_coords, prod_ds, threshold=10e-3)

    @staticmethod
    def test_check_written_value_value_one_infinity(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.get_original_ds = mock.Mock(return_value=fake_original_dataset)
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(prod_ds.dims, coord_indices)}

        prod_ds.data[coord_indices] = numpy.inf
        with pytest.raises(ValueError):
            dm.check_written_value(check_coords, prod_ds, threshold=numpy.inf)

        prod_ds.data[coord_indices] = fake_original_dataset.data[coord_indices]
        fake_original_dataset.data[coord_indices] = numpy.inf
        with pytest.raises(ValueError):
            dm.check_written_value(check_coords, prod_ds, threshold=numpy.inf)

    @staticmethod
    def test_check_two_infinities_ish(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.get_original_ds = mock.Mock(return_value=fake_original_dataset)
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(prod_ds.dims, coord_indices)}

        fake_original_dataset.data[coord_indices] = 2e100
        prod_ds.data[coord_indices] = numpy.inf
        dm.check_written_value(check_coords, prod_ds)

    @staticmethod
    def test_check_written_value_value_one_nan(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.get_original_ds = mock.Mock(return_value=fake_original_dataset)
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(prod_ds.dims, coord_indices)}

        prod_ds.data[coord_indices] = numpy.nan
        with pytest.raises(ValueError):
            dm.check_written_value(check_coords, prod_ds, threshold=numpy.inf)

        prod_ds.data[coord_indices] = fake_original_dataset.data[coord_indices]
        fake_original_dataset.data[coord_indices] = numpy.nan
        with pytest.raises(ValueError):
            dm.check_written_value(check_coords, prod_ds, threshold=numpy.inf)

    @staticmethod
    def test_check_written_value_value_two_nans(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.get_original_ds = mock.Mock(return_value=fake_original_dataset)
        prod_ds = fake_original_dataset.copy(deep=True)
        coord_indices = (42, 2, 3)
        check_coords = {dim: prod_ds[dim].values[i] for dim, i in zip(prod_ds.dims, coord_indices)}

        prod_ds.data[coord_indices] = numpy.nan
        fake_original_dataset.data[coord_indices] = numpy.nan
        dm.check_written_value(check_coords, prod_ds, threshold=numpy.inf)

    @staticmethod
    def test_get_original_ds(manager_class, dataset_at):
        timestamps = numpy.arange(
            numpy.datetime64("2021-10-16T00:00:00.000000000"),
            numpy.datetime64("2021-10-26T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        orig_datasets = [dataset_at(timestamp) for timestamp in timestamps]

        def raw_file_to_dataset(path):
            assert path.startswith("test_path_")
            index = int(path[10:])
            return orig_datasets[index]

        def reformat_orig_ds(ds, path):
            ds.attrs["reformat_args"] = (ds, path)
            return ds

        dm = manager_class()
        dm.raw_file_to_dataset = raw_file_to_dataset
        dm.reformat_orig_ds = reformat_orig_ds
        dm.input_files = mock.Mock(return_value=[f"test_path_{i:02d}" for i in range(10)])

        for i in range(10):
            dataset = dm.get_original_ds({"x": "nobody", "y": "cares", "time": timestamps[i]})
            assert dataset is orig_datasets[i]
            assert dataset.attrs["reformat_args"] == (dataset, f"test_path_{i:02d}")

        with pytest.raises(FileNotFoundError):
            dm.get_original_ds({"x": "nobody", "y": "cares", "time": timestamps[0] - numpy.timedelta64(1, "[D]")})

        with pytest.raises(FileNotFoundError):
            dm.get_original_ds({"x": "nobody", "y": "cares", "time": timestamps[9] + numpy.timedelta64(1, "[D]")})

    @staticmethod
    def test_get_original_ds_with_step(manager_class, hindcast_dataset_at):
        timestamps = numpy.arange(
            numpy.datetime64("2021-10-16T00:00:00.000000000"),
            numpy.datetime64("2021-10-26T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        steps = timestamps + numpy.timedelta64(4, "[h]")
        orig_datasets = [hindcast_dataset_at(timestamp) for timestamp in timestamps]

        def raw_file_to_dataset(path):
            assert path.startswith("test_path_")
            index = int(path[10:12])
            return orig_datasets[index]

        def reformat_orig_ds(ds, path):
            ds.attrs["reformat_args"] = (ds, path)
            return ds

        def path_for(i: int) -> str:
            date_str = pd.Timestamp(timestamps[i]).to_pydatetime().date().isoformat()
            return f"test_path_{i:02d}_{date_str}"

        dm = manager_class()
        dm.raw_file_to_dataset = raw_file_to_dataset
        dm.reformat_orig_ds = reformat_orig_ds
        dm.input_files = mock.Mock(return_value=[path_for(i) for i in range(10)])
        dm.time_dim = "hindcast_reference_time"

        for i in range(10):
            dataset = dm.get_original_ds(
                {"x": "nobody", "y": "cares", "hindcast_reference_time": timestamps[i], "step": steps[i]}
            )
            assert dataset is orig_datasets[i]
            assert dataset.attrs["reformat_args"] == (dataset, path_for(i))

    @staticmethod
    def test_get_original_ds_missing_time(manager_class, hindcast_dataset_at):
        timestamps = numpy.arange(
            numpy.datetime64("2021-10-16T00:00:00.000000000"),
            numpy.datetime64("2021-10-26T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        orig_datasets = [hindcast_dataset_at(timestamp) for timestamp in timestamps]

        def raw_file_to_dataset(path):
            assert path.startswith("test_path_")
            index = int(path[10:])
            return orig_datasets[index]

        dm = manager_class()
        dm.raw_file_to_dataset = raw_file_to_dataset
        dm.reformat_orig_ds = mock.Mock()
        dm.input_files = mock.Mock(return_value=[f"test_path_{i:02d}" for i in range(10)])

        with pytest.raises(ValueError):
            dm.get_original_ds({"x": "nobody", "y": "cares", "time": timestamps[0]})

        dm.reformat_orig_ds.assert_not_called()

    @staticmethod
    def test_get_original_ds_dimensionless_time(manager_class, dataset_at):
        timestamps = numpy.arange(
            numpy.datetime64("2021-10-16T00:00:00.000000000"),
            numpy.datetime64("2021-10-26T00:00:00.000000000"),
            numpy.timedelta64(1, "[D]"),
        )
        orig_datasets = [dataset_at(timestamp).squeeze() for timestamp in timestamps]

        def raw_file_to_dataset(path):
            assert path.startswith("test_path_")
            index = int(path[10:])
            return orig_datasets[index]

        def reformat_orig_ds(ds, path):
            ds.attrs["reformat_args"] = (ds, path)
            return ds

        dm = manager_class()
        dm.raw_file_to_dataset = raw_file_to_dataset
        dm.reformat_orig_ds = reformat_orig_ds
        dm.input_files = mock.Mock(return_value=[f"test_path_{i:02d}" for i in range(10)])

        for i in range(10):
            dataset = dm.get_original_ds({"x": "nobody", "y": "cares", "time": timestamps[i]})
            assert dataset is orig_datasets[i]
            assert dataset.attrs["reformat_args"] == (dataset, f"test_path_{i:02d}")

        with pytest.raises(FileNotFoundError):
            dm.get_original_ds({"x": "nobody", "y": "cares", "time": timestamps[0] - numpy.timedelta64(1, "[D]")})

        with pytest.raises(FileNotFoundError):
            dm.get_original_ds({"x": "nobody", "y": "cares", "time": timestamps[9] + numpy.timedelta64(1, "[D]")})

    @staticmethod
    def test_raw_file_to_dataset_file(manager_class, mocker):
        xr = mocker.patch("gridded_etl_tools.utils.publish.xr")
        dm = manager_class()
        dm.protocol = "file"
        assert dm.raw_file_to_dataset("some/path") is xr.open_dataset.return_value
        xr.open_dataset.assert_called_once_with("some/path")

    @staticmethod
    def test_raw_file_to_dataset_s3(manager_class):
        dm = manager_class()
        dm.load_dataset_from_disk = mock.Mock()
        dm.protocol = "s3"
        dm.use_local_zarr_jsons = True
        assert dm.raw_file_to_dataset(pathlib.PosixPath("some/path")) is dm.load_dataset_from_disk.return_value
        dm.load_dataset_from_disk.assert_called_once_with(zarr_json_path="some/path")

    @staticmethod
    def test_raw_file_to_dataset_s3_no_local_zarr_json(manager_class):
        dm = manager_class()
        dm.zarr_json_to_dataset = mock.Mock()
        dm.protocol = "s3"
        dm.use_local_zarr_jsons = False
        with pytest.raises(ValueError):
            dm.raw_file_to_dataset(pathlib.PosixPath("some/path"))

    @staticmethod
    def test_raw_file_to_dataset_bad_protocol(manager_class):
        dm = manager_class()
        dm.protocol = "nopenoway"
        with pytest.raises(ValueError):
            dm.raw_file_to_dataset("some/path")

    @staticmethod
    def test_reformat_orig_ds(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.postprocess_zarr = mock.Mock()
        dm.rename_data_variable = mock.Mock()
        dm.reformat_orig_ds(fake_original_dataset, "hi/mom.zarr")

        dm.postprocess_zarr.assert_not_called()
        dm.rename_data_variable.assert_called_once_with(fake_original_dataset)

    @staticmethod
    def test_reformat_orig_ds_file_protocol(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.postprocess_zarr = mock.Mock()
        dm.rename_data_variable = mock.Mock()
        dm.protocol = "file"
        dm.reformat_orig_ds(fake_original_dataset, "hi/mom.zarr")

        dm.postprocess_zarr.assert_called_once_with(fake_original_dataset)
        dm.rename_data_variable.assert_called_once_with(dm.postprocess_zarr.return_value)

    @staticmethod
    def test_reformat_orig_ds_single_time_instant(manager_class, single_time_instant_dataset):
        dm = manager_class()
        dm.postprocess_zarr = mock.Mock()
        dm.rename_data_variable = mock.Mock()
        orig_dataset = single_time_instant_dataset.squeeze()
        dm.reformat_orig_ds(orig_dataset, "hi/mom.zarr")

        dataset = dm.rename_data_variable.call_args[0][0]
        assert "time" in dataset.dims

        dm.postprocess_zarr.assert_not_called()
        dm.rename_data_variable.assert_called_once_with(dataset)

    @staticmethod
    def test_reformat_orig_ds_missing_step_dimension(manager_class, fake_original_dataset):
        dm = manager_class()
        dm.postprocess_zarr = mock.Mock()
        dm.rename_data_variable = mock.Mock()
        dm.standard_dims += ["step"]
        dm.reformat_orig_ds(fake_original_dataset, "hi/mom-2022-07-04.zarr")

        dataset = dm.rename_data_variable.call_args[0][0]
        assert "step" in dataset
        assert "step" in dataset.dims
        assert dataset.step[0] == numpy.datetime64("2022-07-04T00:00:00.000000000")

        dm.postprocess_zarr.assert_not_called()
        dm.rename_data_variable.assert_called_once_with(dataset)

    @staticmethod
    def test_convert_standard_raw_times_to_comparable_times(manager_class, fake_original_dataset):
        dm = manager_class()
        time_vals = dm.convert_raw_times_to_comparable_times(fake_original_dataset, time_dim="time")
        assert isinstance(time_vals, (list, numpy.ndarray))
        assert len(time_vals)
        assert numpy.datetime64("1900-01-01") < time_vals[0]

    @staticmethod
    def test_convert_weird_raw_times_to_comparable_times(mocker, manager_class, single_esoteric_time_instant_dataset):
        dm = manager_class()
        mocker.patch(
            "gridded_etl_tools.utils.publish.Publish.convert_raw_times_to_numpy_times",
            convert_raw_times_to_numpy_times,
        )
        time_vals = dm.convert_raw_times_to_comparable_times(single_esoteric_time_instant_dataset, time_dim="time")
        assert isinstance(time_vals, (list, numpy.ndarray))
        assert len(time_vals)
        assert numpy.datetime64("1900-01-01") < time_vals[0]

    @staticmethod
    def test_convert_raw_times_to_numpy_times(manager_class, fake_original_dataset):
        dm = manager_class()
        time_coords = fake_original_dataset.time.values
        numpy.testing.assert_equal(dm.convert_raw_times_to_numpy_times(fake_original_dataset.time), time_coords)


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
