import datetime
import logging
import unittest
import pathlib

import pytest

import gridded_etl_tools

from gridded_etl_tools import dataset_manager
from gridded_etl_tools.utils import encryption, store
from gridded_etl_tools.utils.time import TimeSpan

from .conftest import Beatles, John, Paul, George, Ringo, RingoDaily, Pete, Stuart, PeteBest, StuartSutcliffe


class TestDatasetManager:
    @staticmethod
    def test_constructor_defaults(mocker, manager_class):
        mocker.patch("gridded_etl_tools.dataset_manager.psutil.virtual_memory")
        dataset_manager.psutil.virtual_memory.return_value.total = 64_000_000_000

        mocker.patch("gridded_etl_tools.dataset_manager.multiprocessing.cpu_count")
        dataset_manager.multiprocessing.cpu_count.return_value = 100

        mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.init_logging")
        dm = manager_class()
        dm.init_logging.assert_called_once()
        assert dm.dask_num_threads == 8

    @staticmethod
    def test_constructor_most_simple_args(mocker, manager_class):
        secret_key = encryption.generate_encryption_key()

        mocker.patch("gridded_etl_tools.dataset_manager.psutil.virtual_memory")
        dataset_manager.psutil.virtual_memory.return_value.total = 64_000_000_000

        mocker.patch("gridded_etl_tools.dataset_manager.multiprocessing.cpu_count")
        dataset_manager.multiprocessing.cpu_count.return_value = 100

        mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.init_logging")

        dm = manager_class(
            requested_dask_chunks="requested_dask_chunks",
            requested_zarr_chunks="requested_zarr_chunks",
            rebuild_requested=True,
            custom_output_path=pathlib.Path("output/over/here"),
            custom_input_path=pathlib.Path("input/over/here"),
            console_log=False,
            global_log_level=logging.WARN,
            allow_overwrite=True,
            dask_dashboard_address="123 main st",
            dask_cpu_mem_target_ratio=1 / 16,
            dask_num_workers=2,
            dask_num_threads=4,
            dask_scheduler_protocol="tcp://",
            use_local_zarr_jsons=True,
            skip_prepare_input_files=True,
            encryption_key=secret_key,
            use_compression=False,
            output_zarr3=True,
            align_update_chunks=True,
        )

        assert dm.requested_dask_chunks == "requested_dask_chunks"
        assert dm.requested_zarr_chunks == "requested_zarr_chunks"
        assert dm.rebuild_requested is True
        assert dm.custom_output_path == pathlib.Path("output/over/here")
        dm.init_logging.assert_called_once_with(console_log=False, global_log_level=logging.WARN)
        assert dm.allow_overwrite is True
        assert dm.dask_dashboard_address == "123 main st"
        assert dm.dask_num_workers == 2
        assert dm.dask_num_threads == 4
        assert dm.dask_scheduler_protocol == "tcp://"
        assert dm.dask_use_process_scheduler is True
        assert dm.use_local_zarr_jsons is True
        assert dm.skip_prepare_input_files is True
        assert dm.encryption_key == encryption._hash(bytes.fromhex(secret_key))
        assert dm.use_compression is False
        assert dm.output_zarr3 is True
        assert dm.align_update_chunks is True

        assert isinstance(dm.store, store.Local)

    @staticmethod
    def test_constructor_default_sets_worker_count_to_one(mocker, manager_class):
        mocker.patch("gridded_etl_tools.dataset_manager.psutil.virtual_memory")
        dataset_manager.psutil.virtual_memory.return_value.total = 64_000_000_000

        mocker.patch("gridded_etl_tools.dataset_manager.multiprocessing.cpu_count")
        dataset_manager.multiprocessing.cpu_count.return_value = 100

        dm = manager_class()
        assert dm.dask_num_workers == 1

    @staticmethod
    def test_constructor_raises_if_only_dask_num_workers_provided(manager_class):
        with pytest.raises(ValueError):
            manager_class(dask_num_workers=2)

    @staticmethod
    def test_constructor_raises_if_only_dask_num_threads_provided(manager_class):
        with pytest.raises(ValueError):
            manager_class(dask_num_threads=4)

    @staticmethod
    def test_constructor_s3_store(manager_class):
        dm = manager_class(store="s3", s3_bucket_name="mop_water")
        assert isinstance(dm.store, dataset_manager.S3)
        assert dm.store.bucket == "mop_water"

    @staticmethod
    def test_constructor_match_existing_format(manager_class, mocker):
        # Save these functions to restore the module to initial state when done testing the DM constructor
        restore_has_existing = gridded_etl_tools.utils.store.Local.has_existing
        restore_has_v3_metadata = gridded_etl_tools.utils.store.Local.has_v3_metadata

        # Mock that the store has a v3 Zarr
        gridded_etl_tools.utils.store.Local.has_existing = True
        gridded_etl_tools.utils.store.Local.has_v3_metadata = True

        dm = manager_class(output_zarr3=True)
        with pytest.raises(RuntimeError, match="Existing data is Zarr v3, but output_zarr3 is not set."):
            dm = manager_class(output_zarr3=False)

        # Mock that the store has a v2 Zarr
        gridded_etl_tools.utils.store.Local.has_v3_metadata = False
        dm = manager_class(output_zarr3=False)
        with pytest.raises(RuntimeError, match="Existing data is not Zarr v3, but output_zarr3 is set."):
            dm = manager_class(output_zarr3=True)  # noqa: F841

        # Restore the module to initial state
        gridded_etl_tools.utils.store.Local.has_existing = restore_has_existing
        gridded_etl_tools.utils.store.Local.has_v3_metadata = restore_has_v3_metadata

    @staticmethod
    def test_constructor_bad_store(manager_class):
        with pytest.raises(ValueError):
            manager_class(store="walmart")

    @staticmethod
    def test_constructor_dask_threads_limited_by_number_of_cpus(mocker, manager_class):
        mocker.patch("gridded_etl_tools.dataset_manager.psutil.virtual_memory")
        dataset_manager.psutil.virtual_memory.return_value.total = 64_000_000_000

        mocker.patch("gridded_etl_tools.dataset_manager.multiprocessing.cpu_count")
        dataset_manager.multiprocessing.cpu_count.return_value = 4

        dm = manager_class()
        assert dm.dask_num_threads == 3

    @staticmethod
    def test_constructor_dask_threads_at_least_one(mocker, manager_class):
        mocker.patch("gridded_etl_tools.dataset_manager.psutil.virtual_memory")
        dataset_manager.psutil.virtual_memory.return_value.total = 64_000_000_000

        mocker.patch("gridded_etl_tools.dataset_manager.multiprocessing.cpu_count")
        dataset_manager.multiprocessing.cpu_count.return_value = 1

        dm = manager_class()
        assert dm.dask_num_threads == 1

    @staticmethod
    def test_init_logging(mocker, manager_class):
        mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.log_to_console")
        mocker.patch("gridded_etl_tools.dataset_manager.logging.getLogger")
        dataset_manager.logging.getLogger.return_value.level = logging.ERROR

        dm = manager_class()
        dm.log_to_console.assert_called_once()
        dm.init_logging(False, logging.WARN)
        dm.log_to_console.assert_called_once()
        dataset_manager.logging.getLogger.return_value.setLevel.assert_called_with(logging.WARN)
        assert dataset_manager.logging.getLogger.return_value.setLevel.call_count == 4

    @staticmethod
    def test__str__(manager_class):
        dm = manager_class()
        with pytest.deprecated_call():
            assert str(dm) == "DummyManager"

    @staticmethod
    def test__eq__(manager_class):
        with pytest.deprecated_call():
            assert manager_class() == manager_class()

    @staticmethod
    def test__eq__not_dataset_manager(manager_class):
        with pytest.deprecated_call():
            assert manager_class() != object()

    @staticmethod
    def test__hash__(manager_class):
        dm = manager_class()
        with pytest.deprecated_call():
            assert hash(dm) == hash("DummyManager")

    @staticmethod
    def test_extract(manager_class):
        dm = manager_class()
        dm.extract()

    @staticmethod
    def test_extract_bad_date_range(manager_class):
        dm = manager_class()
        with pytest.raises(ValueError):
            dm.extract(date_range=[datetime.datetime(1967, 10, 2, 0, 0, 0), datetime.datetime(2010, 5, 12, 0, 0, 0)])

    @staticmethod
    def test_transform(manager_class):
        dm = manager_class()
        dm.transform_data_on_disk = unittest.mock.Mock()
        dm.transform_dataset_in_memory = unittest.mock.Mock()

        dm.transform()

        dm.transform_data_on_disk.assert_called_once_with()
        dm.transform_dataset_in_memory.assert_called_once_with()

    @staticmethod
    def test_transform_data_on_disk(manager_class):
        dm = manager_class()
        dm.prepare_input_files = unittest.mock.Mock()
        dm.create_zarr_json = unittest.mock.Mock()
        dm.transform_data_on_disk()

        dm.prepare_input_files.assert_called_once_with()
        dm.create_zarr_json.assert_called_once_with()

    @staticmethod
    def test_transform_dataset_in_memory_new_initial(mocker, manager_class):
        dm = manager_class(skip_prepare_input_files=True)
        dm.store = mocker.Mock(spec=store.Local, has_existing=False)
        dm._update_ds_transform = unittest.mock.Mock()
        dm._initial_ds_transform = unittest.mock.Mock()
        dm.transform_dataset_in_memory()

        dm._update_ds_transform.assert_not_called()
        dm._initial_ds_transform.assert_called_once_with()

    @staticmethod
    def test_transform_dataset_in_memory_update(mocker, manager_class):
        dm = manager_class(skip_prepare_input_files=True, rebuild_requested=False)
        dm.store = mocker.Mock(spec=store.Local, has_existing=True)
        dm._update_ds_transform = unittest.mock.Mock()
        dm._initial_ds_transform = unittest.mock.Mock()
        dm.transform_dataset_in_memory()

        dm._update_ds_transform.assert_called_once_with()
        dm._initial_ds_transform.assert_not_called()

    @staticmethod
    def test_transform_dataset_in_memory_update_rebuild_initial(mocker, manager_class):
        dm = manager_class(skip_prepare_input_files=True, rebuild_requested=True, allow_overwrite=True)
        dm.store = mocker.Mock(spec=store.Local, has_existing=True)
        dm._update_ds_transform = unittest.mock.Mock()
        dm._initial_ds_transform = unittest.mock.Mock()
        dm.transform_dataset_in_memory()

        dm._update_ds_transform.assert_not_called()
        dm._initial_ds_transform.assert_called_once_with()

    @staticmethod
    def test_transform_dataset_in_memory_update_rebuild_initial_but_overwrite_not_allowed(mocker, manager_class):
        dm = manager_class(skip_prepare_input_files=True, rebuild_requested=True, allow_overwrite=False)
        dm.store = mocker.Mock(spec=store.Local, has_existing=True)
        dm._update_ds_transform = unittest.mock.Mock()
        dm._initial_ds_transform = unittest.mock.Mock()
        with pytest.raises(RuntimeError):
            dm.transform_dataset_in_memory()

    @staticmethod
    def test_transform_data_on_disk_skip_prepare_input_files(manager_class):
        dm = manager_class(skip_prepare_input_files=True)
        dm.prepare_input_files = unittest.mock.Mock()
        dm.create_zarr_json = unittest.mock.Mock()
        dm.transform_data_on_disk()

        dm.prepare_input_files.assert_not_called()
        dm.create_zarr_json.assert_called_once_with()

    @staticmethod
    def test_skip_post_parse_qc(manager_class):
        dm = manager_class(skip_post_parse_qc=True)
        dm.get_prod_update_ds = unittest.mock.Mock()
        dm.get_original_ds = unittest.mock.Mock()
        dm.input_files = unittest.mock.Mock()
        dm.get_random_coords = unittest.mock.Mock()
        dm.post_parse_quality_check()

        dm.get_prod_update_ds.assert_not_called()
        dm.input_files.assert_not_called()
        dm.get_original_ds.assert_not_called()
        dm.get_random_coords.assert_not_called()

    @staticmethod
    def test_get_subclasses(manager_class):
        # Test getting all subclasses from the base class
        subclasses = set(manager_class.get_subclasses())
        expected = {Beatles, John, Paul, George, Ringo, RingoDaily, Pete, Stuart, PeteBest, StuartSutcliffe}
        assert subclasses == expected

        # Test getting subclasses from Beatles
        beatles_subclasses = set(Beatles.get_subclasses())
        assert beatles_subclasses == expected - {Beatles}

        # Test getting subclasses from leaf nodes (should be empty)
        assert set(John.get_subclasses()) == set()
        assert set(Paul.get_subclasses()) == set()
        assert set(George.get_subclasses()) == set()
        assert set(Ringo.get_subclasses()) == set()
        assert set(RingoDaily.get_subclasses()) == set()
        assert set(Pete.get_subclasses()) == set()
        assert set(Stuart.get_subclasses()) == set()
        assert set(PeteBest.get_subclasses()) == set()
        assert set(StuartSutcliffe.get_subclasses()) == set()

    @staticmethod
    def test_get_subclass(manager_class):
        # Test getting leaf node classes
        assert manager_class.get_subclass("John") is John
        assert manager_class.get_subclass("Paul") is Paul
        assert manager_class.get_subclass("George") is George
        assert manager_class.get_subclass("Ringo") is Ringo
        assert manager_class.get_subclass("Pete") is Pete
        assert manager_class.get_subclass("Stuart") is Stuart
        assert manager_class.get_subclass("PeteBest") is PeteBest
        assert manager_class.get_subclass("StuartSutcliffe") is StuartSutcliffe

        # Test getting class with time resolution
        assert manager_class.get_subclass("Ringo", time_resolution="hourly") is Ringo
        assert manager_class.get_subclass("Ringo", time_resolution="daily") is RingoDaily

    @staticmethod
    def test_get_subclass_not_found(manager_class):
        # Try getting an ABC
        with pytest.warns(UserWarning, match="failed to set manager from name Beatles"):
            assert manager_class.get_subclass("Beatles") is None

        # Test getting non-existent class from leaf node
        with pytest.warns(UserWarning, match="failed to set manager from name John"):
            assert Ringo.get_subclass("John") is None

        with pytest.warns(UserWarning, match="failed to set manager from name Pete"):
            assert Ringo.get_subclass("Pete") is None

        # Test getting class with wrong time resolution
        with pytest.warns(UserWarning, match="failed to set manager from name Ringo"):
            assert Ringo.get_subclass("Ringo", time_resolution="monthly") is None

    @staticmethod
    def test_from_time_span_string(manager_class):
        # Test valid time spans
        assert manager_class.from_time_span_string("hourly") == TimeSpan.SPAN_HOURLY
        assert manager_class.from_time_span_string("daily") == TimeSpan.SPAN_DAILY
        assert manager_class.from_time_span_string("monthly") == TimeSpan.SPAN_MONTHLY
        assert manager_class.from_time_span_string("yearly") == TimeSpan.SPAN_YEARLY

        # Test case sensitivity
        assert manager_class.from_time_span_string("HOURLY") == TimeSpan.SPAN_HOURLY
        assert manager_class.from_time_span_string("Daily") == TimeSpan.SPAN_DAILY

        # Test invalid time span
        with pytest.raises(ValueError):
            manager_class.from_time_span_string("invalid_span")
