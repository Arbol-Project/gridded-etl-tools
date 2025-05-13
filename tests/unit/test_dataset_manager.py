import datetime
import logging
import unittest

import pytest

from gridded_etl_tools import dataset_manager
from gridded_etl_tools.utils import encryption, store
from gridded_etl_tools.utils.time import TimeSpan

from .conftest import John, Paul, George, Ringo, RingoDaily


class TestDatasetManager:
    @staticmethod
    def test_constructor_defaults(mocker, manager_class):
        mocker.patch("gridded_etl_tools.dataset_manager.psutil.virtual_memory")
        dataset_manager.psutil.virtual_memory.return_value.total = 64_000_000_000

        mocker.patch("gridded_etl_tools.dataset_manager.multiprocessing.cpu_count")
        dataset_manager.multiprocessing.cpu_count.return_value = 100

        mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.log_to_console")
        dm = manager_class()
        dm.log_to_console.assert_called_once_with()
        assert dm.dask_num_threads == 8

    @staticmethod
    def test_constructor_most_simple_args(mocker, manager_class):
        secret_key = encryption.generate_encryption_key()

        mocker.patch("gridded_etl_tools.dataset_manager.psutil.virtual_memory")
        dataset_manager.psutil.virtual_memory.return_value.total = 64_000_000_000

        mocker.patch("gridded_etl_tools.dataset_manager.multiprocessing.cpu_count")
        dataset_manager.multiprocessing.cpu_count.return_value = 100

        mocker.patch("gridded_etl_tools.dataset_manager.logging.getLogger")
        dataset_manager.logging.getLogger.return_value.level = logging.ERROR

        mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.log_to_console")

        dm = manager_class(
            requested_dask_chunks="requested_dask_chunks",
            requested_zarr_chunks="requested_zarr_chunks",
            requested_ipfs_chunker="chunky chunker",
            rebuild_requested=True,
            custom_output_path="output/over/here",
            custom_latest_hash="omghash!",
            custom_input_path="input/over/here",
            console_log=False,
            global_log_level=logging.WARN,
            allow_overwrite=True,
            dask_dashboard_address="123 main st",
            dask_cpu_mem_target_ratio=1 / 16,
            use_local_zarr_jsons=True,
            skip_prepare_input_files=True,
            encryption_key=secret_key,
            use_compression=False,
        )

        assert dm.requested_dask_chunks == "requested_dask_chunks"
        assert dm.requested_zarr_chunks == "requested_zarr_chunks"
        assert dm.requested_ipfs_chunker == "chunky chunker"
        assert dm.rebuild_requested is True
        assert dm.custom_output_path == "output/over/here"
        assert dm.latest_hash() == "omghash!"
        dm.log_to_console.assert_not_called()
        dataset_manager.logging.getLogger.return_value.setLevel.assert_called_once_with(logging.WARN)
        assert dm.allow_overwrite is True
        assert dm.dask_dashboard_address == "123 main st"
        assert dm.dask_num_threads == 4
        assert dm.use_local_zarr_jsons is True
        assert dm.skip_prepare_input_files is True
        assert dm.encryption_key == encryption._hash(bytes.fromhex(secret_key))
        assert dm.use_compression is False

        assert isinstance(dm.store, store.Local)

    @staticmethod
    def test_constructor_ipld_store(manager_class):
        dm = manager_class(store="ipld")
        assert isinstance(dm.store, dataset_manager.IPLD)

    @staticmethod
    def test_constructor_s3_store(manager_class):
        dm = manager_class(store="s3", s3_bucket_name="mop water")
        assert isinstance(dm.store, dataset_manager.S3)
        assert dm.store.bucket == "mop water"

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
        assert dm.new_files == []

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
        dm.populate_metadata = unittest.mock.Mock()
        dm.prepare_input_files = unittest.mock.Mock()
        dm.create_zarr_json = unittest.mock.Mock()
        dm.transform_data_on_disk()

        dm.populate_metadata.assert_called_once_with()
        dm.prepare_input_files.assert_called_once_with()
        dm.create_zarr_json.assert_called_once_with()

    @staticmethod
    def test_transform_dataset_in_memory_new_initial(mocker, manager_class):
        dm = manager_class(skip_prepare_input_files=True)
        dm.store = mocker.Mock(spec=store.IPLD, has_existing=False)
        dm.update_ds_transform = unittest.mock.Mock()
        dm.initial_ds_transform = unittest.mock.Mock()
        dm.transform_dataset_in_memory()

        dm.update_ds_transform.assert_not_called()
        dm.initial_ds_transform.assert_called_once_with()

    @staticmethod
    def test_transform_dataset_in_memory_update(mocker, manager_class):
        dm = manager_class(skip_prepare_input_files=True, rebuild_requested=False)
        dm.store = mocker.Mock(spec=store.IPLD, has_existing=True)
        dm.update_ds_transform = unittest.mock.Mock()
        dm.initial_ds_transform = unittest.mock.Mock()
        dm.transform_dataset_in_memory()

        dm.update_ds_transform.assert_called_once_with()
        dm.initial_ds_transform.assert_not_called()

    @staticmethod
    def test_transform_dataset_in_memory_update_rebuild_initial(mocker, manager_class):
        dm = manager_class(skip_prepare_input_files=True, rebuild_requested=True, allow_overwrite=True)
        dm.store = mocker.Mock(spec=store.IPLD, has_existing=True)
        dm.update_ds_transform = unittest.mock.Mock()
        dm.initial_ds_transform = unittest.mock.Mock()
        dm.transform_dataset_in_memory()

        dm.update_ds_transform.assert_not_called()
        dm.initial_ds_transform.assert_called_once_with()

    @staticmethod
    def test_transform_dataset_in_memory_update_rebuild_initial_but_overwrite_not_allowed(mocker, manager_class):
        dm = manager_class(skip_prepare_input_files=True, rebuild_requested=True, allow_overwrite=False)
        dm.store = mocker.Mock(spec=store.IPLD, has_existing=True)
        dm.update_ds_transform = unittest.mock.Mock()
        dm.initial_ds_transform = unittest.mock.Mock()
        with pytest.raises(RuntimeError):
            dm.transform_dataset_in_memory()

    @staticmethod
    def test_transform_data_on_disk_skip_prepare_input_files(manager_class):
        dm = manager_class(skip_prepare_input_files=True)
        dm.populate_metadata = unittest.mock.Mock()
        dm.prepare_input_files = unittest.mock.Mock()
        dm.create_zarr_json = unittest.mock.Mock()
        dm.transform_data_on_disk()

        dm.populate_metadata.assert_called_once_with()
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
        assert set(manager_class.get_subclasses()) == {John, Paul, George, Ringo, RingoDaily}
        assert set(John.get_subclasses()) == {George, Ringo}
        assert set(Paul.get_subclasses()) == {George, Ringo}
        assert set(George.get_subclasses()) == {Ringo}
        assert set(Ringo.get_subclasses()) == set()

    @staticmethod
    def test_get_subclass(manager_class):
        assert manager_class.get_subclass("John") is John
        assert manager_class.get_subclass("Paul") is Paul
        assert manager_class.get_subclass("George") is George
        assert manager_class.get_subclass("Ringo") is Ringo

        assert John.get_subclass("George") is George
        assert John.get_subclass("Ringo") is Ringo

        assert Paul.get_subclass("George") is George
        assert Paul.get_subclass("Ringo") is Ringo

        assert George.get_subclass("Ringo") is Ringo

    @staticmethod
    def test_get_subclass_time_resolution(manager_class):
        assert manager_class.get_subclass("Ringo") is Ringo
        assert manager_class.get_subclass("Ringo", time_resolution=TimeSpan.from_string("daily")) is RingoDaily
        assert George.get_subclass("Ringo") is Ringo

    @staticmethod
    def test_get_subclass_not_found():
        with pytest.warns(UserWarning, match="John"):
            assert Ringo.get_subclass("John") is None

        with pytest.warns(UserWarning, match="Pete"):
            assert Ringo.get_subclass("Pete") is None
