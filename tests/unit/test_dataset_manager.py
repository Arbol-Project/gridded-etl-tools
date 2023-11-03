import datetime
import logging
import unittest

import pytest

from gridded_etl_tools import dataset_manager
from gridded_etl_tools.utils import encryption, store


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
        assert dm.overwrite_allowed is True
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
        dm.populate_metadata = unittest.mock.Mock()
        dm.prepare_input_files = unittest.mock.Mock()
        dm.create_zarr_json = unittest.mock.Mock()
        dm.transform()

        dm.populate_metadata.assert_called_once_with()
        dm.prepare_input_files.assert_called_once_with()
        dm.create_zarr_json.assert_called_once_with()

    @staticmethod
    def test_transform_skip_prepare_input_files(manager_class):
        dm = manager_class(skip_prepare_input_files=True)
        dm.populate_metadata = unittest.mock.Mock()
        dm.prepare_input_files = unittest.mock.Mock()
        dm.create_zarr_json = unittest.mock.Mock()
        dm.transform()

        dm.populate_metadata.assert_called_once_with()
        dm.prepare_input_files.assert_not_called()
        dm.create_zarr_json.assert_called_once_with()
