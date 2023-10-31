from unittest import mock

import numcodecs

from gridded_etl_tools import dataset_manager
from gridded_etl_tools.utils import encryption
from gridded_etl_tools.utils import store


def unimplemented(*args, **kwargs):
    raise NotImplementedError


class DummyManager(dataset_manager.DatasetManager):
    collection = unimplemented
    concat_dims = unimplemented
    extract = unimplemented
    identical_dims = unimplemented
    prepare_input_files = unimplemented
    remote_protocol = unimplemented
    static_metadata = unimplemented
    temporal_resolution = unimplemented

    unit_of_measurement = "parsecs"
    requested_zarr_chunks = {}
    time_dim = "time"
    encryption_key = None

    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(self, requested_dask_chunks=None, requested_zarr_chunks=None, *args, **kwargs):
        if requested_dask_chunks is None:
            requested_dask_chunks = {}

        if requested_zarr_chunks is None:
            requested_zarr_chunks = {}

        super().__init__("ipfshost", requested_dask_chunks, requested_zarr_chunks, *args, **kwargs)

    def data_var(self):
        return "data"


class TestMetadata:
    def test_encode_vars(self):
        dataset = mock.MagicMock()
        md = DummyManager()
        dataset = md.encode_vars(dataset)
        # TODO: check for things?

    def test_encode_vars_w_encryption_key(self):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {"filters": None}
        encryption_key = encryption.generate_encryption_key()
        md = DummyManager(encryption_key=encryption_key)
        dataset = md.encode_vars(dataset)

        filters = dataset["data"].encoding["filters"]
        assert len(filters) == 1
        assert isinstance(filters[0], encryption.EncryptionFilter)

    def test_encode_vars_w_encryption_key_and_preexisting_filter(self):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {"filters": ["SomeOtherFilter"]}
        encryption_key = encryption.generate_encryption_key()
        md = DummyManager(encryption_key=encryption_key)
        dataset = md.encode_vars(dataset)

        filters = dataset["data"].encoding["filters"]
        assert len(filters) == 2
        assert filters[0] == "SomeOtherFilter"
        assert isinstance(filters[1], encryption.EncryptionFilter)

    def test_remove_unwanted_fields_w_ipld_store(self):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {}
        md = DummyManager()
        md.store = store.IPLD(md)
        dataset = md.remove_unwanted_fields(dataset)
        assert isinstance(dataset["data"].encoding["compressor"], numcodecs.Blosc)

    def test_remove_unwanted_fields_w_ipld_store_no_compression(self):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {}
        md = DummyManager(use_compression=False)
        md.store = store.IPLD(md)
        dataset = md.remove_unwanted_fields(dataset)
        assert dataset["data"].encoding["compressor"] is None
