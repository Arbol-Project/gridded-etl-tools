from unittest import mock

import numcodecs

from gridded_etl_tools.utils import encryption
from gridded_etl_tools.utils import store


class TestMetadata:
    @staticmethod
    def test_encode_vars(manager_class):
        dataset = mock.MagicMock()
        md = manager_class()
        dataset = md.encode_vars(dataset)
        # TODO: check for things?

    @staticmethod
    def test_encode_vars_w_encryption_key(manager_class):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {"filters": None}
        encryption_key = encryption.generate_encryption_key()
        md = manager_class(encryption_key=encryption_key)
        dataset = md.encode_vars(dataset)

        filters = dataset["data"].encoding["filters"]
        assert len(filters) == 1
        assert isinstance(filters[0], encryption.EncryptionFilter)

    @staticmethod
    def test_encode_vars_w_encryption_key_and_preexisting_filter(manager_class):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {"filters": ["SomeOtherFilter"]}
        encryption_key = encryption.generate_encryption_key()
        md = manager_class(encryption_key=encryption_key)
        dataset = md.encode_vars(dataset)

        filters = dataset["data"].encoding["filters"]
        assert len(filters) == 2
        assert filters[0] == "SomeOtherFilter"
        assert isinstance(filters[1], encryption.EncryptionFilter)

    @staticmethod
    def test_remove_unwanted_fields_w_ipld_store(manager_class):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {}
        md = manager_class()
        md.store = store.IPLD(md)
        dataset = md.remove_unwanted_fields(dataset)
        assert isinstance(dataset["data"].encoding["compressor"], numcodecs.Blosc)

    @staticmethod
    def test_remove_unwanted_fields_w_ipld_store_no_compression(manager_class):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {}
        md = manager_class(use_compression=False)
        md.store = store.IPLD(md)
        dataset = md.remove_unwanted_fields(dataset)
        assert dataset["data"].encoding["compressor"] is None

    @staticmethod
    def test_populate_metadata(manager_class):
        md = {"hi": "mom", "hello": "dad"}
        dm = manager_class(static_metadata={"hi": "mom", "hello": "dad"})
        dm.populate_metadata()
        assert dm.metadata == md
