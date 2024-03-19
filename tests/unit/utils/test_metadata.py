import datetime
from unittest import mock

import pytest
import numcodecs
import numpy as np
from requests.exceptions import Timeout

from gridded_etl_tools.utils import encryption, metadata, store


@pytest.fixture
def organized_manager_class(manager_class):
    class Manager(manager_class):
        organization = "Church of the Flying Spaghetti Monster"

    return Manager


class TestMetadata:
    @staticmethod
    def test_default_stac_item(manager_class):
        assert manager_class.default_stac_item() == {
            "stac_version": "1.0.0",
            "type": "Feature",
            "id": "DummyManager",
            "collection": "Vintage Guitars",
            "links": [],
            "assets": {
                "zmetadata": {
                    "title": "DummyManager",
                    "type": "application/json",
                    "description": "Consolidated metadata file for DummyManager Zarr store, readable as a Zarr "
                    "dataset by Xarray",
                    "roles": ["metadata", "zarr-consolidated-metadata"],
                }
            },
        }

    @staticmethod
    def test_default_stac_collection(organized_manager_class):
        # TODO: Validate assumptions about already populated metadata
        dm = organized_manager_class(
            static_metadata={
                "coordinate reference system": "hyper euclidean",
                "license": "to ill",
                "provider description": "tall, awkward",
                "provider url": "http://example.com/hamburgers",
                "publisher": "Grand Royal Records",
                "terms of service": "you get what you get and you don't throw a fit",
                "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
            }
        )
        dm.populate_metadata()
        assert dm.default_stac_collection == {
            "id": "Vintage Guitars",
            "type": "Collection",
            "stac_extensions": ["https://stac-extensions.github.io/projection/v1.0.0/schema.json"],
            "stac_version": "1.0.0",
            "description": "tall, awkward",
            "license": "to ill",
            "collection": "Vintage Guitars",
            "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
            "extent": {"spatial": {"bbox": [[]]}, "temporal": {"interval": [[]]}},
            "links": [{"rel": "self", "type": "application/json", "title": "Vintage Guitars"}],
            "providers": [
                {
                    "name": "Church of the Flying Spaghetti Monster",
                    "description": "",
                    "roles": ["processor"],
                    "url": "",
                },
                {
                    "name": "Grand Royal Records",
                    "description": "tall, awkward",
                    "roles": ["producer"],
                    "url": "http://example.com/hamburgers",
                    "terms of service": "you get what you get and you don't throw a fit",
                },
            ],
            "summaries": {"proj:epsg": "hyper euclidean"},
        }

    @staticmethod
    def test_default_root_stac_catalog(organized_manager_class):
        assert organized_manager_class.default_root_stac_catalog() == {
            "id": "Church of the Flying Spaghetti Monster_data_catalog",
            "type": "Catalog",
            "title": "Church of the Flying Spaghetti Monster Data Catalog",
            "stac_version": "1.0.0",
            "description": "This catalog contains all the data uploaded to Church of the Flying Spaghetti Monster "
            "that has been issued STAC-compliant metadata. The catalogs and collections describe single providers. "
            "Each may contain one or multiple datasets. Each individual dataset has been documented as STAC Items.",
        }

    @staticmethod
    def test_remove_unwanted_fields_w_ipld_store(manager_class):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {}
        md = manager_class()
        md.store = store.IPLD(md)
        md.remove_unwanted_fields(dataset)
        assert isinstance(dataset["data"].encoding["compressor"], numcodecs.Blosc)

    @staticmethod
    def test_remove_unwanted_fields_w_ipld_store_no_compression(manager_class):
        dataset = mock.MagicMock()
        dataset["data"].encoding = {}
        md = manager_class(use_compression=False)
        md.store = store.IPLD(md)
        md.remove_unwanted_fields(dataset)
        assert dataset["data"].encoding["compressor"] is None

    @staticmethod
    def test_populate_metadata(manager_class):
        md = {"hi": "mom", "hello": "dad"}
        dm = manager_class(static_metadata={"hi": "mom", "hello": "dad"})
        dm.populate_metadata()
        assert dm.metadata == md

    @staticmethod
    def test_check_stac_exists_ipld(manager_class):
        dm = manager_class()
        dm.check_stac_on_ipns = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)
        assert dm.check_stac_exists("The Jungle Book", metadata.StacType.CATALOG) is dm.check_stac_on_ipns.return_value
        dm.check_stac_on_ipns.assert_called_once_with("The Jungle Book")
        dm.store.metadata_exists.assert_not_called()

    @staticmethod
    def test_check_stac_exists_not_ipld(manager_class):
        dm = manager_class()
        dm.check_stac_on_ipns = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)
        assert (
            dm.check_stac_exists("The Jungle Book", metadata.StacType.CATALOG) is dm.store.metadata_exists.return_value
        )
        dm.check_stac_on_ipns.assert_not_called()
        dm.store.metadata_exists.assert_called_once_with("The Jungle Book", metadata.StacType.CATALOG.value)

    @staticmethod
    def test_publish_stac_ipld(manager_class):
        dm = manager_class()
        dm.ipns_publish = mock.Mock()
        dm.ipfs_put = mock.Mock()
        dm.json_to_bytes = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)

        dm.publish_stac("The Jungle Book", {"hi": "mom!"}, metadata.StacType.CATALOG)
        dm.json_to_bytes.assert_called_once_with({"hi": "mom!"})
        dm.ipfs_put.assert_called_once_with(dm.json_to_bytes.return_value)
        dm.ipns_publish.assert_called_once_with("The Jungle Book", dm.ipfs_put.return_value)
        dm.store.push_metadata.assert_not_called()

    @staticmethod
    def test_publish_stac_not_ipld(manager_class):
        dm = manager_class()
        dm.ipns_publish = mock.Mock()
        dm.ipfs_put = mock.Mock()
        dm.json_to_bytes = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)

        dm.publish_stac("The Jungle Book", {"hi": "mom!"}, metadata.StacType.CATALOG)
        dm.json_to_bytes.assert_not_called()
        dm.ipfs_put.assert_not_called()
        dm.ipns_publish.assert_not_called()
        dm.store.push_metadata.assert_called_once_with(
            "The Jungle Book", {"hi": "mom!"}, metadata.StacType.CATALOG.value
        )

    @staticmethod
    def test_retrieve_stac_ipld(manager_class):
        dm = manager_class()
        dm.ipns_retrieve_object = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)
        assert dm.retrieve_stac("The Jungle Book", metadata.StacType.CATALOG) is dm.ipns_retrieve_object.return_value
        dm.ipns_retrieve_object.assert_called_once_with("The Jungle Book")
        dm.store.retrieve_metadata.assert_not_called()

    @staticmethod
    def test_retrieve_stac_not_ipld(manager_class):
        dm = manager_class()
        dm.ipns_retrieve_object = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)
        assert (
            dm.retrieve_stac("The Jungle Book", metadata.StacType.CATALOG) is dm.store.retrieve_metadata.return_value
        )
        dm.ipns_retrieve_object.assert_not_called()
        dm.store.retrieve_metadata.assert_called_once_with("The Jungle Book", metadata.StacType.CATALOG.value)

    @staticmethod
    def test_get_href_ipld(manager_class):
        dm = manager_class()
        dm.ipns_generate_name = mock.Mock()
        dm.store = mock.Mock(spec=store.IPLD)
        assert dm.get_href("The Jungle Book", metadata.StacType.CATALOG) is dm.ipns_generate_name.return_value
        dm.ipns_generate_name.assert_called_once_with(key="The Jungle Book")

    @staticmethod
    def test_get_href_not_ipld(manager_class):
        dm = manager_class()
        dm.ipns_generate_name = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)
        assert dm.get_href("The Jungle Book", metadata.StacType.CATALOG) is dm.store.get_metadata_path.return_value
        dm.ipns_generate_name.assert_not_called()
        dm.store.get_metadata_path.assert_called_once_with("The Jungle Book", metadata.StacType.CATALOG.value)

    @staticmethod
    def test_create_root_stac_catalog(organized_manager_class):
        dm = organized_manager_class()
        dm.publish_stac = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.check_stac_exists = mock.Mock(return_value=False)
        dm.get_href = mock.Mock(return_value="/it/is/here.json")

        dm.create_root_stac_catalog()
        dm.publish_stac.assert_called_once_with(
            "Church of the Flying Spaghetti Monster Data Catalog",
            {
                "id": "Church of the Flying Spaghetti Monster_data_catalog",
                "type": "Catalog",
                "title": "Church of the Flying Spaghetti Monster Data Catalog",
                "stac_version": "1.0.0",
                "description": "This catalog contains all the data uploaded to Church of the Flying Spaghetti Monster "
                "that has been issued STAC-compliant metadata. The catalogs and collections describe single "
                "providers. Each may contain one or multiple datasets. Each individual dataset has been documented as "
                "STAC Items.",
                "links": [
                    {
                        "rel": "root",
                        "href": "/it/is/here.json",
                        "type": "application/json",
                        "title": "Church of the Flying Spaghetti Monster root catalog",
                    },
                    {
                        "rel": "self",
                        "href": "/it/is/here.json",
                        "type": "application/json",
                        "title": "Church of the Flying Spaghetti Monster root catalog",
                    },
                ],
            },
            metadata.StacType.CATALOG,
        )
        dm.check_stac_exists.assert_called_once_with(
            "Church of the Flying Spaghetti Monster Data Catalog", metadata.StacType.CATALOG
        )
        dm.get_href.assert_called_once_with(
            "Church of the Flying Spaghetti Monster Data Catalog", metadata.StacType.CATALOG
        )

    @staticmethod
    def test_create_root_stac_catalog_already_exists(manager_class):
        dm = manager_class()
        dm.publish_stac = mock.Mock()
        dm.check_stac_exists = mock.Mock(return_value=True)
        dm.create_root_stac_catalog()
        dm.publish_stac.assert_not_called()

    @staticmethod
    def test_create_stac_collection(organized_manager_class, fake_original_dataset, mocker):
        dm = organized_manager_class(
            static_metadata={
                "coordinate reference system": "hyper euclidean",
                "license": "to ill",
                "provider description": "tall, awkward",
                "provider url": "http://example.com/hamburgers",
                "publisher": "Grand Royal Records",
                "terms of service": "you get what you get and you don't throw a fit",
                "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
            }
        )
        dm.populate_metadata()
        dm.publish_stac = mock.Mock()
        dm.retrieve_stac = mock.Mock(
            return_value=(
                {
                    "id": "Church of the Flying Spaghetti Monster_data_catalog",
                    "type": "Catalog",
                    "title": "Church of the Flying Spaghetti Monster Data Catalog",
                    "stac_version": "1.0.0",
                    "description": "This catalog contains all the data uploaded to Church of the Flying Spaghetti "
                    "Monster that has been issued STAC-compliant metadata. The catalogs and collections describe "
                    "single providers. Each may contain one or multiple datasets. Each individual dataset has been "
                    "documented as STAC Items.",
                    "links": [
                        {
                            "rel": "root",
                            "href": "/it/is/here.json",
                            "type": "application/json",
                            "title": "Church of the Flying Spaghetti Monster root catalog",
                        },
                        {
                            "rel": "self",
                            "href": "/it/is/here.json",
                            "type": "application/json",
                            "title": "Church of the Flying Spaghetti Monster root catalog",
                        },
                    ],
                },
                "proto://path/to/catalog",
            )
        )
        dm.update_stac_collection = mock.Mock()
        stac_collection = dm.default_stac_collection
        stac_collection["links"].append({"rel": "not myself today"})
        mocker.patch("gridded_etl_tools.utils.metadata.Metadata.default_stac_collection", stac_collection)
        dm.get_href = mock.Mock(return_value="/here/is/the/collection.json")

        dm.create_stac_collection(fake_original_dataset)

        dm.publish_stac.assert_has_calls(
            [
                mock.call(
                    "Church of the Flying Spaghetti Monster Data Catalog",
                    {
                        "id": "Church of the Flying Spaghetti Monster_data_catalog",
                        "type": "Catalog",
                        "title": "Church of the Flying Spaghetti Monster Data Catalog",
                        "stac_version": "1.0.0",
                        "description": "This catalog contains all the data uploaded to Church of the Flying Spaghetti "
                        "Monster that has been issued STAC-compliant metadata. The catalogs and collections describe "
                        "single providers. Each may contain one or multiple datasets. Each individual dataset has "
                        "been documented as STAC Items.",
                        "links": [
                            {
                                "rel": "root",
                                "href": "/it/is/here.json",
                                "type": "application/json",
                                "title": "Church of the Flying Spaghetti Monster root catalog",
                            },
                            {
                                "rel": "self",
                                "href": "/it/is/here.json",
                                "type": "application/json",
                                "title": "Church of the Flying Spaghetti Monster root catalog",
                            },
                            {
                                "rel": "child",
                                "href": "/here/is/the/collection.json",
                                "type": "application/json",
                                "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
                            },
                        ],
                    },
                    metadata.StacType.CATALOG,
                ),
                mock.call(
                    "Vintage Guitars",
                    {
                        "id": "Vintage Guitars",
                        "type": "Collection",
                        "stac_extensions": ["https://stac-extensions.github.io/projection/v1.0.0/schema.json"],
                        "stac_version": "1.0.0",
                        "description": "tall, awkward",
                        "license": "to ill",
                        "collection": "Vintage Guitars",
                        "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
                        "extent": {
                            "spatial": {"bbox": [[100.0, 10.0, 130.0, 40.0]]},
                            "temporal": {"interval": [["2021-09-16T00:00:00Z", "2022-01-31T00:00:00Z"]]},
                        },
                        "links": [
                            {
                                "rel": "self",
                                "type": "application/json",
                                "title": "Vintage Guitars",
                                "href": "/here/is/the/collection.json",
                            },
                            {"rel": "not myself today"},
                            {
                                "rel": "root",
                                "href": "proto://path/to/catalog",
                                "type": "application/json",
                                "title": "Church of the Flying Spaghetti Monster root catalog",
                            },
                            {
                                "rel": "parent",
                                "href": "proto://path/to/catalog",
                                "type": "application/json",
                                "title": "Church of the Flying Spaghetti Monster root catalog",
                            },
                        ],
                        "providers": [
                            {
                                "name": "Church of the Flying Spaghetti Monster",
                                "description": "",
                                "roles": ["processor"],
                                "url": "",
                            },
                            {
                                "name": "Grand Royal Records",
                                "description": "tall, awkward",
                                "roles": ["producer"],
                                "url": "http://example.com/hamburgers",
                                "terms of service": "you get what you get and you don't throw a fit",
                            },
                        ],
                        "summaries": {"proj:epsg": "hyper euclidean"},
                    },
                    metadata.StacType.COLLECTION,
                ),
            ]
        )
        dm.retrieve_stac.assert_called_once_with(
            "Church of the Flying Spaghetti Monster Data Catalog", metadata.StacType.CATALOG
        )
        dm.update_stac_collection.assert_not_called()
        dm.get_href.assert_called_once_with("Vintage Guitars", metadata.StacType.COLLECTION)

    @staticmethod
    def test_create_stac_collection_rebuild(organized_manager_class, fake_original_dataset):
        dm = organized_manager_class(
            static_metadata={
                "coordinate reference system": "hyper euclidean",
                "license": "to ill",
                "provider description": "tall, awkward",
                "provider url": "http://example.com/hamburgers",
                "publisher": "Grand Royal Records",
                "terms of service": "you get what you get and you don't throw a fit",
                "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
            }
        )
        dm.populate_metadata()
        dm.publish_stac = mock.Mock()
        dm.retrieve_stac = mock.Mock(
            return_value=(
                {
                    "id": "Church of the Flying Spaghetti Monster_data_catalog",
                    "type": "Catalog",
                    "title": "Church of the Flying Spaghetti Monster Data Catalog",
                    "stac_version": "1.0.0",
                    "description": "This catalog contains all the data uploaded to Church of the Flying Spaghetti "
                    "Monster that has been issued STAC-compliant metadata. The catalogs and collections describe "
                    "single providers. Each may contain one or multiple datasets. Each individual dataset has been "
                    "documented as STAC Items.",
                    "links": [
                        {
                            "rel": "root",
                            "href": "/path/to/catalog.json",
                            "type": "application/json",
                            "title": "Church of the Flying Spaghetti Monster root catalog",
                        },
                        {
                            "rel": "self",
                            "href": "/path/to/catalog.json",
                            "type": "application/json",
                            "title": "Church of the Flying Spaghetti Monster root catalog",
                        },
                    ],
                },
                "proto://path/to/catalog",
            )
        )
        dm.update_stac_collection = mock.Mock()
        dm.check_stac_exists = mock.Mock(return_value=True)
        dm.get_href = mock.Mock(return_value="/path/to/collection.json")

        dm.create_stac_collection(fake_original_dataset, rebuild=True)

        dm.publish_stac.assert_has_calls(
            [
                mock.call(
                    "Church of the Flying Spaghetti Monster Data Catalog",
                    {
                        "id": "Church of the Flying Spaghetti Monster_data_catalog",
                        "type": "Catalog",
                        "title": "Church of the Flying Spaghetti Monster Data Catalog",
                        "stac_version": "1.0.0",
                        "description": "This catalog contains all the data uploaded to Church of the Flying Spaghetti "
                        "Monster that has been issued STAC-compliant metadata. The catalogs and collections describe "
                        "single providers. Each may contain one or multiple datasets. Each individual dataset has "
                        "been documented as STAC Items.",
                        "links": [
                            {
                                "rel": "root",
                                "href": "/path/to/catalog.json",
                                "type": "application/json",
                                "title": "Church of the Flying Spaghetti Monster root catalog",
                            },
                            {
                                "rel": "self",
                                "href": "/path/to/catalog.json",
                                "type": "application/json",
                                "title": "Church of the Flying Spaghetti Monster root catalog",
                            },
                            {
                                "rel": "child",
                                "href": "/path/to/collection.json",
                                "type": "application/json",
                                "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
                            },
                        ],
                    },
                    metadata.StacType.CATALOG,
                ),
                mock.call(
                    "Vintage Guitars",
                    {
                        "id": "Vintage Guitars",
                        "type": "Collection",
                        "stac_extensions": ["https://stac-extensions.github.io/projection/v1.0.0/schema.json"],
                        "stac_version": "1.0.0",
                        "description": "tall, awkward",
                        "license": "to ill",
                        "collection": "Vintage Guitars",
                        "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
                        "extent": {
                            "spatial": {"bbox": [[100.0, 10.0, 130.0, 40.0]]},
                            "temporal": {"interval": [["2021-09-16T00:00:00Z", "2022-01-31T00:00:00Z"]]},
                        },
                        "links": [
                            {
                                "rel": "self",
                                "type": "application/json",
                                "title": "Vintage Guitars",
                                "href": "/path/to/collection.json",
                            },
                            {
                                "rel": "root",
                                "href": "proto://path/to/catalog",
                                "type": "application/json",
                                "title": "Church of the Flying Spaghetti Monster root catalog",
                            },
                            {
                                "rel": "parent",
                                "href": "proto://path/to/catalog",
                                "type": "application/json",
                                "title": "Church of the Flying Spaghetti Monster root catalog",
                            },
                        ],
                        "providers": [
                            {
                                "name": "Church of the Flying Spaghetti Monster",
                                "description": "",
                                "roles": ["processor"],
                                "url": "",
                            },
                            {
                                "name": "Grand Royal Records",
                                "description": "tall, awkward",
                                "roles": ["producer"],
                                "url": "http://example.com/hamburgers",
                                "terms of service": "you get what you get and you don't throw a fit",
                            },
                        ],
                        "summaries": {"proj:epsg": "hyper euclidean"},
                    },
                    metadata.StacType.COLLECTION,
                ),
            ]
        )
        dm.retrieve_stac.assert_called_once_with(
            "Church of the Flying Spaghetti Monster Data Catalog", metadata.StacType.CATALOG
        )
        dm.update_stac_collection.assert_not_called()
        dm.get_href.assert_called_once_with("Vintage Guitars", metadata.StacType.COLLECTION)

    @staticmethod
    def test_create_stac_collection_root_catalog_already_in_links(organized_manager_class, fake_original_dataset):
        dm = organized_manager_class(
            static_metadata={
                "coordinate reference system": "hyper euclidean",
                "license": "to ill",
                "provider description": "tall, awkward",
                "provider url": "http://example.com/hamburgers",
                "publisher": "Grand Royal Records",
                "terms of service": "you get what you get and you don't throw a fit",
                "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
            }
        )
        dm.populate_metadata()
        dm.publish_stac = mock.Mock()
        dm.retrieve_stac = mock.Mock(
            return_value=(
                {
                    "id": "Church of the Flying Spaghetti Monster_data_catalog",
                    "type": "Catalog",
                    "title": "Church of the Flying Spaghetti Monster Data Catalog",
                    "stac_version": "1.0.0",
                    "description": "This catalog contains all the data uploaded to Church of the Flying Spaghetti "
                    "Monster that has been issued STAC-compliant metadata. The catalogs and collections describe "
                    "single providers. Each may contain one or multiple datasets. Each individual dataset has been "
                    "documented as "
                    "STAC Items.",
                    "links": [
                        {
                            "rel": "root",
                            "href": "/path/to/catalog.json",
                            "type": "application/json",
                            "title": "Church of the Flying Spaghetti Monster root catalog",
                        },
                        {
                            "rel": "self",
                            "href": "/path/to/catalog.json",
                            "type": "application/json",
                            "title": "Church of the Flying Spaghetti Monster root catalog",
                        },
                        {
                            "rel": "child",
                            "href": "/what/ever",
                            "type": "application/json",
                            "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
                        },
                    ],
                },
                "proto://path/to/catalog",
            )
        )
        dm.update_stac_collection = mock.Mock()
        dm.get_href = mock.Mock(return_value="/path/to/collection.json")

        dm.create_stac_collection(fake_original_dataset)

        dm.publish_stac.assert_called_once_with(
            "Vintage Guitars",
            {
                "id": "Vintage Guitars",
                "type": "Collection",
                "stac_extensions": ["https://stac-extensions.github.io/projection/v1.0.0/schema.json"],
                "stac_version": "1.0.0",
                "description": "tall, awkward",
                "license": "to ill",
                "collection": "Vintage Guitars",
                "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
                "extent": {
                    "spatial": {"bbox": [[100.0, 10.0, 130.0, 40.0]]},
                    "temporal": {"interval": [["2021-09-16T00:00:00Z", "2022-01-31T00:00:00Z"]]},
                },
                "links": [
                    {
                        "rel": "self",
                        "type": "application/json",
                        "title": "Vintage Guitars",
                        "href": "/path/to/collection.json",
                    },
                    {
                        "rel": "root",
                        "href": "proto://path/to/catalog",
                        "type": "application/json",
                        "title": "Church of the Flying Spaghetti Monster root catalog",
                    },
                    {
                        "rel": "parent",
                        "href": "proto://path/to/catalog",
                        "type": "application/json",
                        "title": "Church of the Flying Spaghetti Monster root catalog",
                    },
                ],
                "providers": [
                    {
                        "name": "Church of the Flying Spaghetti Monster",
                        "description": "",
                        "roles": ["processor"],
                        "url": "",
                    },
                    {
                        "name": "Grand Royal Records",
                        "description": "tall, awkward",
                        "roles": ["producer"],
                        "url": "http://example.com/hamburgers",
                        "terms of service": "you get what you get and you don't throw a fit",
                    },
                ],
                "summaries": {"proj:epsg": "hyper euclidean"},
            },
            metadata.StacType.COLLECTION,
        )
        dm.retrieve_stac.assert_called_once_with(
            "Church of the Flying Spaghetti Monster Data Catalog", metadata.StacType.CATALOG
        )
        dm.update_stac_collection.assert_not_called()
        dm.get_href.assert_called_once_with("Vintage Guitars", metadata.StacType.COLLECTION)

    @staticmethod
    def test_create_stac_collection_already_created(manager_class, fake_original_dataset, mocker):
        dm = manager_class(
            static_metadata={
                "coordinate reference system": "hyper euclidean",
                "license": "to ill",
                "provider description": "tall, awkward",
                "provider url": "http://example.com/hamburgers",
                "publisher": "Grand Royal Records",
                "terms of service": "you get what you get and you don't throw a fit",
                "title": "Faccia di Broccoli: La Mia Vita nelle Miniere",
            }
        )
        dm.populate_metadata()
        dm.publish_stac = mock.Mock()
        dm.retrieve_stac = mock.Mock()
        dm.update_stac_collection = mock.Mock()
        stac_collection = dm.default_stac_collection
        stac_collection["links"].append({"rel": "not myself today"})
        mocker.patch("gridded_etl_tools.utils.metadata.Metadata.default_stac_collection", stac_collection)
        dm.check_stac_exists = mock.Mock(return_value=True)
        dm.create_stac_collection(fake_original_dataset)

        dm.publish_stac.assert_not_called()
        dm.retrieve_stac.assert_not_called()
        dm.update_stac_collection.assert_called_once_with(fake_original_dataset)

    @staticmethod
    def test_create_stac_item_ipld(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.utcnow = mock.Mock(return_value=datetime.datetime(2010, 5, 12, 2, 42))
        dt_mock.timezone = datetime.timezone

        dm = manager_class()
        dm.store = mock.Mock(spec=store.IPLD)
        dm.register_stac_item = mock.Mock()
        dm.latest_hash = mock.Mock(return_value="QmThisOneHere")
        dm.create_stac_item(fake_original_dataset)

        dm.register_stac_item.assert_called_once_with(
            {
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": "DummyManager",
                "collection": "Vintage Guitars",
                "links": [],
                "assets": {
                    "zmetadata": {
                        "title": "DummyManager",
                        "type": "application/json",
                        "description": "Consolidated metadata file for DummyManager Zarr store, readable as a Zarr "
                        "dataset by Xarray",
                        "roles": ["metadata", "zarr-consolidated-metadata"],
                        "href": {"/": "QmThisOneHere"},
                    }
                },
                "bbox": [100.0, 10.0, 130.0, 40.0],
                "geometry": '{"type": "Polygon", "coordinates": [[[130.0, 10.0], [130.0, 40.0], [100.0, 40.0], '
                "[100.0, 10.0], [130.0, 10.0]]]}",
                "properties": {
                    "array_size": {"latitude": 4, "longitude": 4, "time": 138},
                    "start_datetime": "2021-09-16T00:00:00Z",
                    "end_datetime": "2022-01-31T00:00:00Z",
                    "updated": "2010-05-12T0Z",
                },
            }
        )

    @staticmethod
    def test_create_stac_item_ipld_forecast(manager_class, forecast_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.utcnow = mock.Mock(return_value=datetime.datetime(2010, 5, 12, 2, 42))
        dt_mock.timezone = datetime.timezone

        dm = manager_class()
        dm.store = mock.Mock(spec=store.IPLD)
        dm.register_stac_item = mock.Mock()
        dm.latest_hash = mock.Mock(return_value="QmThisOneHere")
        dm.time_dim = "forecast_reference_time"
        dm.create_stac_item(forecast_dataset)

        dm.register_stac_item.assert_called_once_with(
            {
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": "DummyManager",
                "collection": "Vintage Guitars",
                "links": [],
                "assets": {
                    "zmetadata": {
                        "title": "DummyManager",
                        "type": "application/json",
                        "description": "Consolidated metadata file for DummyManager Zarr store, readable as a Zarr "
                        "dataset by Xarray",
                        "roles": ["metadata", "zarr-consolidated-metadata"],
                        "href": {"/": "QmThisOneHere"},
                    }
                },
                "bbox": [100.0, 10.0, 130.0, 40.0],
                "geometry": '{"type": "Polygon", "coordinates": [[[130.0, 10.0], [130.0, 40.0], [100.0, 40.0], '
                "[100.0, 10.0], [130.0, 10.0]]]}",
                "properties": {
                    "array_size": {"latitude": 4, "longitude": 4, "forecast_reference_time": 138, "step": 4},
                    "start_datetime": "2021-09-16T00:00:00Z",
                    "end_datetime": "2022-01-31T00:00:00Z",
                    "updated": "2010-05-12T0Z",
                },
            }
        )

    @staticmethod
    def test_create_stac_item_not_ipld(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.utcnow = mock.Mock(return_value=datetime.datetime(2010, 5, 12, 2, 42))
        dt_mock.timezone = datetime.timezone

        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface, path="it/goes/here")
        dm.register_stac_item = mock.Mock()
        dm.latest_hash = mock.Mock(return_value="QmThisOneHere")
        dm.create_stac_item(fake_original_dataset)

        dm.register_stac_item.assert_called_once_with(
            {
                "stac_version": "1.0.0",
                "type": "Feature",
                "id": "DummyManager",
                "collection": "Vintage Guitars",
                "links": [],
                "assets": {
                    "zmetadata": {
                        "title": "DummyManager",
                        "type": "application/json",
                        "description": "Consolidated metadata file for DummyManager Zarr store, readable as a Zarr "
                        "dataset by Xarray",
                        "roles": ["metadata", "zarr-consolidated-metadata"],
                        "href": "it/goes/here",
                    }
                },
                "bbox": [100.0, 10.0, 130.0, 40.0],
                "geometry": '{"type": "Polygon", "coordinates": [[[130.0, 10.0], [130.0, 40.0], [100.0, 40.0], '
                "[100.0, 10.0], [130.0, 10.0]]]}",
                "properties": {
                    "array_size": {"latitude": 4, "longitude": 4, "time": 138},
                    "start_datetime": "2021-09-16T00:00:00Z",
                    "end_datetime": "2022-01-31T00:00:00Z",
                    "updated": "2010-05-12T0Z",
                },
            }
        )

    @staticmethod
    def test_zarr_md_to_stac_format(manager_class, fake_original_dataset):
        dm = manager_class()
        fake_original_dataset.attrs["missing_value"] = 42
        fake_original_dataset.attrs["temporal resolution"] = np.float32(0.125)
        fake_original_dataset.encoding["preferred_chunks"] = "chocolate and peanut"
        fake_original_dataset["data"].encoding["standard name"] = "chris"

        assert dm.zarr_md_to_stac_format(fake_original_dataset) == {
            "standard name": "chris",
            "temporal resolution": 0.125,
            "Zarr chunk size": "chocolate and peanut",
        }

    @staticmethod
    def test_zarr_md_to_stac_format_with_dtype(manager_class, fake_original_dataset):
        dm = manager_class()
        fake_original_dataset.attrs["missing_value"] = 42
        fake_original_dataset.attrs["dtype"] = np.dtype("float32")
        fake_original_dataset.encoding["preferred_chunks"] = "chocolate and peanut"
        fake_original_dataset["data"].encoding["standard name"] = "chris"

        assert dm.zarr_md_to_stac_format(fake_original_dataset) == {
            "dtype": "float32",
            "standard name": "chris",
            "Zarr chunk size": "chocolate and peanut",
        }

    @staticmethod
    def test_register_stac_item_ipld(manager_class):
        stac_collection = {
            "title": "War and Peace",
            "links": [],
        }
        stac_item = {
            "Look": "I'm",
            "a": "stac item",
            "links": [],
            "assets": {"zmetadata": {"title": "Asset and Peace"}},
        }

        md = manager_class()
        md.publish_stac = mock.Mock()
        md.store = mock.Mock(spec=store.IPLD)
        md.retrieve_stac = mock.Mock(side_effect=[(stac_collection, "/path/to/stac/collection"), Timeout])
        md.get_href = mock.Mock(return_value="/path/to/new/item")
        md.ipns_resolve = mock.Mock(return_value="QmSomeHash")

        md.register_stac_item(stac_item)

        md.publish_stac.assert_has_calls(
            [
                mock.call(
                    "DummyManager-daily",
                    {
                        "Look": "I'm",
                        "a": "stac item",
                        "links": [
                            {
                                "rel": "parent",
                                "href": "/path/to/stac/collection",
                                "type": "application/geo+json",
                                "title": "War and Peace",
                            },
                            {
                                "rel": "self",
                                "href": "/path/to/new/item",
                                "type": "application/geo+json",
                                "title": "DummyManager metadata",
                            },
                        ],
                        "assets": {"zmetadata": {"title": "Asset and Peace"}},
                    },
                    metadata.StacType.ITEM,
                ),
                mock.call(
                    "Vintage Guitars",
                    {
                        "title": "War and Peace",
                        "links": [
                            {
                                "rel": "item",
                                "href": "/path/to/new/item",
                                "type": "application/json",
                                "title": "Asset and Peace",
                            }
                        ],
                    },
                    metadata.StacType.COLLECTION,
                ),
            ]
        )
        md.retrieve_stac.assert_has_calls(
            [
                mock.call("Vintage Guitars", metadata.StacType.COLLECTION),
                mock.call("DummyManager-daily", metadata.StacType.ITEM),
            ]
        )
        md.get_href.assert_called_once_with("DummyManager-daily", metadata.StacType.ITEM)
        md.ipns_resolve.assert_not_called()

    @staticmethod
    def test_register_stac_item_already_exists_ipld(manager_class):
        stac_collection = {
            "title": "War and Peace",
            "links": [{"rel": "lol"}, {"rel": "item", "title": "Asset and Peace", "href": "/old/path/to/item"}],
        }
        old_stac_cid = mock.Mock(
            spec=("set",), set=mock.Mock(return_value=mock.MagicMock(__str__=mock.Mock(return_value="QmOldStacItem")))
        )
        old_stac_item = {
            "Look": "I'm",
            "the old": "stac item",
            "assets": {"zmetadata": {"title": "Asset and Peace", "href": old_stac_cid}},
        }
        stac_item = {
            "Look": "I'm",
            "a": "stac item",
            "links": [],
            "assets": {"zmetadata": {"title": "Asset and Peace"}},
        }

        md = manager_class()
        md.publish_stac = mock.Mock()
        md.store = mock.Mock(spec=store.IPLD)
        md.retrieve_stac = mock.Mock(
            side_effect=[(stac_collection, "/path/to/stac/collection"), (old_stac_item, "/path/to/stac/item")]
        )
        md.get_href = mock.Mock(return_value="/path/to/new/item")
        md.ipns_resolve = mock.Mock(return_value="QmSomeHash")

        md.register_stac_item(stac_item)

        md.publish_stac.assert_called_once_with(
            "DummyManager-daily",
            {
                "Look": "I'm",
                "a": "stac item",
                "links": [
                    {
                        "rel": "parent",
                        "href": "/path/to/stac/collection",
                        "type": "application/geo+json",
                        "title": "War and Peace",
                    },
                    {
                        "rel": "prev",
                        "href": "QmOldStacItem",
                        "metadata href": {"/": "QmSomeHash"},
                        "type": "application/geo+json",
                        "title": "Asset and Peace",
                    },
                    {
                        "rel": "self",
                        "href": "/path/to/stac/item",
                        "type": "application/geo+json",
                        "title": "DummyManager metadata",
                    },
                ],
                "assets": {"zmetadata": {"title": "Asset and Peace"}},
            },
            metadata.StacType.ITEM,
        )
        md.retrieve_stac.assert_has_calls(
            [
                mock.call("Vintage Guitars", metadata.StacType.COLLECTION),
                mock.call("DummyManager-daily", metadata.StacType.ITEM),
            ]
        )
        md.get_href.assert_not_called()
        md.ipns_resolve.assert_called_once_with("DummyManager-daily")
        old_stac_cid.set.assert_called_once_with(base="base32")

    @staticmethod
    def test_register_stac_item_already_exists_not_ipld(manager_class):
        stac_collection = {
            "title": "War and Peace",
            "links": [{"rel": "lol"}, {"rel": "item", "title": "Asset and Peace", "href": "/old/path/to/item"}],
        }
        old_stac_item = {
            "Look": "I'm",
            "the old": "stac item",
            "assets": {"zmetadata": {"title": "Asset and Peace", "href": "QmOldStacItem"}},
        }
        stac_item = {
            "Look": "I'm",
            "a": "stac item",
            "links": [],
            "assets": {"zmetadata": {"title": "Asset and Peace"}},
        }

        md = manager_class()
        md.publish_stac = mock.Mock()
        md.store = mock.Mock(spec=store.StoreInterface)
        md.retrieve_stac = mock.Mock(
            side_effect=[(stac_collection, "/path/to/stac/collection"), (old_stac_item, "/path/to/stac/item")]
        )
        md.get_href = mock.Mock(return_value="/path/to/new/item")
        md.ipns_resolve = mock.Mock(return_value="QmSomeHash")

        md.register_stac_item(stac_item)

        md.publish_stac.assert_called_once_with(
            "DummyManager-daily",
            {
                "Look": "I'm",
                "a": "stac item",
                "links": [
                    {
                        "rel": "parent",
                        "href": "/path/to/stac/collection",
                        "type": "application/geo+json",
                        "title": "War and Peace",
                    },
                    {
                        "rel": "self",
                        "href": "/path/to/stac/item",
                        "type": "application/geo+json",
                        "title": "DummyManager metadata",
                    },
                ],
                "assets": {"zmetadata": {"title": "Asset and Peace"}},
            },
            metadata.StacType.ITEM,
        )
        md.retrieve_stac.assert_has_calls(
            [
                mock.call("Vintage Guitars", metadata.StacType.COLLECTION),
                mock.call("DummyManager-daily", metadata.StacType.ITEM),
            ]
        )
        md.get_href.assert_not_called()
        md.ipns_resolve.assert_not_called()

    @staticmethod
    def test_update_stac_collection(manager_class, fake_original_dataset):
        md = manager_class()
        md.publish_stac = mock.Mock()

        stac_collection = [{"extent": {"spatial": {"bbox": [-24, -12, 24, 12]}, "temporal": {}}}, {"nothing": "here"}]
        md.retrieve_stac = mock.Mock(return_value=stac_collection)

        md.update_stac_collection(fake_original_dataset)

        md.retrieve_stac.assert_called_once_with("Vintage Guitars", metadata.StacType.COLLECTION)
        md.publish_stac.assert_called_once_with(
            "Vintage Guitars",
            {
                "extent": {
                    "spatial": {"bbox": [[-24.0, -24.0, 130.0, 40.0]]},
                    "temporal": {"interval": [["2021-09-16T00:00:00Z", "2022-01-31T00:00:00Z"]]},
                }
            },
            metadata.StacType.COLLECTION,
        )

    @staticmethod
    def test_load_stac_metadata(manager_class):
        md = manager_class()
        md.store = mock.Mock(spec=store.IPLD)
        md.retrieve_stac = mock.Mock(return_value=["foo", "bar"])

        assert md.load_stac_metadata() == "foo"

        md.retrieve_stac.assert_called_once_with("DummyManager-daily", metadata.StacType.ITEM)

    @staticmethod
    def test_load_stac_metadata_pass_key(manager_class):
        md = manager_class()
        md.store = mock.Mock(spec=store.IPLD)
        md.retrieve_stac = mock.Mock(return_value=["foo", "bar"])

        assert md.load_stac_metadata(key="chiave") == "foo"

        md.retrieve_stac.assert_called_once_with("chiave", metadata.StacType.ITEM)

    @staticmethod
    def test_load_stac_metadata_timeout(manager_class):
        md = manager_class()
        md.store = mock.Mock(spec=store.IPLD)
        md.retrieve_stac = mock.Mock(side_effect=Timeout)

        assert md.load_stac_metadata() == {}

        md.retrieve_stac.assert_called_once_with("DummyManager-daily", metadata.StacType.ITEM)

    @staticmethod
    def test_load_stac_metadata_not_ipld(manager_class):
        md = manager_class()
        md.store = mock.Mock(spec=store.StoreInterface)
        md.retrieve_stac = mock.Mock(return_value=["foo", "bar"])

        assert md.load_stac_metadata() is None

        md.retrieve_stac.assert_not_called()

    @staticmethod
    def test_set_zarr_metadata(manager_class):
        dataset = mock.Mock()
        md = manager_class()

        md.rename_data_variable = mock.Mock()
        renamed = md.rename_data_variable.return_value

        md.remove_unwanted_fields = mock.Mock()
        md.encode_vars = mock.Mock()
        md.merge_in_outside_metadata = mock.Mock()
        md.suppress_invalid_attributes = mock.Mock()

        dataset.attrs = {}
        assert md.set_zarr_metadata(dataset) is renamed

        md.rename_data_variable.assert_called_once_with(dataset)
        md.remove_unwanted_fields.assert_called_once_with(renamed)
        md.encode_vars.assert_called_once_with(renamed)
        md.suppress_invalid_attributes.assert_called_once_with(renamed)

    @staticmethod
    def test_suppress_invalid_attributes(manager_class):
        dataset = mock.Mock()
        md = manager_class()

        dataset.attrs = {"foo": "bar", "baz": None, "goo": {"gar": "gaz"}}
        md.suppress_invalid_attributes(dataset)

        assert dataset.attrs == {"foo": "bar", "baz": "", "goo": '{"gar": "gaz"}'}

    @staticmethod
    def test_rename_data_variable(manager_class):
        dataset = mock.Mock(data_vars=["one", "two", "three"])
        renamed = dataset.rename_vars.return_value

        md = manager_class()
        assert md.rename_data_variable(dataset) is renamed

        dataset.rename_vars.assert_called_once_with({"one": "data"})

    @staticmethod
    def test_rename_data_variable_value_error(manager_class):
        dataset = mock.Mock(data_vars=["one", "two", "three"])
        dataset.rename_vars.side_effect = ValueError

        md = manager_class()
        assert md.rename_data_variable(dataset) is dataset

        dataset.rename_vars.assert_called_once_with({"one": "data"})

    @staticmethod
    def test_encode_vars(manager_class, fake_original_dataset):
        dataset = fake_original_dataset
        assert dataset.encoding == {}
        assert dataset["data"].encoding == {}
        assert dataset.time.encoding == {}

        md = manager_class()
        md.encode_vars(dataset)

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": "", "missing_value": ""}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": "",
            "missing_value": "",
            "chunks": (),
            "preferred_chunks": {},
        }
        assert dataset.time.encoding == {
            "long_name": "time",
            "calendar": "gregorian",
            "units": "days since 1975-07-07 0:0:0 0",
        }

    @staticmethod
    def test_encode_vars_forecast(manager_class, forecast_dataset):
        dataset = forecast_dataset
        assert dataset.encoding == {}
        assert dataset["data"].encoding == {}
        assert dataset.forecast_reference_time.encoding == {}

        md = manager_class()
        md.time_dim = "forecast_reference_time"
        md.encode_vars(dataset)

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": "", "missing_value": ""}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": "",
            "missing_value": "",
            "chunks": (),
            "preferred_chunks": {},
        }
        assert dataset.forecast_reference_time.encoding == {
            "long_name": "initial time of forecast",
            "standard_name": "forecast_reference_time",
            "calendar": "proleptic_gregorian",
            "units": "days since 1975-07-07 0:0:0 0",
        }

    @staticmethod
    def test_encode_vars_hindcast(manager_class, hindcast_dataset):
        dataset = hindcast_dataset
        assert dataset.encoding == {}
        assert dataset["data"].encoding == {}
        assert dataset.hindcast_reference_time.encoding == {}

        md = manager_class()
        md.time_dim = "hindcast_reference_time"
        md.encode_vars(dataset) is dataset

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": "", "missing_value": ""}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": "",
            "missing_value": "",
            "chunks": (),
            "preferred_chunks": {},
        }
        assert dataset.hindcast_reference_time.encoding == {
            "long_name": "initial time of forecast",
            "standard_name": "hindcast_reference_time",
            "calendar": "proleptic_gregorian",
            "units": "days since 1975-07-07 0:0:0 0",
        }

    @staticmethod
    def test_encode_vars_w_encryption_key(manager_class, fake_original_dataset):
        dataset = fake_original_dataset
        assert dataset.encoding == {}
        assert dataset["data"].encoding == {}
        assert dataset.time.encoding == {}

        encryption_key = encryption.generate_encryption_key()
        md = manager_class(encryption_key=encryption_key)
        md.encode_vars(dataset)

        filters = dataset["data"].encoding["filters"]
        assert len(filters) == 1
        assert isinstance(filters[0], encryption.EncryptionFilter)

    @staticmethod
    def test_encode_vars_w_encryption_key_and_preexisting_filter(manager_class, fake_original_dataset):
        dataset = fake_original_dataset
        dataset["data"].encoding = {"filters": ["SomeOtherFilter"]}
        assert dataset.encoding == {}
        assert dataset.time.encoding == {}

        encryption_key = encryption.generate_encryption_key()
        md = manager_class(encryption_key=encryption_key)
        md.encode_vars(dataset)

        filters = dataset["data"].encoding["filters"]
        assert len(filters) == 2
        assert filters[0] == "SomeOtherFilter"
        assert isinstance(filters[1], encryption.EncryptionFilter)

    @staticmethod
    def test_encode_vars_time_units_known(manager_class, fake_original_dataset):
        dataset = fake_original_dataset
        assert dataset.encoding == {}
        assert dataset["data"].encoding == {}
        dataset.time.encoding = {"units": "picoseconds since the big bang"}

        md = manager_class()
        md.encode_vars(dataset)

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": "", "missing_value": ""}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": "",
            "missing_value": "",
            "chunks": (),
            "preferred_chunks": {},
        }
        assert dataset.time.encoding == {
            "long_name": "time",
            "calendar": "gregorian",
            "units": "picoseconds since the big bang",
        }

    @staticmethod
    def test_merge_in_outside_metadata_not_ipld(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.utcnow.return_value = datetime.datetime(2000, 1, 1, 0, 0, 0)
        dt_mock.timezone = datetime.timezone
        dataset = fake_original_dataset
        dataset.attrs = {"foo": "bar"}

        md = manager_class(static_metadata={"bar": "baz"})
        md.populate_metadata()
        md.store = mock.Mock(spec=store.StoreInterface)
        md.store.dataset.return_value.attrs = {"date range": ("2000010100", "2021091600")}
        md.merge_in_outside_metadata(dataset)

        assert dataset.attrs == {
            "foo": "bar",
            "bar": "baz",
            "created": "2000-01-01T0Z",
            "update_previous_end_date": "2021091600",
            "date range": ("2000010100", "2022013100"),
            "update_date_range": ("2021091600", "2022013100"),
            "bbox": (100.0, 10.0, 130.0, 40.0),
            "update_is_append_only": True,
        }

    @staticmethod
    def test_merge_in_outside_metadata_not_ipld_no_previous_dataset(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.utcnow.return_value = datetime.datetime(2000, 1, 1, 0, 0, 0)
        dt_mock.timezone = datetime.timezone
        dataset = fake_original_dataset
        dataset.attrs = {"foo": "bar"}

        md = manager_class(static_metadata={"bar": "baz"})
        md.populate_metadata()
        md.store = mock.Mock(spec=store.StoreInterface, has_existing=False)
        md.merge_in_outside_metadata(dataset)

        assert dataset.attrs == {
            "foo": "bar",
            "bar": "baz",
            "created": "2000-01-01T0Z",
            "update_previous_end_date": "",
            "date range": ("2021091600", "2022013100"),
            "update_date_range": ("2021091600", "2022013100"),
            "bbox": (100.0, 10.0, 130.0, 40.0),
            "update_is_append_only": True,
        }

    @staticmethod
    def test_merge_in_outside_metadata_ipld_no_existing_created(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.utcnow.return_value = datetime.datetime(2000, 1, 1, 0, 0, 0)
        dt_mock.timezone = datetime.timezone

        dataset = fake_original_dataset
        dataset.attrs = {"foo": "bar"}

        md = manager_class(static_metadata={"bar": "baz"})
        md.populate_metadata()
        md.store = mock.Mock(spec=store.IPLD)
        md.load_stac_metadata = mock.Mock(return_value={"properties": {"date range": ("2000010100", "2021091600")}})
        md.merge_in_outside_metadata(dataset)

        assert dataset.attrs == {
            "foo": "bar",
            "bar": "baz",
            "created": "2000-01-01T0Z",
            "update_previous_end_date": "2021091600",
            "date range": ("2000010100", "2022013100"),
            "update_date_range": ("2021091600", "2022013100"),
            "bbox": (100.0, 10.0, 130.0, 40.0),
            "update_is_append_only": True,
        }
        md.load_stac_metadata.assert_called_once_with()

    @staticmethod
    def test_merge_in_outside_metadata_ipld_existing_created(manager_class, fake_original_dataset, mocker):
        dataset = fake_original_dataset
        dataset.attrs = {"foo": "bar"}

        md = manager_class(static_metadata={"bar": "baz"})
        md.populate_metadata()
        md.store = mock.Mock(spec=store.IPLD)
        md.load_stac_metadata = mock.Mock(
            return_value={"properties": {"date range": ("2000010100", "2021091600"), "created": "1999-01-01T0Z"}}
        )
        md.merge_in_outside_metadata(dataset)

        assert dataset.attrs == {
            "foo": "bar",
            "bar": "baz",
            "created": "1999-01-01T0Z",
            "update_previous_end_date": "2021091600",
            "date range": ("2000010100", "2022013100"),
            "update_date_range": ("2021091600", "2022013100"),
            "bbox": (100.0, 10.0, 130.0, 40.0),
            "update_is_append_only": True,
        }
        md.load_stac_metadata.assert_called_once_with()

    @staticmethod
    def test_merge_in_outside_metadata_ipld_stac_timeout(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.utcnow.return_value = datetime.datetime(2000, 1, 1, 0, 0, 0)
        dt_mock.timezone = datetime.timezone

        dataset = fake_original_dataset
        dataset.attrs = {"foo": "bar"}

        md = manager_class(static_metadata={"bar": "baz"})
        md.populate_metadata()
        md.store = mock.Mock(spec=store.IPLD)
        md.load_stac_metadata = mock.Mock(side_effect=Timeout)
        md.merge_in_outside_metadata(dataset)

        assert dataset.attrs == {
            "foo": "bar",
            "bar": "baz",
            "created": "2000-01-01T0Z",
            "update_previous_end_date": "",
            "date range": ("2021091600", "2022013100"),
            "update_date_range": ("2021091600", "2022013100"),
            "bbox": (100.0, 10.0, 130.0, 40.0),
            "update_is_append_only": True,
        }
        md.load_stac_metadata.assert_called_once_with()
