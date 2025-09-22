import datetime
from unittest import mock

import pathlib
import pytest
import numpy as np
import numcodecs
import zarr

# Imports used for legacy encoding change tests
# import os
# import json
# import xarray as xr
# from time import sleep

from requests.exceptions import Timeout

from gridded_etl_tools.utils import encryption, metadata, store
from ...common import clean_up_input_paths


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
    def test_set_initial_compression(manager_class, fake_original_dataset):
        dataset = fake_original_dataset
        for coord in dataset.coords:
            dataset[coord].encoding = {}
        dataset["data"].encoding = {}

        dm = manager_class(use_compression=True)
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.store.has_existing = False

        dm.set_initial_compression(dataset)
        for coord in dataset.coords:
            assert dataset[coord].encoding["compressors"] == numcodecs.Blosc()
        assert dataset["data"].encoding["compressors"] == numcodecs.Blosc()

        dm = manager_class(use_compression=True, output_zarr3=True)
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.store.has_existing = False

        dm.set_initial_compression(dataset)
        for coord in dataset.coords:
            assert dataset[coord].encoding["compressors"] == zarr.codecs.BloscCodec(cname="lz4")
        assert dataset["data"].encoding["compressors"] == zarr.codecs.BloscCodec(cname="lz4")

    @staticmethod
    def test_set_initial_compression_no_compression(manager_class, fake_original_dataset):
        """Test that `set_initial_compression` does nothing if compression is disabled"""
        dm = manager_class(use_compression=False)
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.store.has_existing = False

        dataset = fake_original_dataset
        dm.set_initial_compression(dataset)

        # Check that compressor is None for coordinates and data variable
        for coord in dataset.coords:
            assert dataset[coord].encoding["compressors"] is None

        assert dataset[dm.data_var].encoding["compressors"] is None

    @staticmethod
    def test_set_initial_compression_existing_store(manager_class, fake_original_dataset):
        """Test that `set_initial_compression` does nothing if the store already exists"""
        dm = manager_class(use_compression=True)
        dm.store = mock.Mock(spec=store.StoreInterface)
        dm.store.has_existing = True

        dataset = fake_original_dataset
        # Remove any existing compression encoding
        for coord in dataset.coords:
            dataset[coord].encoding.pop("compressors", None)
        dataset[dm.data_var].encoding.pop("compressors", None)

        dm.set_initial_compression(dataset)

        # Check that no compression was added
        for coord in dataset.coords:
            assert "compressors" not in dataset[coord].encoding

        assert "compressors" not in dataset[dm.data_var].encoding

    @staticmethod
    def test_populate_metadata(manager_class):
        md = {"hi": "mom", "hello": "dad"}
        dm = manager_class(static_metadata={"hi": "mom", "hello": "dad"})
        dm.populate_metadata()
        assert dm.metadata == md

    @staticmethod
    def test_check_stac_exists(manager_class):
        dm = manager_class()
        dm.check_stac_on_ipns = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)
        assert (
            dm.check_stac_exists("The Jungle Book", metadata.StacType.CATALOG) is dm.store.metadata_exists.return_value
        )
        dm.check_stac_on_ipns.assert_not_called()
        dm.store.metadata_exists.assert_called_once_with("The Jungle Book", metadata.StacType.CATALOG.value)

    @staticmethod
    def test_publish_stac(manager_class):
        dm = manager_class()
        dm.json_to_bytes = mock.Mock()
        dm.store = mock.Mock(spec=store.StoreInterface)

        dm.publish_stac("The Jungle Book", {"hi": "mom!"}, metadata.StacType.CATALOG)
        dm.json_to_bytes.assert_not_called()
        dm.store.push_metadata.assert_called_once_with(
            "The Jungle Book", {"hi": "mom!"}, metadata.StacType.CATALOG.value
        )

    @staticmethod
    def test_retrieve_stac(manager_class):
        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)
        assert (
            dm.retrieve_stac("The Jungle Book", metadata.StacType.CATALOG) is dm.store.retrieve_metadata.return_value
        )
        dm.store.retrieve_metadata.assert_called_once_with("The Jungle Book", metadata.StacType.CATALOG.value)

    @staticmethod
    def test_get_href(manager_class):
        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface)
        assert dm.get_href("The Jungle Book", metadata.StacType.CATALOG) is dm.store.get_metadata_path.return_value
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
    def test_create_stac_item(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.now = mock.Mock(return_value=datetime.datetime(2010, 5, 12, 2, 42))
        dt_mock.timezone = datetime.timezone

        dm = manager_class()
        dm.store = mock.Mock(spec=store.StoreInterface, path="it/goes/here")
        dm.register_stac_item = mock.Mock()
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
                    "dataset_category": "observation",
                    "array_size": {"latitude": 4, "longitude": 4, "time": 138},
                    "start_datetime": "2021-09-16T00:00:00Z",
                    "end_datetime": "2022-01-31T00:00:00Z",
                    "updated": "2010-05-12T02:42:00Z",
                },
            }
        )

    @staticmethod
    def test_create_stac_item_forecast_reference_time(manager_class, forecast_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.now = mock.Mock(return_value=datetime.datetime(2010, 5, 12, 2, 42))
        dt_mock.timezone = datetime.timezone

        dm = manager_class()
        dm.time_dim = "forecast_reference_time"
        dm.store = mock.Mock(spec=store.StoreInterface, path="it/goes/here")
        dm.register_stac_item = mock.Mock()
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
                        "href": "it/goes/here",
                    }
                },
                "bbox": [100.0, 10.0, 130.0, 40.0],
                "geometry": '{"type": "Polygon", "coordinates": [[[130.0, 10.0], [130.0, 40.0], [100.0, 40.0], '
                "[100.0, 10.0], [130.0, 10.0]]]}",
                "properties": {
                    "dataset_category": "observation",
                    "array_size": {"latitude": 4, "longitude": 4, "forecast_reference_time": 138, "step": 4},
                    "start_datetime": "2021-09-16T00:00:00Z",
                    "end_datetime": "2022-01-31T00:00:00Z",
                    "updated": "2010-05-12T02:42:00Z",
                },
            }
        )

    @staticmethod
    def test_zarr_md_to_stac_format(manager_class, fake_original_dataset):
        dm = manager_class()
        fake_original_dataset.attrs["_FillValue"] = 42
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
        fake_original_dataset.attrs["_FillValue"] = 42
        fake_original_dataset.attrs["dtype"] = np.dtype("float32")
        fake_original_dataset.encoding["preferred_chunks"] = "chocolate and peanut"
        fake_original_dataset["data"].encoding["standard name"] = "chris"

        assert dm.zarr_md_to_stac_format(fake_original_dataset) == {
            "dtype": "float32",
            "standard name": "chris",
            "Zarr chunk size": "chocolate and peanut",
        }

    @staticmethod
    def test_register_stac_item_already_exists(manager_class):
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
        md.store = mock.Mock(spec=store.Local)
        md.retrieve_stac = mock.Mock(return_value=["foo", "bar"])

        assert md.load_stac_metadata() == "foo"

        md.retrieve_stac.assert_called_once_with("DummyManager-daily", metadata.StacType.ITEM)

    @staticmethod
    def test_load_stac_metadata_pass_key(manager_class):
        md = manager_class()
        md.store = mock.Mock(spec=store.Local)
        md.retrieve_stac = mock.Mock(return_value=["foo", "bar"])

        assert md.load_stac_metadata(key="chiave") == "foo"

        md.retrieve_stac.assert_called_once_with("chiave", metadata.StacType.ITEM)

    @staticmethod
    def test_load_stac_metadata_timeout(manager_class):
        md = manager_class()
        md.store = mock.Mock(spec=store.Local)
        md.retrieve_stac = mock.Mock(side_effect=Timeout)

        assert md.load_stac_metadata() == {}

        md.retrieve_stac.assert_called_once_with("DummyManager-daily", metadata.StacType.ITEM)

    @staticmethod
    def test_load_stac_metadata_keyerror(manager_class):
        md = manager_class()
        md.store = mock.Mock(spec=store.StoreInterface)
        md.retrieve_stac = mock.Mock(side_effect=KeyError)

        assert md.load_stac_metadata() == {}

        md.retrieve_stac.assert_called_once_with("DummyManager-daily", metadata.StacType.ITEM)

    @staticmethod
    def test_set_zarr_metadata(manager_class):
        dataset = mock.Mock()
        md = manager_class()

        md.rename_data_variable = mock.Mock()
        renamed = md.rename_data_variable.return_value

        md.remove_unwanted_fields = mock.Mock()
        md.encode_vars = mock.Mock()
        md.set_initial_compression = mock.Mock()
        md.merge_in_outside_metadata = mock.Mock()

        dataset.attrs = {}
        assert md.set_zarr_metadata(dataset) is renamed

        md.rename_data_variable.assert_called_once_with(dataset)
        md.remove_unwanted_fields.assert_called_once_with(renamed)
        md.set_initial_compression.assert_called_once_with(renamed)
        md.encode_vars.assert_called_once_with(renamed)

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
        md.requested_zarr_chunks = {"latitude": 1, "longitude": 1, "time": 1}
        md.store = mock.Mock(spec=store.StoreInterface, has_existing=False)
        md.encode_vars(dataset)
        mv = md.missing_value

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": mv, "missing_value": mv}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": mv,
            "chunks": (1, 1, 1),
            "preferred_chunks": {"latitude": 1, "longitude": 1, "time": 1},
            "missing_value": mv,
        }
        assert dataset.time.encoding == {
            "long_name": "time",
            "calendar": "gregorian",
            "units": "days since 1975-07-07 0:0:0 0",
            "chunks": (1, 1, 1),
            "preferred_chunks": {"latitude": 1, "longitude": 1, "time": 1},
        }

    @staticmethod
    def test_encode_vars_chunks_has_existing(manager_class, fake_original_dataset):
        dataset = fake_original_dataset
        assert dataset.encoding == {}
        assert dataset["data"].encoding == {}
        assert dataset.time.encoding == {}

        md = manager_class()
        md.requested_zarr_chunks = {"latitude": 1, "longitude": 1, "time": 1}
        md.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        md.encode_vars(dataset)
        mv = md.missing_value

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": mv, "missing_value": mv}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": mv,
            "chunks": None,
            "preferred_chunks": None,
            "missing_value": mv,
        }
        assert dataset.time.encoding == {
            "long_name": "time",
            "calendar": "gregorian",
            "units": "days since 1975-07-07 0:0:0 0",
            "chunks": None,
            "preferred_chunks": None,
        }

    @staticmethod
    def test_encode_vars_forecast(manager_class, forecast_dataset):
        dataset = forecast_dataset
        assert dataset.encoding == {}
        assert dataset["data"].encoding == {}
        assert dataset.forecast_reference_time.encoding == {}

        md = manager_class()
        md.time_dim = "forecast_reference_time"
        md.requested_zarr_chunks = {"latitude": 1, "longitude": 1, "step": 1, "forecast_reference_time": 1}
        md.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        md.encode_vars(dataset)
        mv = md.missing_value

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": mv, "missing_value": mv}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": mv,
            "chunks": None,
            "preferred_chunks": None,
            "missing_value": mv,
        }
        assert dataset.forecast_reference_time.encoding == {
            "long_name": "initial time of forecast",
            "standard_name": "forecast_reference_time",
            "calendar": "proleptic_gregorian",
            "units": "days since 1975-07-07 0:0:0 0",
            "chunks": None,
            "preferred_chunks": None,
        }

    @staticmethod
    def test_encode_vars_hindcast(manager_class, hindcast_dataset):
        dataset = hindcast_dataset
        assert dataset.encoding == {}
        assert dataset["data"].encoding == {}
        assert dataset.hindcast_reference_time.encoding == {}

        md = manager_class()
        md.time_dim = "hindcast_reference_time"
        md.requested_zarr_chunks = {
            "latitude": 1,
            "longitude": 1,
            "step": 1,
            "forecast_reference_time": 1,
            "hindcast_reference_time": 1,
        }
        md.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        md.encode_vars(dataset) is dataset
        mv = md.missing_value

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": mv, "missing_value": mv}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": mv,
            "chunks": None,
            "preferred_chunks": None,
            "missing_value": mv,
        }
        assert dataset.hindcast_reference_time.encoding == {
            "long_name": "initial time of forecast",
            "standard_name": "hindcast_reference_time",
            "calendar": "proleptic_gregorian",
            "units": "days since 1975-07-07 0:0:0 0",
            "chunks": None,
            "preferred_chunks": None,
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
        md.store = mock.Mock(spec=store.StoreInterface, has_existing=True)
        md.requested_zarr_chunks = {"latitude": 1, "longitude": 1, "time": 1}
        md.encode_vars(dataset)
        mv = md.missing_value

        assert dataset.encoding == {"data": {"dtype": "<f4", "_FillValue": mv, "missing_value": mv}}
        assert dataset["data"].encoding == {
            "dtype": "<f4",
            "units": "parsecs",
            "_FillValue": mv,
            "chunks": None,
            "preferred_chunks": None,
            "missing_value": mv,
        }
        assert dataset.time.encoding == {
            "long_name": "time",
            "calendar": "gregorian",
            "units": "picoseconds since the big bang",
            "chunks": None,
            "preferred_chunks": None,
        }

    @staticmethod
    def test_merge_in_outside_metadata(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.now.return_value = datetime.datetime(2000, 1, 1, 0, 0, 0)
        dt_mock.timezone = datetime.timezone
        dataset = fake_original_dataset
        dataset.attrs = {"foo": "bar"}

        md = manager_class(static_metadata={"bar": "baz"})
        md.populate_metadata()
        md.store = mock.Mock(spec=store.StoreInterface)
        mock_dataset = mock.Mock()
        mock_dataset.attrs = {"date range": ("2000010100", "2021091600")}
        md.store.dataset = mock.Mock(return_value=mock_dataset)
        md.store.retrieve_metadata = mock.Mock(return_value=({"foo": "bar", "properties": {}}, "foo/bar"))
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
    def test_merge_in_outside_metadata_new_creation_date(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.now.return_value = datetime.datetime(2000, 1, 1, 0, 0, 0)
        dt_mock.timezone = datetime.timezone
        dataset = fake_original_dataset
        dataset.attrs = {"foo": "bar"}

        md = manager_class(static_metadata={"bar": "baz"})
        md.populate_metadata()
        md.store = mock.Mock(spec=store.StoreInterface)
        mock_dataset = mock.Mock()
        mock_dataset.attrs = {"date range": ("2000010100", "2021091600")}
        md.store.dataset = mock.Mock(return_value=mock_dataset)
        md.store.retrieve_metadata = mock.Mock(
            return_value=({"foo": "bar", "properties": {"created": "2020-01-01T0Z"}}, "foo/bar")
        )
        md.merge_in_outside_metadata(dataset)

        assert dataset.attrs == {
            "foo": "bar",
            "bar": "baz",
            "created": "2020-01-01T0Z",
            "update_previous_end_date": "2021091600",
            "date range": ("2000010100", "2022013100"),
            "update_date_range": ("2021091600", "2022013100"),
            "bbox": (100.0, 10.0, 130.0, 40.0),
            "update_is_append_only": True,
        }

    @staticmethod
    def test_merge_in_outside_metadata_no_previous_dataset(manager_class, fake_original_dataset, mocker):
        dt_mock = mocker.patch("gridded_etl_tools.utils.metadata.datetime")
        dt_mock.datetime.now.return_value = datetime.datetime(2000, 1, 1, 0, 0, 0)
        dt_mock.timezone = datetime.timezone
        dataset = fake_original_dataset
        dataset.attrs = {"foo": "bar"}

        md = manager_class(static_metadata={"bar": "baz"})
        md.populate_metadata()
        md.store = mock.Mock(spec=store.StoreInterface, has_existing=False)
        md.store.retrieve_metadata = mock.Mock(return_value=({"foo": "bar", "properties": {}}, "foo/bar"))
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
    @pytest.fixture
    def encoding_test_output():
        """Fixture to create and cleanup output directory for encoding tests"""
        output_path = pathlib.Path("./tests/unit/utils/output")
        pathlib.Path.mkdir(output_path, parents=True, exist_ok=True)
        yield output_path
        clean_up_input_paths(output_path)

    # LEGACY ENCODING CHANGE TESTS DISABLED AS THEY RELY ON ZARR V2
    # WHICH IS NOT AVAILABLE IN THE CI ENVIRONMENT

    # @staticmethod
    # def test_update_array_encoding_field(manager_class, fake_original_dataset, encoding_test_output):
    #     """Test modifying encoding of a single coordinate using zarr library directly"""

    #     # Setup
    #     dm = manager_class(="local")
    #     dm.store = mock.Mock(spec=store.StoreInterface)

    #     path_mock = mock.PropertyMock(return_value=encoding_test_output / "encoding_test.zarr")
    #     type(dm.store).path = path_mock

    #     dataset = fake_original_dataset
    #     coordinate_to_change = "latitude"

    #     # Write dataset to zarr
    #     dataset.to_zarr(dm.store.path, mode="w")

    #     # Record initial modification times of all arrays
    #     root = zarr.open_group(dm.store.path, mode="r")
    #     initial_mtimes = {
    #         name: os.path.getmtime(os.path.join(dm.store.path, name, ".zarray")) for name in root.array_keys()
    #     }

    #     # Get initial encoding for selected coordinate, xarray and zarr versions
    #     initial_lat_encoding_xr = xr.open_dataset(dm.store.path, engine="zarr")[coordinate_to_change].encoding.copy()
    #     with open(os.path.join(dm.store.path, coordinate_to_change, ".zarray"), "r") as f:
    #         initial_lat_encoding_zarr = json.load(f)

    #     # Verify initial setup changes at both Zarr and Xarray level
    #     assert initial_lat_encoding_xr["compressor"].cname == "lz4"
    #     assert initial_lat_encoding_zarr["compressor"]["cname"] == "lz4"

    #     # Wait a moment to ensure modification times would be different if files are touched
    #     sleep(0.1)

    #     # Modify the selected coordinate's encoding
    #     new_compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    #     dm.update_array_encoding(target_array=coordinate_to_change, update_key={"compressor": new_compressor})

    #     # Get final modification times to show that only the selected coordinate was modified
    #     root = zarr.open_group(dm.store.path, mode="r")
    #     final_mtimes = {
    #         name: os.path.getmtime(os.path.join(dm.store.path, name, ".zarray")) for name in root.array_keys()
    #     }

    #     # Get final encoding for the selected coordinate, xarray and zarr versions
    #     final_lat_encoding_xr = xr.open_dataset(dm.store.path, consolidated=True, engine="zarr")[
    #         coordinate_to_change
    #     ].encoding.copy()
    #     with open(os.path.join(dm.store.path, coordinate_to_change, ".zarray"), "r") as f:
    #         final_lat_encoding_zarr = json.load(f)

    #     # Verify selected coordinate, and only the selected coordinate, was modified
    #     for name in root.array_keys():
    #         if name == coordinate_to_change:
    #             assert (
    #                 final_mtimes[name] > initial_mtimes[name]
    #             ), "Changed coordinate .zarray should have been modified"
    #             assert root[name].compressor.cname == "zstd", "Changed coordinate should have new " \
    #             "compression settings"
    #         else:
    #             assert final_mtimes[name] == initial_mtimes[name], f"{name} .zarray should not have been modified"
    #             assert root[name].compressor.cname != "zstd", f"{name} compression should be unchanged"
    #     # Check .zarray files just to be sure
    #     assert final_lat_encoding_xr["compressor"].cname == "zstd", "Xarray should show new compression type"
    #     assert final_lat_encoding_zarr["compressor"]["cname"] == "zstd", "Zarr should show new compression type"

    # @staticmethod
    # def test_remove_array_encoding_field(manager_class, fake_original_dataset, encoding_test_output):
    #     """Test modifying encoding of a single coordinate using zarr library directly"""

    #     # Setup
    #     dm = manager_class(store="local")
    #     dm.store = mock.Mock(spec=store.StoreInterface)

    #     path_mock = mock.PropertyMock(return_value=encoding_test_output / "encoding_test.zarr")
    #     type(dm.store).path = path_mock

    #     dataset = fake_original_dataset
    #     coordinate_to_change = "latitude"
    #     key_to_remove = "missing_value"

    #     # Write dataset to zarr
    #     dataset[coordinate_to_change].encoding.update({key_to_remove: 42})
    #     dataset.to_zarr(dm.store.path, mode="w")

    #     # Record initial modification times of all arrays
    #     root = zarr.open_group(dm.store.path, mode="r")

    #     # Get initial encoding for selected coordinate, xarray and zarr versions
    #     initial_lat_encoding_xr = xr.open_dataset(dm.store.path, engine="zarr")[coordinate_to_change].encoding.copy()
    #     with open(os.path.join(dm.store.path, coordinate_to_change, ".zarray"), "r") as f:
    #         initial_lat_encoding_zarr = json.load(f)

    #     # Verify initial setup changes at both Zarr and Xarray level
    #     if key_to_remove in metadata.XARRAY_ENCODING_FIELDS:  # pragma NO COVER
    #         assert initial_lat_encoding_xr[key_to_remove] == 42
    #     if key_to_remove in metadata.ZARR_ENCODING_FIELDS:  # pragma NO COVER
    #         assert initial_lat_encoding_zarr[key_to_remove] == 42

    #     # Modify the selected coordinate's encoding
    #     dm.remove_array_encoding(target_array=coordinate_to_change, remove_key=key_to_remove)

    #     # Get final modification times to show that only the selected coordinate was modified
    #     root = zarr.open_group(dm.store.path, mode="r")

    #     # Get final encoding for the selected coordinate, xarray and zarr versions
    #     final_lat_encoding_xr = xr.open_dataset(dm.store.path, consolidated=True, engine="zarr")[
    #         coordinate_to_change
    #     ].encoding.copy()
    #     with open(os.path.join(dm.store.path, coordinate_to_change, ".zarray"), "r") as f:
    #         final_lat_encoding_zarr = json.load(f)

    #     if key_to_remove in metadata.XARRAY_ENCODING_FIELDS:  # pragma NO COVER
    #         assert (
    #             key_to_remove in initial_lat_encoding_xr
    #         ), f"Test setup should include {key_to_remove} in Xarray encoding"
    #         assert (
    #             key_to_remove not in final_lat_encoding_xr
    #         ), f"{key_to_remove} should have been removed from Xarray encoding"

    #     # Verify only selected coordinate was modified
    #     for name in root.array_keys():
    #         if name == coordinate_to_change and key_to_remove in metadata.ZARR_ENCODING_FIELDS:  # pragma NO COVER
    #             assert (
    #                 key_to_remove not in final_lat_encoding_zarr
    #             ), f"{key_to_remove} should have been removed from encoding"

    # @staticmethod
    # def test_modify_array_encoding_no_changes(manager_class, fake_original_dataset, encoding_test_output):
    #     """Test catching no changes specified"""

    #     # Setup
    #     dm = manager_class(store="local")
    #     dm.store = mock.Mock(spec=store.StoreInterface)

    #     path_mock = mock.PropertyMock(return_value=encoding_test_output / "encoding_test.zarr")
    #     type(dm.store).path = path_mock

    #     coordinate_to_change = "latitude"

    #     # Modify nothing
    #     with pytest.raises(ValueError, match="No changes to the array encoding were specified"):
    #         dm._modify_array_encoding(target_array=coordinate_to_change, update_key=None, remove_key=None)

    # @staticmethod
    # def test_update_array_encoding_not_coordinate(manager_class, fake_original_dataset, encoding_test_output):
    #     """Test catching non-coordinate array names"""

    #     # Setup
    #     dm = manager_class(store="local")
    #     dm.store = mock.Mock(spec=store.StoreInterface)
    #     path_mock = mock.PropertyMock(return_value=encoding_test_output / "encoding_test.zarr")
    #     type(dm.store).path = path_mock

    #     # Pass a non-coordinate array name
    #     with pytest.raises(
    #         ValueError, match="Target array precipitation is not in this dataset's list of coordinate dimensions"
    #     ):
    #         dm.update_array_encoding(target_array="precipitation", update_key={"fill_value": 999_999_999})

    #     # Pass a standard coordinate name but now the standard dims are wrong
    #     def mock_set_key_dims(self, *args, **kwargs):
    #         self.standard_dims = ["non-standard", "other non-standard"]

    #     dm.set_key_dims = mock.Mock(side_effect=mock_set_key_dims(dm))
    #     with pytest.raises(
    #         ValueError, match="Target array latitude is not in this dataset's list of coordinate dimensions"
    #     ):
    #         dm.update_array_encoding(target_array="latitude", update_key={"fill_value": 999_999_999})

    # @staticmethod
    # def test_update_array_encoding_invalid_encoding_field(manager_class, fake_original_dataset, encoding_test_output):
    #     """Test specifying an invalid encoding field according to Xarray and Zarr standards"""

    #     # Setup
    #     dm = manager_class(store="local")
    #     dm.standard_dims = ["latitude", "longitude", "time"]
    #     dm.store = mock.Mock(spec=store.StoreInterface)

    #     path_mock = mock.PropertyMock(return_value=encoding_test_output / "encoding_test.zarr")
    #     type(dm.store).path = path_mock

    #     coordinate_to_change = "longitude"

    #     # Pass a non-encoding array name
    #     with pytest.raises(ValueError, match="Invalid key"):
    #         dm.update_array_encoding(target_array=coordinate_to_change, update_key={"fill_er_up_value": 999_999_999})
