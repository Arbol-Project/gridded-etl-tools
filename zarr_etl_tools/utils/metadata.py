import json
import datetime
import shapely
import numcodecs

import numpy as np
import xarray as xr
from shapely import geometry

from .ipfs import IPFS
from .convenience import Convenience
from .store import IPLD

from abc import abstractmethod
from requests.exceptions import Timeout as TimeoutError


class Metadata(Convenience, IPFS):
    """
    Base class containing metadata creation and editing methods Zarr ETLs
    Includes STAC Metadata templates for Items, Collections, and the root Catalog
    """

    @classmethod
    def default_stac_item(cls) -> dict:
        """
        Default metadata template for STAC Items

        Returns
        -------
        dict
            The STAC Item metadata template with all pre-fillable values populated
        """
        return {
            "stac_version": "1.0.0",
            "type": "Feature",
            "id": cls.name(),
            "collection": cls.collection(),
            "links": [],
            "assets": {
                "zmetadata": {
                    "title": cls.name(),
                    "type": "application/json",
                    "description": f"Consolidated metadata file for {cls.name()} Zarr store, readable as a Zarr dataset by Xarray",
                    "roles": ["metadata", "zarr-consolidated-metadata"],
                }
            },
        }

    @property
    def default_stac_collection(self) -> dict:
        """
        Default metadata template for STAC collections

        Returns
        -------
        dict
            The STAC Collection metadata template with all pre-fillable values populated
        """
        return {
            "id": self.collection(),
            "type": "Collection",
            "stac_extensions": [
                "https://stac-extensions.github.io/projection/v1.0.0/schema.json"
            ],
            "stac_version": "1.0.0",
            "description": self.metadata["provider description"],
            "license": self.metadata["license"],
            "collection": self.collection(),
            "title": self.metadata["title"],
            "extent": {"spatial": {"bbox": [[]]}, "temporal": {"interval": [[]]}},
            "links": [
                {"rel": "self", "type": "application/json", "title": self.collection()}
            ],
            "providers": [
                {
                    "name": f"{self.host_organization}",
                    "description": "",  # provide description for your organization here
                    "roles": ["processor"],  #
                    "url": "",  # provide URL for your organization here
                },
                {
                    "name": self.metadata["publisher"],
                    "description": self.metadata["provider description"],
                    "roles": ["producer"],
                    "url": self.metadata["provider url"],
                    "terms of service": self.metadata["terms of service"],
                },
            ],
            "summaries": {"proj:epsg": self.metadata["coordinate reference system"]},
        }

    @classmethod
    def default_root_stac_catalog(cls) -> dict:
        """
        Default metadata template for the {self.host_organization} root Catalog

        Returns
        -------
        dict
            The STAC Catalog metadata template with all pre-fillable values populated
        """
        return {
            "id": f"{cls.host_organization}_data_catalog",
            "type": "Catalog",
            "title": f"{cls.host_organization} Data Catalog",
            "stac_version": "1.0.0",
            "description": f"This catalog contains all the data uploaded to {cls.host_organization} \
                that has been issued STAC-compliant metadata. The catalogs and collections describe single providers. \
                Each may contain one or multiple datasets. Each individual dataset has been documented as STAC Items.",
        }

    @abstractmethod
    def static_metadata(cls):
        """
        Placeholder for static metadata pertaining to each ETL
        """
        ...

    def create_root_stac_catalog(self):
        """
        Prepare a root catalog for your organization if it doesn't already exist.
        """
        root_catalog = self.default_root_stac_catalog()
        root_catalog_exists = self.check_stac_on_ipns(root_catalog["title"])
        if not root_catalog_exists:
            # Publish the root catalog if it doesn't exist
            self.info(f"Publishing root {self.host_organization} STAC Catalog")
            root_catalog_ipns_name = self.ipns_generate_name(key=root_catalog["title"])
            root_catalog["links"] = [
                {
                    "rel": "root",
                    "href": root_catalog_ipns_name,
                    "type": "application/json",
                    "title": f"{self.host_organization} root catalog",
                },
                {
                    "rel": "self",
                    "href": root_catalog_ipns_name,
                    "type": "application/json",
                    "title": f"{self.host_organization} root catalog",
                },
            ]
            self.ipns_publish(
                root_catalog["title"], self.ipfs_put(self.json_to_bytes(root_catalog))
            )
        else:
            self.info(
                f"Root {self.host_organization} STAC Catalog already exists, building collection"
            )
            ...

    def create_stac_collection(self, dataset: xr.Dataset, rebuild: bool = False):
        """
        Prepare a parent collection for the dataset the first time it's created.
        In order to check for the collection's existence we must populate the relevant dictionary first in order to use its attributes.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being published
        rebuild : bool, optional
            Whether to recreate the STAC Collection from scratch or not
        """
        stac_coll = self.default_stac_collection
        # Check if the collection already exists to avoid duplication of effort
        # Populate and publish a collection if one doesn't exist, or a rebuild was requested
        if rebuild or not self.check_stac_on_ipns(self.collection()):
            if rebuild:
                self.info(
                    "Collection rebuild requested. Creating new collection, pushing it to IPFS, and populating the main catalog"
                )
            # Populate data-driven attributes of the collection
            minx, miny, maxx, maxy = self.bbox_coords(dataset)
            stac_coll["extent"]["spatial"]["bbox"] = [[minx, miny, maxx, maxy]]
            stac_coll["extent"]["temporal"]["interval"] = [
                [
                    self.numpydate_to_py(dataset.time.values.min()).isoformat() + "Z",
                    self.numpydate_to_py(dataset.time.values.max()).isoformat() + "Z",
                ]
            ]
            # Create an IPNS name corresponding to the collection in order to note this in the collection and root catalog.
            coll_ipns_name = self.ipns_generate_name(key=self.collection())
            # Append collection to the root catalog if it doesn't already exist
            root_catalog, root_catalog_ipns_name, _ = self.ipns_retrieve_object(
                self.default_root_stac_catalog()["title"]
            )
            if not any(
                stac_coll["title"] in link_dict.values()
                for link_dict in root_catalog["links"]
            ):
                self.info(
                    f"Appending collection link to root {self.host_organization} STAC Catalog"
                )
                root_catalog["links"].append(
                    {
                        "rel": "child",
                        "href": coll_ipns_name,
                        "type": "application/json",
                        "title": stac_coll["title"],
                    }
                )
                self.ipns_publish(
                    root_catalog["title"],
                    self.ipfs_put(self.json_to_bytes(root_catalog)),
                )
            # Add links and publish this collection
            for link_dict in stac_coll["links"]:
                if link_dict["rel"] == "self":
                    link_dict["href"] = coll_ipns_name
            stac_coll["links"] = stac_coll["links"] + [
                {
                    "rel": "root",
                    "href": root_catalog_ipns_name,
                    "type": "application/json",
                    "title": f"{self.host_organization} root catalog",
                },
                {
                    "rel": "parent",
                    "href": root_catalog_ipns_name,
                    "type": "application/json",
                    "title": f"{self.host_organization} root catalog",
                },
            ]
            self.ipns_publish(
                self.collection(), self.ipfs_put(self.json_to_bytes(stac_coll))
            )
        else:
            self.info(
                "Found existing STAC Collection for this dataset, skipping creation"
            )

    def create_stac_item(self, dataset: xr.Dataset, dataset_hash: str):
        """
        Reformat previously prepared metadata and prepare key overviews of a dataset's spatial, temporal, and data dimensions into a STAC Item-compliant JSON.
        Push this JSON to IPFS and its relevant parent STAC Collection via `register_stac_item`

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being published
        dataset_hash : str
            The CID of the dataset being published
        """
        self.info("Creating STAC Item")
        stac_item = self.default_stac_item()
        # Filter relevant existing metadata from the Zarr, flatten to a single level, and output as a dict for the "properties" key in STAC metadata
        properties_dict = self.zarr_md_to_stac_format(dataset)
        # Include the array size
        properties_dict["array_size"] = {
            "latitude": dataset.latitude.size,
            "longitude": dataset.longitude.size,
            "time": dataset.time.size,
        }
        # Set up date items in STAC-compliant style
        properties_dict["start_datetime"] = (
            self.numpydate_to_py(dataset.time.values[0]).isoformat() + "Z"
        )
        properties_dict["end_datetime"] = (
            self.numpydate_to_py(dataset.time.values[-1]).isoformat() + "Z"
        )
        properties_dict["updated"] = (
            datetime.datetime.utcnow()
            .replace(tzinfo=datetime.timezone.utc)
            .isoformat()[:-13]
            + "Z"
        )
        # Populate the empty STAC Item
        minx, miny, maxx, maxy = self.bbox_coords(dataset)
        stac_item["bbox"] = [minx, miny, maxx, maxy]
        stac_item["geometry"] = json.dumps(
            shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy))
        )
        stac_item["assets"]["zmetadata"]["href"] = {"/": dataset_hash}
        stac_item["properties"] = properties_dict
        # Push to IPFS w/ link to IPNS name
        self.register_stac_item(stac_item)

    def zarr_md_to_stac_format(self, dataset: xr.Dataset) -> dict:
        """
        Reduce attributes metadata of a Zarr dataset to a flat dictionary with STAC-compliant keys and values

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being published

        Returns
        -------
        dict
            Single-level dictionary containing all relevant metadata fields from the Zarr dataset with STAC-compliant names and formats
        """
        # Filter down and name correctly needed attributes
        zarr_attrs = {
            "missing value",
            "dtype",
            "preferred_chunks",
            "Conventions",
            "spatial resolution",
            "spatial precision",
            "temporal resolution",
            "update cadence",
            "unit of measurement",
            "tags",
            "standard name",
            "long name",
            "date range",
            "dataset description",
            "dataset download url",
            "created",
            "updated",
            "finalization date",
            "analysis data download url",
            "reanalysis data download url",
            "input history cid",
            "input history JSON documentation",
        }
        all_md = {
            **dataset.attrs,
            **dataset.encoding,
            **dataset[self.data_var()].encoding,
        }
        all_md = {key: all_md[key] for key in zarr_attrs if key in all_md}
        rename_dict = {
            "preferred_chunks": "Zarr chunk size",
            "missing value": "Fill value",
            "Conventions": "CF convention",
        }
        properties_dict = {rename_dict.get(k, k): v for k, v in all_md.items()}
        # Reformat attributes
        for k, v in properties_dict.items():
            if type(v) is np.float32:
                properties_dict[k] = round(
                    float(v), 8
                )  # np.float32s aren't JSON serializable
        properties_dict["dtype"] = str(properties_dict["dtype"])

        return properties_dict

    def register_stac_item(self, stac_item: dict):
        """'
        Register a new dataset in an existing STAC Collection and/or replace a dataset's STAC Item with an updated one

        Parameters
        ----------
        stac_item : dict
            A dictionary of metadata prepared in `create_stac_item` for publication as a standalone STAC metadata file
        """
        self.info("Registering STAC Item in IPFS and its parent STAC Collection")
        # Generate variables of interest
        stac_coll, coll_ipns_name, _ = self.ipns_retrieve_object(str(self.collection()))
        # Register links
        stac_item["links"].append(
            {
                "rel": "parent",
                "href": coll_ipns_name,
                "type": "application/geo+json",
                "title": stac_coll["title"],
            }
        )
        try:
            # Check if an older version of the STAC Item exists and if so create a "previous" link for them in the new STAC Item
            (
                old_stac_item,
                item_ipns_name,
                old_item_ipfs_hash,
            ) = self.ipns_retrieve_object(self.json_key())
            stac_item["links"].append(
                {
                    "rel": "prev",
                    "href": str(
                        old_stac_item["assets"]["zmetadata"]["href"].set(
                            base=self._default_base
                        )
                    ),  # convert CID object back to hash str
                    "metadata href": {"/": old_item_ipfs_hash},
                    "type": "application/geo+json",
                    "title": stac_item["assets"]["zmetadata"]["title"],
                }
            )
            self.info("Updating 'previous' link in dataset's STAC Item")
        except (KeyError, TimeoutError):
            # Otherwise create an IPNS name for your new dataset
            self.info("No previous STAC Item found")
            item_ipns_name = self.ipns_generate_name(key=self.json_key())
        stac_item["links"].append(
            {
                "rel": "self",
                "href": item_ipns_name,
                "type": "application/geo+json",
                "title": f"{self.name()} metadata",
            }
        )
        # push final STAC Item to IPNS
        self.info("Pushing STAC Item to IPFS and linking to IPNS")
        self.ipns_publish(self.json_key(), self.ipfs_put(self.json_to_bytes(stac_item)))
        # register item as a link in the relevant collection, if not already there, and push updated collection to IPFS
        if any(
            stac_item["assets"]["zmetadata"]["title"] in link_dict["title"]
            for link_dict in stac_coll["links"]
            if "title" in link_dict.keys()
        ):
            self.info("Found existing STAC Item in STAC Collection")
            ...
        else:
            self.info(
                "No existing STAC Item found in this dataset's parent collection, inserting a child link"
            )
            # register IPNS name hrefs in both the Item and Collection and publish updated objects to IPNS
            stac_coll["links"].append(
                {
                    "rel": "item",
                    "href": item_ipns_name,
                    "type": "application/json",
                    "title": stac_item["assets"]["zmetadata"]["title"],
                }
            )
            self.ipns_publish(
                self.collection(), self.ipfs_put(self.json_to_bytes(stac_coll))
            )

    def update_stac_collection(self, dataset: xr.Dataset):
        """'
        Update fields in STAC Collection corresponding to a dataset

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being published
        """
        self.info("Updating dataset's parent STAC Collection")
        # Get existing STAC Collection and add new datasets to it, if necessary
        stac_coll = self.ipns_retrieve_object(str((self.collection())))[0]
        # Update spatial extent
        min_bbox_coords = np.minimum(
            stac_coll["extent"]["spatial"]["bbox"][0], [self.bbox_coords(dataset)]
        )[0][0:2]
        max_bbox_coords = np.maximum(
            stac_coll["extent"]["spatial"]["bbox"][0], [self.bbox_coords(dataset)]
        )[0][2:4]
        stac_coll["extent"]["spatial"]["bbox"] = [
            list(np.concatenate([min_bbox_coords, max_bbox_coords]))
        ]
        # Update time interval
        stac_coll["extent"]["temporal"]["interval"] = [
            [
                self.numpydate_to_py(dataset.time.values.min()).isoformat() + "Z",
                self.numpydate_to_py(dataset.time.values.max()).isoformat() + "Z",
            ]
        ]
        # Publish STAC Collection with updated fields
        self.ipns_publish(
            self.collection(), self.ipfs_put(self.json_to_bytes(stac_coll))
        )

    def only_update_metadata(self):
        """
        Standalone function to update STAC Item and Collection metadata for a previously published dataset without running a parse.
        This function is called via a `--only-metadata` flag to run_etl.py.
        """
        # Find the published dataset's IPFS hash and use it to pull in the dataset
        self.dataset_hash = self.latest_hash()
        self.info(f"Published dataset IPFS hash is {str(self.dataset_hash)}")
        dataset = self.zarr_hash_to_dataset(self.latest_hash())
        # generate metadata and insert it into the Zarr's attributes
        self.populate_metadata()
        dataset = self.set_zarr_metadata(dataset)
        # Update metadata for this dataset
        self.create_stac_item(dataset, self.dataset_hash)
        self.update_stac_collection(dataset)

    def load_stac_metadata(self, key: str = None) -> str | dict:
        """
        Return the latest version of a dataset's STAC Item from IPFS (via IPNS)

        Parameters
        ----------
        key : str
            The human readable IPNS key string referencing a given object

        Returns
        -------
        str | dict
            Either an IPNS name hash or an empty dictionary (if no IPNS name hash found)

        """
        if not key:
            key = self.json_key()
        try:
            return self.ipns_retrieve_object(key)[0]
        except (KeyError, TimeoutError):
            self.warn(
                "STAC metadata requested but no STAC object found at the provided key"
            )
            return {}

    # NON-STAC METADATA

    def populate_metadata(self):
        """
        Fill the metadata with values describing this set, using the static_metadata as a base template.
        """
        if hasattr(self, "metadata") and self.metadata is not None:
            self.metadata = self.metadata.update(self.static_metadata)
        else:
            self.metadata = self.static_metadata

    def set_zarr_metadata(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Function to append to or update key metadata information to the attributes and encoding of the output Zarr.
        Additionally filters out unwanted keys and fields.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update

        Returns
        -------
        dataset : xarray.Dataset
            The dataset being published, after metadata update

        """
        # Rename data variable to desired name, if necessary
        dataset = dataset.rename_vars(
            {[key for key in dataset.data_vars][0]: self.data_var()}
        )

        # Encode data types and missing value indicators for the data variable
        dataset = self.encode_vars(dataset)

        # Merge in relevant static / STAC metadata and create additional attributes
        dataset = self.merge_in_outside_metadata(dataset)

        # Set all fields to uncompressed and remove filters leftover from input files
        dataset = self.remove_unwanted_fields(dataset)

        # Xarray cannot export dictionaries or None as attributes (lists and tuples are OK)
        for attr in dataset.attrs.keys():
            if type(dataset.attrs[attr]) is dict or type(dataset.attrs[attr]) is None:
                dataset.attrs[attr] = ""

        return dataset

    def encode_vars(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Encode important data points related to the data variable.
        These are useful both for reference and to control Xarray's reading/writing behavior.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update

        Returns
        -------
        dataset : xarray.Dataset
            The dataset being published, after metadata update

        """
        # Encode fields for the data variable in the main encoding dict and the data var's own encoding dict (for thoroughness)
        dataset.encoding = {
            self.data_var(): {
                "dtype": self.data_var_dtype,
                "_FillValue": self.missing_value_indicator(),
                "missing_value": self.missing_value_indicator(),  # deprecated by NUG but maintained for backwards compatibility
            }
        }
        dataset[self.data_var()].encoding.update(
            {
                "units": self.unit_of_measurement,
                "_FillValue": self.missing_value_indicator(),
                "missing_value": self.missing_value_indicator(),  # deprecated by NUG but maintained for backwards compatibility
                "chunks": tuple(val for val in self.requested_zarr_chunks.values()),
                "preferred_chunks": self.requested_zarr_chunks,
            }
        )
        # Encode 'time' dimension with the Climate and Forecast Convention standards used by major climate data providers.
        dataset.time.encoding.update(
            {
                "long_name": "time",
                "calendar": "gregorian",
            }
        )
        if "units" not in dataset.time.encoding.keys():
            # reformat the dataset_start_date datetime to a CF compliant string if it exists....
            if hasattr(self, "dataset_start_date"):
                dataset.time.encoding.update(
                    {
                        "units": f"days since {self.dataset_start_date.isoformat().replace('T00:00:00', ' 0:0:0 0')}",
                    }
                )
            # otherwise use None to indicate this is unknown at present
            else:
                dataset.time.encoding.update(
                    {
                        "units": None,
                    }
                )

        return dataset

    def merge_in_outside_metadata(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Join static/STAC metadata fields to the dataset's metadata fields and adjust them as appropriate

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update

        Returns
        -------
        dataset : xarray.Dataset
            The dataset being published, after metadata update

        """
        # Load static metadata into the dataset's attributes
        dataset.attrs = {**dataset.attrs, **self.metadata}

        # Determine date to use for "created" field. On S3 and local, use current time. On IPLD, look for an existing creation time.
        now = (
            datetime.datetime.utcnow()
            .replace(tzinfo=datetime.timezone.utc)
            .isoformat()[:-13]
            + "Z"
        )
        if isinstance(self.store, IPLD):
            existing_stac_metadata = self.load_stac_metadata()
            if (
                existing_stac_metadata
                and "created" in existing_stac_metadata["properties"]
            ):
                created = existing_stac_metadata["properties"]["created"]
            else:
                created = now
        else:
            created = now
        dataset.attrs["created"] = created

        # Write the date range. Use existing data if possible to get the original start date. Even though the date range can be parsed by
        # opening the Zarr, it can be faster to access directly through the Zarr's `.zmetadata` file, so it gets written here.
        if isinstance(self.store, IPLD):
            # Set date range. Use start from previous dataset's metadata if it exists or the input dataset if this is the first run.
            try:
                stac_metadata = self.load_stac_metadata()
                dataset.attrs["update_previous_end_date"] = stac_metadata["properties"][
                    "date range"
                ][1]
                dataset.attrs["date range"] = (
                    stac_metadata["properties"]["date range"][0],
                    datetime.datetime.strftime(
                        self.get_date_range_from_dataset(dataset)[1], "%Y%m%d%H"
                    ),
                )
            except (KeyError, TimeoutError):
                dataset.attrs["update_previous_end_date"] = ""
                dataset.attrs["date range"] = self.date_range_to_string(
                    self.get_date_range_from_dataset(dataset)
                )
        else:
            # Use existing Zarr if possible, otherwise get the dates from the input dataset.
            if self.store.has_existing:
                previous_start, previous_end = self.store.dataset().attrs["date range"]
                dataset.attrs["update_previous_end_date"] = previous_end
                dataset.attrs["date range"] = (
                    previous_start,
                    self.date_range_to_string(
                        self.get_date_range_from_dataset(dataset)
                    )[1],
                )
            else:
                dataset.attrs["update_previous_end_date"] = ""
                dataset.attrs["date range"] = self.date_range_to_string(
                    self.get_date_range_from_dataset(dataset)
                )

        # Write the update date range by taking the date range of the xr.Dataset submitted to this function. This assumes this function
        # is called and the metadata is written before the original xr.Dataset is combined with the insert xr.Dataset.
        dataset.attrs["update_date_range"] = self.date_range_to_string(
            self.get_date_range_from_dataset(dataset)
        )

        # Include the bounding box in the Zarr metadata
        dataset.attrs["bbox"] = self.bbox_coords(dataset)

        # This defaults to `True`, so set it here and it will be overwritten when it is determined there is data at previously existing dates
        # being overwritten.
        dataset.attrs["update_is_append_only"] = True
        self.info("If updating, indicating the dataset is only appending data")

        return dataset

    def remove_unwanted_fields(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Remove filters, compression, and other unwanted encoding/metadata inheritances from input files

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update

        Returns
        -------
        dataset : xarray.Dataset
            The dataset being published, after metadata update

        """
        # Leave compression off for IPLD
        if self.store == "ipld":
            compressor = None
        else:
            compressor = numcodecs.Blosc()

        for coord in ["latitude", "longitude"]:
            dataset[coord].attrs.pop("chunks", None)
            dataset[coord].attrs.pop("preferred_chunks", None)
            dataset[coord].encoding.pop("_FillValue", None)
            dataset[coord].encoding.pop("missing_value", None)
            dataset[coord].encoding["compressor"] = compressor
        dataset[self.data_var()].encoding.pop("filters", None)
        dataset[self.data_var()].encoding["compressor"] = compressor

        return dataset
