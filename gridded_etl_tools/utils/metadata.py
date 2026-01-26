from enum import Enum
import json
import datetime
import typing
import zarr
import shapely.geometry
import numcodecs

import numpy as np
import xarray as xr

from .encryption import EncryptionFilter
from .convenience import Convenience

from requests.exceptions import Timeout as TimeoutError

XARRAY_ENCODING_FIELDS = [
    "dtype",
    "scale_factor",
    "add_offset",
    "_FillValue",
    "missing_value",
    "chunksizes",
    "zlib",
    "complevel",
    "shuffle",
    "fletcher32",
    "contiguous",
    "units",
    "calendar",
]

ZARR_ENCODING_FIELDS = [
    "chunks",
    "compressor",
    "filters",
    "order",
    "dtype",
    "fill_value",
    "object_codec",
    "dimension_separator",
]


class MetadataDescriptor:
    """
    Descriptor that provides lazy-loading metadata from initial_metadata.

    On first access, populates the instance's metadata dict from initial_metadata.
    Supports both reading and direct assignment.

    Usage
    -----
    Access ``self.metadata`` to get the metadata dict, which auto-populates
    from ``initial_metadata`` on first access::

        dm = MyManager()
        print(dm.metadata["name"])  # Auto-populated from initial_metadata

    Add dynamic fields directly::

        dm.metadata["custom_field"] = some_computed_value
    """

    def __set_name__(self, owner, name):
        self.private_name = f"__{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        # Check if metadata already exists on instance
        if not hasattr(obj, self.private_name) or getattr(obj, self.private_name) is None:
            # Auto-populate from initial_metadata
            setattr(obj, self.private_name, obj.initial_metadata.copy())

        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        # Allow direct assignment
        setattr(obj, self.private_name, value)


class StacType(Enum):
    ITEM = "datasets"
    COLLECTION = "collections"
    CATALOG = ""


class Metadata(Convenience):
    """
    Base class containing metadata creation and editing methods for Zarr ETLs.
    Includes STAC Metadata templates for Items, Collections, and the root Catalog.

    Metadata Handling
    -----------------
    The ``metadata`` attribute auto-populates from ``initial_metadata`` on first access.
    This means you can access ``self.metadata`` anywhere in your ETL without explicit
    initialization::

        dm = MyManager()
        print(dm.metadata["name"])  # Auto-populated from initial_metadata

    **Adding metadata in subclasses** (preferred for fields using class attributes):
    Override ``initial_metadata`` to add fields that depend on ``self``::

        @property
        def initial_metadata(self):
            metadata = super().initial_metadata
            metadata.update({
                "data download url": self.dataset_download_url,
                "custom field": self.some_class_attribute,
            })
            return metadata

    **Adding runtime metadata** (for fields computed during ETL execution):
    Assign directly to ``self.metadata`` where the data becomes available::

        # In publish_metadata() or similar, after extract() has run
        self.metadata["finalization date"] = self.input_history.get("final_date")

    The ``initial_metadata`` property defines the base metadata fields and is the
    primary extension point for dataset-specific metadata.
    """

    # Descriptor that auto-populates from initial_metadata on first access
    metadata = MetadataDescriptor()

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
            "id": cls.dataset_name,
            "collection": cls.collection_name,
            "links": [],
            "assets": {
                "zmetadata": {
                    "title": cls.dataset_name,
                    "type": "application/json",
                    "description": f"Consolidated metadata file for {cls.dataset_name} Zarr store, readable as a Zarr "
                    "dataset by Xarray",
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
            "id": self.collection_name,
            "type": "Collection",
            "stac_extensions": ["https://stac-extensions.github.io/projection/v1.0.0/schema.json"],
            "stac_version": "1.0.0",
            "description": self.metadata["provider description"],
            "license": self.metadata["license"],
            "collection": self.collection_name,
            "title": self.metadata["title"],
            "extent": {"spatial": {"bbox": [[]]}, "temporal": {"interval": [[]]}},
            "links": [{"rel": "self", "type": "application/json", "title": self.collection_name}],
            "providers": [
                {
                    "name": f"{self.organization}",
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
        Default metadata template for the {self.host_organization()} root Catalog

        Returns
        -------
        dict
            The STAC Catalog metadata template with all pre-fillable values populated
        """
        return {
            "id": f"{cls.organization}_data_catalog",
            "type": "Catalog",
            "title": f"{cls.organization} Data Catalog",
            "stac_version": "1.0.0",
            "description": f"This catalog contains all the data uploaded to {cls.organization} "
            "that has been issued STAC-compliant metadata. The catalogs and collections describe single "
            "providers. Each may contain one or multiple datasets. Each individual dataset has been documented as "
            "STAC Items.",
        }

    @property
    def initial_metadata(self):
        """
        Property returning a dictionary of metadata fields for this ETL.

        This is the primary extension point for managers to define dataset-specific metadata.
        Since this is a property with access to ``self``, it can reference class attributes
        and instance properties to build metadata dynamically.

        **Inheritance Pattern**:
        Subclasses should call ``super().initial_metadata`` and update the returned dict
        using ``metadata.update()`` rather than defining a complete new dictionary::

            @property
            def initial_metadata(self):
                metadata = super().initial_metadata
                metadata.update({
                    "data download url http": self.dataset_download_url_http,
                    "data download url ftp": self.dataset_download_ftp_server,
                })
                return metadata

        This pattern ensures that:

        1. Base metadata fields from parent classes are preserved
        2. Child classes can add or override specific fields
        3. Fields can reference ``self`` to access class attributes and properties

        **How it connects to self.metadata**:
        The ``self.metadata`` attribute auto-populates from this property on first access.
        Once populated, ``self.metadata`` is cached and won't call ``initial_metadata`` again.
        To add fields that depend on runtime state (e.g., data computed during ``extract()``),
        assign directly to ``self.metadata`` after it has been accessed::

            # In publish_metadata(), after extract() has populated self.input_history
            self.metadata["finalization date"] = self.input_history.get("final_date")

        Returns
        -------
        dict
            Dictionary of metadata key-value pairs to be written to Zarr attributes
            and STAC metadata.
        """
        return {
            "name": self.dataset_name,
            "updated": str(datetime.datetime.now(tz=datetime.timezone.utc)),
            "missing value": self.missing_value,
            "tags": self.tags,
            "standard name": self.standard_name,
            "long name": self.long_name,
            "unit of measurement": self.unit_of_measurement,
            "final lag in days": self.final_lag_in_days,
            "preliminary lag in days": self.preliminary_lag_in_days,
            "expected_nan_frequency": self.expected_nan_frequency,
            "coordinate reference system": self.coordinate_reference_system,
            "spatial resolution": self.spatial_resolution,
            "spatial precision": self.spatial_precision,
            "temporal resolution": str(self.time_resolution),
            "update cadence": str(self.update_cadence) if self.update_cadence else None,
            "provider url": self.provider_url,
            "data download url": self.data_download_url,
            "publisher": self.publisher,
            "title": self.title,
            "provider description": self.provider_description,
            "dataset description": self.dataset_description,
            "license": self.license,
            "terms of service": self.terms_of_service,
            "version": self.version,
            "release status": self.release_status,
            "region": self.region,
        }

    def check_stac_exists(self, title: str, stac_type: StacType) -> bool:
        """Check if a STAC entity exists in the backing store

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Type of STAC entity (Catalog, Collection or Item)

        Returns
        -------
        bool
            Whether the entity exists in the backing store
        """
        return self.store.metadata_exists(title, stac_type.value)

    def publish_stac(self, title: str, stac_content: dict, stac_type: StacType):
        """Publish a STAC entity to the backing store

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_content : dict
            content of the stac entity
        stac_type : StacType
            Type of STAC entity (Catalog, Collection or Item)
        """
        self.store.push_metadata(title, stac_content, stac_type.value)

    def retrieve_stac(self, title: str, stac_type: StacType) -> tuple[dict, str]:
        """Retrieve a STAC entity and its href from the backing store

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Type of STAC entity (Catalog, Collection or Item)

        Returns
        -------
        tuple[dict, str | pathlib.Path]
            tuple of STAC content and the href for the STAC
        """
        return self.store.retrieve_metadata(title, stac_type.value)

    def get_href(self, title: str, stac_type: StacType) -> str:
        """Get a STAC entity's href from the backing store. Might be
        an IPNS name, a local path or a s3 path depending on the store

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Type of STAC entity (Catalog, Collection or Item)

        Returns
        -------
        str
            string representation of href.
        """
        return self.store.get_metadata_path(title, stac_type.value)

    def create_root_stac_catalog(self):
        """
        Prepare a root catalog for your organization if it doesn't already exist.
        """
        root_catalog = self.default_root_stac_catalog()
        root_catalog_exists = self.check_stac_exists(root_catalog["title"], StacType.CATALOG)
        if not root_catalog_exists:
            # Publish the root catalog if it doesn't exist
            self.info(f"Publishing root {self.organization} STAC Catalog")
            catalog_href = self.get_href(root_catalog["title"], StacType.CATALOG)
            root_catalog["links"] = [
                {
                    "rel": "root",
                    "href": catalog_href,
                    "type": "application/json",
                    "title": f"{self.organization} root catalog",
                },
                {
                    "rel": "self",
                    "href": catalog_href,
                    "type": "application/json",
                    "title": f"{self.organization} root catalog",
                },
            ]
            self.publish_stac(root_catalog["title"], root_catalog, StacType.CATALOG)
        else:
            self.info(f"Root {self.organization} STAC Catalog already exists, building collection")

    def create_stac_collection(self, dataset: xr.Dataset, rebuild: bool = False):
        """
        Prepare a parent collection for the dataset the first time it's created. In order to check for the collection's
        existence we must populate the relevant dictionary first in order to use its attributes.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being published
        rebuild : bool, optional
            Whether to recreate the STAC Collection from scratch or not
        Returns
        -------
        bool
            Whether the stac collection was created
        """
        stac_coll = self.default_stac_collection
        # Check if the collection already exists to avoid duplication of effort
        # Populate and publish a collection if one doesn't exist, or a rebuild was requested
        if rebuild or not self.check_stac_exists(self.collection_name, StacType.COLLECTION):
            if rebuild:
                self.info(
                    "Collection rebuild requested. Creating new collection, pushing it to the store, "
                    "and populating the main catalog"
                )
            # Populate data-driven attributes of the collection
            minx, miny, maxx, maxy = self.bbox_coords(dataset)
            stac_coll["extent"]["spatial"]["bbox"] = [[minx, miny, maxx, maxy]]
            stac_coll["extent"]["temporal"]["interval"] = [
                [
                    self.numpydate_to_py(dataset[self.time_dim].values.min()).isoformat() + "Z",
                    self.numpydate_to_py(dataset[self.time_dim].values.max()).isoformat() + "Z",
                ]
            ]
            # Create an href corresponding to the collection in order to note this in the collection and root catalog.
            href = self.get_href(self.collection_name, StacType.COLLECTION)
            # Append collection to the root catalog if it doesn't already exist
            root_catalog, root_catalog_href = self.retrieve_stac(
                self.default_root_stac_catalog()["title"], StacType.CATALOG
            )
            if not any(stac_coll["title"] in link_dict.values() for link_dict in root_catalog["links"]):
                self.info(f"Appending collection link to root {self.organization} STAC Catalog")
                root_catalog["links"].append(
                    {
                        "rel": "child",
                        "href": href,
                        "type": "application/json",
                        "title": stac_coll["title"],
                    }
                )
                self.publish_stac(
                    root_catalog["title"],
                    root_catalog,
                    StacType.CATALOG,
                )
            # Add links and publish this collection
            for link_dict in stac_coll["links"]:
                if link_dict["rel"] == "self":
                    link_dict["href"] = href
            stac_coll["links"] = stac_coll["links"] + [
                {
                    "rel": "root",
                    "href": root_catalog_href,
                    "type": "application/json",
                    "title": f"{self.organization} root catalog",
                },
                {
                    "rel": "parent",
                    "href": root_catalog_href,
                    "type": "application/json",
                    "title": f"{self.organization} root catalog",
                },
            ]
            self.publish_stac(self.collection_name, stac_coll, StacType.COLLECTION)
        else:
            self.info("Found existing STAC Collection for this dataset, skipping creation and updating instead")
            self.update_stac_collection(dataset)

    def create_stac_item(self, dataset: xr.Dataset):
        """
        Reformat previously prepared metadata and prepare key overviews of a dataset's spatial, temporal,
        and data dimensions into a STAC Item-compliant JSON. Push this JSON to the store and its relevant parent
        STAC Collection via `register_stac_item`

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being published
        """
        self.info("Creating STAC Item")
        stac_item = self.default_stac_item()
        # Filter relevant existing metadata from the Zarr, flatten to a single level, and output as a dict for the
        # "properties" key in STAC metadata
        properties_dict = self.zarr_md_to_stac_format(dataset)

        properties_dict["dataset_category"] = self.dataset_category

        # Include the array size
        spatial_size_dict = {dim_name: dataset[dim_name].size for dim_name in self.spatial_dims}

        properties_dict["array_size"] = {
            **spatial_size_dict,
            self.time_dim: dataset[self.time_dim].size,
        }
        if self.time_dim == "forecast_reference_time":
            properties_dict["array_size"].update({"step": dataset.step.size})
        # Set up date items in STAC-compliant style
        properties_dict["start_datetime"] = self.numpydate_to_py(dataset[self.time_dim].values[0]).isoformat() + "Z"
        properties_dict["end_datetime"] = self.numpydate_to_py(dataset[self.time_dim].values[-1]).isoformat() + "Z"
        properties_dict["updated"] = (
            datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"
        )
        # Populate the empty STAC Item
        minx, miny, maxx, maxy = self.bbox_coords(dataset)
        stac_item["bbox"] = [minx, miny, maxx, maxy]
        stac_item["geometry"] = json.dumps(shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy)))
        zarr_href = str(self.store.path)

        stac_item["assets"]["zmetadata"]["href"] = zarr_href
        stac_item["properties"] = properties_dict
        # Push to backing store w/ link to href
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
            Single-level dictionary containing all relevant metadata fields from the Zarr dataset with STAC-compliant
            names and formats
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
            **dataset[self.data_var].encoding,
        }
        rename_dict = {
            "preferred_chunks": "Zarr chunk size",
            "missing value": "Fill value",
            "Conventions": "CF convention",
        }
        properties_dict = {rename_dict.get(key, key): all_md[key] for key in zarr_attrs if key in all_md}
        # Reformat attributes
        for k, v in properties_dict.items():
            if type(v) is np.float32:
                properties_dict[k] = round(float(v), 8)  # np.float32s aren't JSON serializable

        if "dtype" in properties_dict:
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
        self.info("Registering STAC Item in the store and its parent STAC Collection")
        # Generate variables of interest
        stac_coll, collection_href = self.retrieve_stac(self.collection_name, StacType.COLLECTION)
        # Register links
        stac_item["links"].append(
            {
                "rel": "parent",
                "href": str(collection_href),
                "type": "application/geo+json",
                "title": stac_coll["title"],
            }
        )
        # Check if an older version of the STAC Item exists and if so create a "previous" link for them in the new STAC
        # Item
        try:
            old_stac_item, href = self.retrieve_stac(self.key(), StacType.ITEM)

        # TODO: It would be better to not have KeyError in here, as it it's easy for that to be a different exception
        # than the one you think you're catching. It would be better to have retrieve_stac return
        # None if they can't find the key, and then use an if statement to check the return value for None.
        except (KeyError, TimeoutError, FileNotFoundError):
            # Otherwise create a STAC name for your new dataset
            self.info("No previous STAC Item found")
            href = self.get_href(self.key(), StacType.ITEM)

        stac_item["links"].append(
            {
                "rel": "self",
                "href": str(href),
                "type": "application/geo+json",
                "title": f"{self.dataset_name} metadata",
            }
        )
        # push final STAC Item to backing store
        self.info("Pushing STAC Item to backing store")
        self.publish_stac(self.key(), stac_item, StacType.ITEM)
        # register item as a link in the relevant collection, if not already there,
        # and push updated collection to the store
        if any(
            stac_item["assets"]["zmetadata"]["title"] in link_dict["title"]
            for link_dict in stac_coll["links"]
            if "title" in link_dict.keys()
        ):
            self.info("Found existing STAC Item in STAC Collection")
        else:
            self.info("No existing STAC Item found in this dataset's parent collection, inserting a child link")
            # register hrefs in both the Item and Collection and publish updated objects
            stac_coll["links"].append(
                {
                    "rel": "item",
                    "href": str(href),
                    "type": "application/json",
                    "title": stac_item["assets"]["zmetadata"]["title"],
                }
            )
            self.publish_stac(self.collection_name, stac_coll, StacType.COLLECTION)

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
        stac_coll = self.retrieve_stac(self.collection_name, StacType.COLLECTION)[0]
        # Update spatial extent
        min_bbox_coords = np.minimum(stac_coll["extent"]["spatial"]["bbox"][0], [self.bbox_coords(dataset)])[0][0:2]
        max_bbox_coords = np.maximum(stac_coll["extent"]["spatial"]["bbox"][0], [self.bbox_coords(dataset)])[0][2:4]
        stac_coll["extent"]["spatial"]["bbox"] = [list(np.concatenate([min_bbox_coords, max_bbox_coords]))]
        # Update time interval
        stac_coll["extent"]["temporal"]["interval"] = [
            [
                self.numpydate_to_py(dataset[self.time_dim].values.min()).isoformat() + "Z",
                self.numpydate_to_py(dataset[self.time_dim].values.max()).isoformat() + "Z",
            ]
        ]
        # Publish STAC Collection with updated fields
        self.publish_stac(self.collection_name, stac_coll, StacType.COLLECTION)

    def load_stac_metadata(self, key: str = None) -> str | dict:
        """
        Return the latest version of a dataset's STAC Item from S3

        Parameters
        ----------
        key : str
            The s3 path referencing a given STAC Item

        Returns
        -------
        str | dict
            Either a STAC Item or an empty dictionary (if no STAC Item found)

        """
        if not key:
            key = self.key()
        try:
            stac, _ = self.retrieve_stac(key, StacType.ITEM)
            return stac
        except (KeyError, TimeoutError, FileNotFoundError):
            self.warn(
                f"STAC metadata requested at {key} but no STAC object found at the provided key. "
                "Returning empty dictionary"
            )
            return {}

    # NON-STAC METADATA

    def set_zarr_metadata(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Append or update key metadata information to the attributes and encoding of the output Zarr.
        Additionally filters out unwanted or invalid keys and fields.

        This method is called during the parse phase and will access ``self.metadata``,
        which auto-populates from ``initial_metadata`` on first access. Any dynamic
        metadata fields should be added to ``self.metadata`` before this method is called,
        typically in ``publish_metadata()`` or a custom transform method.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update

        Returns
        -------
        dataset : xarray.Dataset
            The dataset being published, after metadata update
        """
        # Rename data variable to desired name, if necessary.
        dataset = self.rename_data_variable(dataset)
        # Set all fields to uncompressed and remove filters leftover from input files
        self.remove_unwanted_fields(dataset)
        # Consistently apply Blosc lz4 compression to all coordinates and the data variable
        self.set_initial_compression(dataset)
        # Encode data types and missing value indicators for the data variable
        self.encode_ds(dataset)
        # Merge in relevant static / STAC metadata and create additional attributes
        self.merge_in_outside_metadata(dataset)

        return dataset

    def rename_data_variable(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Rename data variables to desired name if they are not already set
        Normally this would take place w/in `postprocess_zarr` but as it's a classmethod it can't
        call dataset specific methods and properties. Therefore we run it here.

        Rename will return a ValueError if the name already exists, in which case we pass

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-rename

        Returns
        -------
        dataset : xarray.Dataset
            The dataset being published, post-rename
        """
        data_var = first(dataset.data_vars)
        try:
            return dataset.rename_vars({data_var: self.data_var})
        except ValueError:
            self.info(f"Duplicate name conflict detected during rename, leaving {data_var} in place")
            return dataset

    def encode_ds(self, dataset: xr.Dataset):
        """
        Encode important data points related to the data and time variables. These are useful both for reference and to
        control Xarray's reading/writing behavior.

        More recent versions of Xarray + Dask choke when updating with pre-chunked update datasets, so all chunking
        information (as well as chunking itself) will be aggressively removed.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update
        """
        # Encode fields for the data variable in the main encoding dict and the data var's own encoding dict (for
        # thoroughness)
        dataset.encoding = {
            self.data_var: {
                "dtype": self.data_var_dtype,
                "_FillValue": self.missing_value,
            }
        }
        dataset[self.data_var].encoding.update(
            {
                "dtype": self.data_var_dtype,
                "units": self.unit_of_measurement,
                "_FillValue": self.missing_value,
            }
        )
        # More recent versions of Xarray + Dask choke when updating with pre-chunked update datasets,
        # so all chunking information (as well as chunking itself) must be aggressively removed.
        if self.store.has_existing and not self.rebuild_requested:
            chunks = preferred_chunks = None
        else:
            # Initial parses need chunking information to be present in the encoding dict
            chunks = tuple(val for val in self.requested_zarr_chunks.values())
            preferred_chunks = self.requested_zarr_chunks
        for aspect in list(dataset.dims) + [self.data_var]:
            dataset[aspect].encoding.update(
                {
                    "chunks": chunks,
                    "preferred_chunks": preferred_chunks,
                }
            )
        # Encode 'time' dimension with the Climate and Forecast Convention standards used by major climate data
        # providers.
        if "time" in dataset:
            dataset.time.encoding.update(
                {
                    "long_name": "time",
                    "calendar": "gregorian",
                }
            )
        elif "forecast_reference_time" in dataset and self.time_dim == "forecast_reference_time":
            dataset.forecast_reference_time.encoding.update(
                {
                    "long_name": "initial time of forecast",
                    "standard_name": "forecast_reference_time",
                    "calendar": "proleptic_gregorian",
                }
            )
        elif "hindcast_reference_time" in dataset and self.time_dim == "hindcast_reference_time":  # pragma NO BRANCH
            dataset.hindcast_reference_time.encoding.update(
                {
                    "long_name": "initial time of forecast",
                    "standard_name": "hindcast_reference_time",
                    "calendar": "proleptic_gregorian",
                }
            )

        if "units" not in dataset[self.time_dim].encoding.keys():
            # reformat the dataset_start_date datetime to a CF compliant string if it exists....
            dataset[self.time_dim].encoding.update(
                {
                    "units": f"days since {self.dataset_start_date.isoformat().replace('T00:00:00', ' 0:0:0 0')}",
                }
            )

        # Ensure a type-appropriate _FillValue is populated to each coordinate
        for coord in dataset.coords:
            if coord != self.data_var:
                if coord == self.time_dim or np.issubdtype(
                    dataset[coord].dtype, np.integer
                ):  # will include time dim(s)
                    dataset[coord].encoding[
                        "_FillValue"
                    ] = -9223372036854775808  # implausible value to prevent accidental masking
                else:
                    dataset[coord].encoding["_FillValue"] = "NaN"

        # Encrypt variable data if requested
        if self.encryption_key is not None:
            encoding = dataset[self.data_var].encoding
            filters = encoding.get("filters")
            if not filters:
                encoding["filters"] = filters = []
            filters.append(EncryptionFilter(self.encryption_key))

    def merge_in_outside_metadata(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Join metadata fields from ``self.metadata`` to the dataset's attributes.

        This method merges the auto-populated ``self.metadata`` dict (from ``initial_metadata``)
        plus any dynamically added fields into the dataset's attributes. It also adds
        computed fields like date range and bounding box.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update
        """
        # Load static metadata into the dataset's attributes
        dataset.attrs = {**dataset.attrs, **self.metadata}

        # Get existing stac_metadata, if possible
        stac_metadata = self.load_stac_metadata()

        # Determine date to use for "created" field. On S3 and local, use current time.
        if stac_metadata and "created" in stac_metadata["properties"]:
            dataset.attrs["created"] = stac_metadata["properties"]["created"]
        else:
            dataset.attrs["created"] = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()[:-7] + "Z"

        # Write the date range. Use existing data if possible to get the original start date. Even though the date
        # range can be parsed by opening the Zarr, it can be faster to access directly through the Zarr's `.zmetadata`
        # file, so it gets written here.
        # Use existing Zarr if possible, otherwise get the dates from the input dataset.
        if self.store.has_existing:
            previous_start, previous_end = self.store.dataset().attrs["date range"]
            dataset.attrs["update_previous_end_date"] = previous_end
            dataset.attrs["date range"] = (
                previous_start,
                self.date_range_to_string(self.get_date_range_from_dataset(dataset))[1],
            )
        else:
            dataset.attrs["update_previous_end_date"] = ""
            dataset.attrs["date range"] = self.date_range_to_string(self.get_date_range_from_dataset(dataset))

        # Write the update date range by taking the date range of the xr.Dataset submitted to this function. This
        # assumes this function is called and the metadata is written before the original xr.Dataset is combined with
        # the insert xr.Dataset.
        dataset.attrs["update_date_range"] = self.date_range_to_string(self.get_date_range_from_dataset(dataset))

        # Include the bounding box in the Zarr metadata
        dataset.attrs["bbox"] = self.bbox_coords(dataset)

        # This defaults to `True`, so set it here and it will be overwritten when it is determined there is data at
        # previously existing dates being overwritten.
        dataset.attrs["update_is_append_only"] = True
        self.info("If updating, indicating the dataset is only appending data")

    def remove_unwanted_fields(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Remove filters, compression, and other unwanted encoding/metadata inheritances from input files

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update
        """
        for coord in self.spatial_dims:
            dataset[coord].attrs.pop("chunks", None)
            dataset[coord].attrs.pop("preferred_chunks", None)
            dataset[coord].encoding.pop("_FillValue", None)
            dataset[coord].encoding.pop("missing_value", None)
        dataset[self.data_var].encoding.pop("filters", None)

    def set_initial_compression(self, dataset: xr.Dataset):
        """
        If the dataset is new and uses compression, compress all coordinate and data variables in the dataset.
        Does nothing if the dataset is not new; use `update_array_encoding` to change compression in this case.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset being published, pre-metadata update
        """
        if not self.use_compression:
            compressor = None
        elif self.output_zarr3:
            compressor = zarr.codecs.BloscCodec(cname="lz4")
        else:
            compressor = numcodecs.Blosc()

        if not self.store.has_existing:
            for coord in dataset.coords:
                dataset[coord].encoding["compressors"] = compressor
            dataset[self.data_var].encoding["compressors"] = compressor

    def update_array_encoding(
        self,
        target_array: str,
        update_key: dict,
    ):  # pragma NO COVER -- legacy function that needs Zarr v2 to work
        """
        Update an array encoding field in the dataset's Zarr store.

        Parameters
        ----------
        target_array : str
            The name of the array to modify
        update_key : dict
            A key:value pair to insert into or update in the array encoding
        """
        self._modify_array_encoding(target_array, update_key=update_key, remove_key=None)

    def remove_array_encoding(
        self,
        target_array: str,
        remove_key: str,
    ):  # pragma NO COVER -- legacy function that needs Zarr v2 to work
        """
        Remove an array encoding field from the dataset's Zarr store.

        Parameters
        ----------
        target_array : str
            The name of the array to modify
        remove_key : str
            The key to remove from the array encoding
        """
        self._modify_array_encoding(target_array, update_key=None, remove_key=remove_key)

    def _modify_array_encoding(
        self,
        target_array: str,
        update_key: dict | None = None,
        remove_key: str | None = None,
    ):  # pragma NO COVER -- legacy function that needs Zarr v2 to work
        """
        Modify the encoding of an array -- coordinate or data variable -- in the dataset's Zarr store.

        NOTE this function is intended for use "gardening" production Zarrs and should be used with caution.
        It collides with some points of incongruity between Zarr and Xarray's encoding standards.
        Most notably, Zarr does not support the `missing_value` field in the encoding dict, unlike Xarray,
        which does for backwards compatibility with NetCDFs, although this functionality is now deprecated.
        Also, Zarr will silently convert "_FillValue" to "fill_value" whereas Xarray will represent it as "_FillValue".

        Handling all of this is messy -- so again, this function should be used with caution. Test first!

        Parameters
        ----------
        target_array : str
            The name of the array to modify
        update_key : dict | None, optional
            A key:value pair to insert into or update in the array encoding, by default None
        remove_key : str | None, optional
            A key to remove from the array encoding and attributes, by default None
        """
        # Exit if no changes to the array encoding were specified
        if not any([update_key, remove_key]):
            raise ValueError("No changes to the array encoding were specified")

        # If the key does not match a valid encoding field, raise an error
        if update_key and list(update_key.keys())[0] not in XARRAY_ENCODING_FIELDS + ZARR_ENCODING_FIELDS:
            raise ValueError(f"Invalid key {list(update_key.keys())[0]} for array encoding")

        # Exit if the target array is not a coordinate dimension;
        # any changes to data variable would involve effectively re-writing the entire Zarr
        # and should be handled with a re-parse
        self.set_key_dims()
        if target_array not in self.standard_dims:
            raise ValueError(
                f"Target array {target_array} is not in this dataset's "
                f"list of coordinate dimensions: {self.standard_dims}"
            )

        # Open the Zarr store and get the old array
        root = zarr.open_group(self.store.path, mode="r+")
        old_array = root[target_array]
        data = old_array[:]

        array_kwargs = {
            "shape": old_array.shape,
            "chunks": old_array.chunks,
            "dtype": old_array.dtype,
            "compressor": old_array.compressor,
            "fill_value": old_array.fill_value,
            "order": old_array.order,
        }

        # Preserve _ARRAY_DIMENSIONS in .zattrs
        array_attrs = dict(old_array.attrs).copy()
        if "_ARRAY_DIMENSIONS" not in array_attrs:  # pragma NO COVER
            array_attrs["_ARRAY_DIMENSIONS"] = [target_array]

        # Make changes to the array encoding (the .zarray file)
        if update_key:
            array_kwargs.update(update_key)

        # Make changes to the array attributes (the .zattrs file)
        if remove_key:
            array_attrs.pop(remove_key, None)

        # Populate and consolidate changes
        del root[target_array]
        new_array = root.create_dataset(target_array, **array_kwargs)
        new_array[:] = data
        new_array.attrs.update(array_attrs)

        zarr.consolidate_metadata(store=self.store.path)


def first(i: typing.Iterable):
    """Return the first item in an iterable

    Raises:
        StopIteration if iterable is empty.
    """
    return next(iter(i))
