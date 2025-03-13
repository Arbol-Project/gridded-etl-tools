# The annotations dict and TYPE_CHECKING var are necessary for referencing types that aren't fully imported yet. See
# https://peps.python.org/pep-0563/
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma NO COVER
    from .. import dataset_manager

import datetime
import json

import shutil
import s3fs
import xarray as xr
import pathlib
import fsspec
import collections

from abc import abstractmethod, ABC
from typing import Any


class StoreInterface(ABC):
    """
    Base class for an interface that can be used to access a dataset's Zarr.

    Zarrs can be stored in different types of data stores, for example S3 and the local filesystem, each of
    which is accessed slightly differently in Python. This class abstracts the access to the underlying data store by
    providing functions that access the Zarr on the store in a uniform way, regardless of which is being used.
    """

    def __init__(self, dm: dataset_manager.DatasetManager):
        """
        Create a new `StoreInterface`. Pass the dataset manager this store is being associated with, so the interface
        will have access to dataset properties.

        Parameters
        ----------
        dm : dataset_manager.DatasetManager
            The dataset to be read or written.
        """
        self.dm = dm

    @abstractmethod
    def mapper(self, **kwargs) -> collections.abc.MutableMapping:
        """
        Parameters
        ----------
        **kwargs : dict
            Implementation specific keywords. TODO: standardize interface across implementations

        Returns
        -------
        collections.abc.MutableMapping
            A key/value mapping of files to contents
        """

    @property
    @abstractmethod
    def has_existing(self) -> bool:
        """
        Returns
        -------
        bool
            Return `True` if there is existing data for this dataset on the store.
        """

    @abstractmethod
    def metadata_exists(self, title: str, stac_type: str) -> bool:
        """
        Check whether metadata exists at a given local path

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        bool
            Whether metadata exists at path
        """

    @abstractmethod
    def push_metadata(self, title: str, stac_content: dict, stac_type: str):
        """
        Publish metadata entity to s3 store. Tracks historical state
        of metadata as well

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_content : dict
            content of the stac entity
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)
        """

    @abstractmethod
    def retrieve_metadata(self, title: str, stac_type: str) -> tuple[dict, str]:
        """
        Retrieve metadata entity from local store

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        dict
            Tuple of content of stac entity as dict and the local path as a string
        """

    @abstractmethod
    def get_metadata_path(self, title: str, stac_type: str) -> str:
        """
        Get the s3 path for a given STAC title and type

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        str
            The s3 path for this entity as a string
        """

    @property
    def path(self) -> str:
        """
        Get the S3-protocol URL to the parent `DatasetManager`'s Zarr .

        Returns
        -------
        str
            A URL string starting with "s3://" followed by the path to the Zarr.
        """

    def dataset(self, **kwargs) -> xr.Dataset | None:
        """
        Parameters
        ----------
        **kwargs
            Implementation specific keyword arguments to forward to the Zarr store. S3 and Local accept
            `refresh`.

        Returns
        -------
        xr.Dataset | None
            The dataset opened in xarray or None if there is no dataset currently stored.
        """
        if self.has_existing:
            return xr.open_zarr(store=self.path, **kwargs)
        else:
            return None

    @abstractmethod
    def write_metadata_only(self, attributes: dict):
        """
        Writes the metadata to the stored Zarr.

        Open the Zarr's `.zmetadata` and `.zattr` files with the JSON library, update the values with the values in the
        given dict, and write the files.

        These changes will be reflected in the attributes dict of subsequent calls to `DatasetManager.store.dataset`
        without needing to call `DatasetManager.to_zarr`.

        Parameters
        ----------
        attributes
            A dict of metadata attributes to add or update to the Zarr
        """


class S3(StoreInterface):
    """
    Provides an interface for reading and writing a dataset's Zarr on S3.

    To connect to a Zarr on S3 (i.e., at "s3://[bucket]/[dataset_key].zarr"), create a new S3 object using a
    `dataset_manager.DatasetManager` object and bucket name, and define both `AWS_ACCESS_KEY_ID` and
    `AWS_SECRET_ACCESS_KEY` in the ~/.aws/credentials file or shell environment.

    After initialization, use the member functions to access the Zarr.
    """

    def __init__(self, dm: dataset_manager.DatasetManager, bucket: str):
        """
        Get an interface to a dataset's Zarr on S3 in the specified bucket.

        Parameters
        ----------
        dm : dataset_manager.DatasetManager
            The dataset to be read or written.
        bucket : str
            The name of the S3 bucket to connect to (s3://[bucket])
        """
        super().__init__(dm)
        if not bucket:
            raise ValueError("Must provide bucket name if parsing to S3")
        self.bucket = bucket

    def fs(self, refresh: bool = False, profile: str | None = None) -> s3fs.S3FileSystem:
        """
        Get an `s3fs.S3FileSystem` object. No authentication is performed on this step. Authentication will be
        performed according to the rules at https://s3fs.readthedocs.io/en/latest/#credentials when accessing the data.

        By default, the filesystem is only created once, the first time this function is called. To force it create a
        new one, set `refresh` to `True`.

        Parameters
        ----------
        refresh : bool
            If set to `True`, a new `s3fs.S3FileSystem` will be created even if this object has one already
        profile : str
            Manually specify the name of the AWS profile to be used when creating an S3FileSystem.
            Overrides the default profile grabbed silently by aiobotocore.

        Returns
        -------
        s3fs.S3FileSystem
            A filesystem object for interfacing with S3
        """
        if refresh or not hasattr(self, "_fs"):
            self._fs = s3fs.S3FileSystem(profile=profile)
            self.dm.info(
                "Initialized S3 filesystem. Credentials will be looked up according to rules at "
                "https://s3fs.readthedocs.io/en/latest/#credentials"
            )
        return self._fs

    @property
    def path(self) -> str:
        """
        Get the S3-protocol URL to the parent `DatasetManager`'s Zarr .

        Returns
        -------
        str
            A URL string starting with "s3://" followed by the path to the Zarr.
        """
        if self.dm.custom_output_path:
            return self.dm.custom_output_path
        else:
            return f"s3://{self.bucket}/datasets/{self.dm.key()}.zarr"

    def __str__(self) -> str:
        # TODO: Is anything relying on this? It's not super intuitive behavior. If this is for debugging in a REPL, it
        # is more common to implement __repr__ which generally returns a string that could be code to instantiate the
        # instance.
        return self.path

    def mapper(self, refresh: bool = False, **kwargs) -> fsspec.mapping.FSMap:
        """
        Get a `MutableMapping` representing the S3 key/value store. By default, the mapper will be created only once,
        when this function is first called.

        To force a new mapper, set `refresh` to `True`. To use an output path other than the default path returned by
        self.path, set a `custom_output_path` when the DatasetManager is instantiated and it will be passed through to
        here. This path must be a valid S3 destination for which you have write permissions.

        Parameters
        ----------
        refresh : bool
            Set to `True` to force a new mapper to be created even if this object has one already
        **kwargs : dict
            Arbitrary keyword args supported

        Returns
        -------
        s3fs.S3Map
            A `MutableMapping` which is the S3 key/value store
        """
        if refresh or not hasattr(self, "_mapper"):
            self._mapper = s3fs.S3Map(root=self.path, s3=self.fs())
        return self._mapper

    @property
    def has_existing(self) -> bool:
        """
        Returns
        -------
        bool
            Return `True` if there is a Zarr at `S3.path`
        """
        return self.fs().exists(self.path)

    def push_metadata(self, title: str, stac_content: dict, stac_type: str):
        """
        Publish metadata entity to s3 store. Tracks historical state
        of metadata as well

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_content : dict
            content of the stac entity
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)
        """
        metadata_path = self.get_metadata_path(title, stac_type)
        fs = self.fs()
        if fs.exists(metadata_path):
            # Generate history file
            old_mod_time = fs.ls(metadata_path, detail=True)[0]["LastModified"]
            history_file_name = f"{title}-{old_mod_time.isoformat(sep='T')}.json"
            history_path = f"s3://{self.bucket}/history/{title}/{history_file_name}"
            fs.copy(metadata_path, history_path)

        fs.write_text(metadata_path, json.dumps(stac_content))

    def retrieve_metadata(self, title: str, stac_type: str) -> tuple[dict, str]:
        """
        Retrieve metadata entity from s3 store

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        dict
            Tuple of content of stac entity as dict and the s3 path as a string
        """
        metadata_path = self.get_metadata_path(title, stac_type)
        return json.loads(self.fs().cat(metadata_path)), metadata_path

    def metadata_exists(self, title: str, stac_type: str) -> bool:
        """
        Check whether metadata exists at a given s3 path

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        bool
            Whether metadata exists at path
        """
        metadata_path = self.get_metadata_path(title, stac_type)
        return self.fs().exists(metadata_path)

    def get_metadata_path(self, title: str, stac_type: str) -> str:
        """
        Get the s3 path for a given STAC title and type

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        str
            The s3 path for this entity as a string
        """
        if stac_type:
            return f"s3://{self.bucket}/metadata/{stac_type}/{title}.json"
        else:
            return f"s3://{self.bucket}/metadata/{title}.json"

    def write_metadata_only_v2(self, update_attrs: dict[str, Any]):
        """
        Old method of writing metadata. Kept for backwards compatibility.
        """
        # Edit both .zmetadata and .zattrs
        fs = self.fs()

        for z_path in (".zmetadata", ".zattrs"):
            # Read current metadata from Zarr
            with fs.open(f"{self.path}/{z_path}") as z_contents:
                current_attributes = json.load(z_contents)

            # Update given attributes at the appropriate location depending on which z file
            if z_path == ".zmetadata":
                current_attributes["metadata"][".zattrs"].update(update_attrs)
            else:
                current_attributes.update(update_attrs)

            # Write back to Zarr
            with fs.open(f"{self.path}/{z_path}", "w") as z_contents:
                json.dump(current_attributes, z_contents)

    def write_metadata_only(self, update_attrs: dict[str, Any]):
        """
        Update metadata within the master zarr.json file contained within v3 Zarrs

        Parameters
        ----------
        update_attrs : dict[str, Any]
            A dictionary of attributes to update in the zarr.json file
        """
        fs = self.fs()

        # Read current metadata from Zarr
        with fs.open(f"{self.path}/zarr.json") as z_contents:
            current_attributes = json.load(z_contents)

        # Update given attributes
        current_attributes["attributes"].update(update_attrs)

        # Write back to Zarr
        with fs.open(f"{self.path}/zarr.json", "w") as z_contents:
            json.dump(current_attributes, z_contents)


class Local(StoreInterface):
    """
    Provides an interface for reading and writing a dataset's Zarr on the local filesystem.

    The path of the Zarr is assumed to be the return value of `Local.dm.output_path`. That is the path used
    automatically under normal conditions, although it can be overriden by passing the `custom_output_path` parameter
    to the relevant DatasetManager
    """

    def __init__(self, dm: dataset_manager.DatasetManager, folder: pathlib.Path | str = "."):
        """
        Parameters
        ----------
        dm : dataset_manager.DatasetManager
            The dataset to be read or written.
        folder: pathlib.Path | str
            The folder to write metadata into. Defaults to the current working directory.
        """
        self.dm = dm
        self.folder = folder

    def fs(self, refresh: bool = False) -> fsspec.implementations.local.LocalFileSystem:
        """
        Get an `fsspec.implementations.local.LocalFileSystem` object. By default, the filesystem is only created once,
        the first time this function is called. To force it create a new one, set `refresh` to `True`.

        Parameters
        ----------
        refresh : bool
            If set to `True`, a new `fsspec.implementations.local.LocalFileSystem` will be created even if this object
            has one already

        Returns
        -------
        fsspec.implementations.local.LocalFileSystem
            A filesystem object for interfacing with the local filesystem
        """
        if refresh or not hasattr(self, "_fs"):
            self._fs = fsspec.filesystem("file")
        return self._fs

    def mapper(self, refresh=False, **kwargs) -> fsspec.mapping.FSMap:
        """
        Get a `MutableMapping` representing a local filesystem key/value store.
        By default, the mapper will be created only once, when this function is first
        called. To force a new mapper, set `refresh` to `True`.

        Parameters
        ----------
        refresh : bool
            Set to `True` to force a new mapper to be created even if this object has one already.
        **kwargs : dict
            Arbitrary keyword args supported

        Returns
        -------
        fsspec.mapping.FSMap
            A `MutableMapping` which is a key/value representation of the local filesystem
        """
        if refresh or not hasattr(self, "_mapper"):
            self._mapper = self.fs().get_mapper(self.path)
        return self._mapper

    def __str__(self) -> str:
        # TODO: Is anything relying on this? It's not super intuitive behavior. If this is for debugging in a REPL, it
        # is more common to implement __repr__ which generally returns a string that could be code to instantiate the
        # instance.
        return str(self.path)

    @property
    def path(self) -> pathlib.Path:
        """
        Returns
        -------
        pathlib.Path
            Path to the Zarr on the local filesystem
        """
        if self.dm.custom_output_path:
            return self.dm.custom_output_path
        else:
            return self.dm.output_path().joinpath(f"{self.dm.dataset_name}.zarr")

    @property
    def has_existing(self) -> bool:
        """
        Returns
        -------
        bool
            Return `True` if there is a local Zarr for this dataset, `False` otherwise.
        """
        return self.path.exists()

    def push_metadata(self, title: str, stac_content: dict, stac_type: str):
        """
        Publish metadata entity to local store. Tracks historical state
        of metadata as well

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_content : dict
            content of the stac entity
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)
        """
        metadata_path = pathlib.Path(self.get_metadata_path(title, stac_type))
        if metadata_path.exists():
            # Generate history file
            old_mod_time = (
                datetime.datetime.fromtimestamp(metadata_path.stat().st_mtime)
                .astimezone(datetime.timezone.utc)
                .replace(tzinfo=None)
            )
            history_path = (
                pathlib.Path(self.folder) / "history" / title / f"{title}-{old_mod_time.isoformat(sep='T')}.json"
            )
            history_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(metadata_path, history_path)

        # Write new metadata to file (may overwrite)
        metadata_path.parent.mkdir(exist_ok=True, parents=True)
        with open(metadata_path, "w") as f:
            json.dump(stac_content, f)

    def retrieve_metadata(self, title: str, stac_type: str) -> tuple[dict, str]:
        """
        Retrieve metadata entity from local store

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        dict
            Tuple of content of stac entity as dict and the local path as a string
        """
        metadata_path = self.get_metadata_path(title, stac_type)
        with open(metadata_path) as f:
            return json.load(f), str(metadata_path)

    def metadata_exists(self, title: str, stac_type: str) -> bool:
        """
        Check whether metadata exists at a given local path

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        bool
            Whether metadata exists at path
        """
        metadata_path = self.get_metadata_path(title, stac_type)
        return pathlib.Path(metadata_path).exists()

    def get_metadata_path(self, title: str, stac_type: str) -> str:
        """
        Get the local path for a given STAC title and type

        Parameters
        ----------
        title : str
            STAC Entity title
        stac_type : StacType
            Path part corresponding to type of STAC entity
            (empty string for Catalog, 'collections' for Collection or 'datasets' for Item)

        Returns
        -------
        str
            The s3 path for this entity
        """
        return str((pathlib.Path(self.folder) / "metadata" / stac_type / f"{title}.json").resolve())

    def write_metadata_only(self, update_attrs: dict[str, Any]):
        # Edit both .zmetadata and .zattrs
        for z_path in (".zmetadata", ".zattrs"):
            # Read current metadata from Zarr
            with open(self.path / z_path) as z_contents:
                current_attributes = json.load(z_contents)

            # Update given attributes at the appropriate location depending on which z file
            if z_path == ".zmetadata":
                current_attributes["metadata"][".zattrs"].update(update_attrs)
            else:
                current_attributes.update(update_attrs)

            # Write back to Zarr
            with open(self.path / z_path, "w") as z_contents:
                json.dump(current_attributes, z_contents)
