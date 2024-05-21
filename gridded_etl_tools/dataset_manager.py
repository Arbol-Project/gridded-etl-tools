# This is necessary for referencing types that aren't fully imported yet. See https://peps.python.org/pep-0563/
from __future__ import annotations

from abc import abstractmethod, ABC
import datetime
import typing
import warnings
import deprecation
import logging
import multiprocessing
import multiprocessing.pool
import sys
import xarray as xr

import psutil

from .utils.encryption import register_encryption_key
from .utils.logging import Logging
from .utils.publish import Publish
from .utils.ipfs import IPFS
from .utils.store import Local, IPLD, S3


class DatasetManager(Logging, Publish, ABC, IPFS):
    """
    This is a base class for data parsers. It is intended to be inherited and implemented by child classes specific to
    each data source.

    It is the base class for any climate data set published in a format that is compatible with being opened in
    `xarray` and transformed into a Zarr. Usable formats so far include netCDF and GRIB2. Sets in this category include
    CHIRPS,CPC,ERA5,VHI,and RTMA.

    For example,for data sourced from CHIRPS,there is a CHIRPS general class that implements most of CHIRPS parsing,
    and further inheriting that class is a fully implemented CHIRPS05 class which updates,parses,and verifies CHIRPS
    .05 data
    """

    SPAN_HOURLY = "hourly"
    SPAN_SIX_HOURLY = "6hourly"
    SPAN_DAILY = "daily"
    SPAN_WEEKLY = "weekly"
    SPAN_MONTHLY = "monthly"
    SPAN_YEARLY = "yearly"
    SPAN_SEASONAL = "seasonal"
    DATE_FORMAT_FOLDER = "%Y%m%d"
    DATE_HOURLY_FORMAT_FOLDER = "%Y%m%d%H"
    DATE_FORMAT_METADATA = "%Y/%m/%d"

    def __init__(
        self,
        requested_dask_chunks,
        requested_zarr_chunks,
        requested_ipfs_chunker=None,
        rebuild_requested=False,
        custom_output_path=None,
        custom_latest_hash=None,
        custom_input_path=None,
        console_log=True,
        global_log_level=logging.DEBUG,
        store=None,
        s3_bucket_name=None,
        allow_overwrite=False,
        ipfs_host="http://127.0.0.1:5001",
        dask_dashboard_address: str = "127.0.0.1:8787",
        dask_worker_memory_target: float = 0.65,
        dask_worker_memory_spill: float = 0.65,
        dask_cpu_mem_target_ratio: float = 4 / 32,
        use_local_zarr_jsons: bool = False,
        skip_prepare_input_files: bool = False,
        skip_post_parse_qc: bool = False,
        skip_post_parse_api_check: bool = False,
        encryption_key: str = None,
        use_compression: bool = True,
        dry_run: bool = False,
        *args,
        **kwargs,
    ):
        """
        Set member variables to defaults. Set up logging to console and any other requested logs.

        Parameters
        ----------
        ipfs_host : str, optional
            The address of the IPFS HTTP API to use for IPFS operations
        rebuild_requested : bool, optional
            Sets `DatasetManager.rebuild_requested`. If this parameter is set, the manager requests and parses all
            available data from beginning to end.
        custom_output_path : str, optional
            Overrides the default path returned by `StoreInterface.path` for Local and S3 stores.
        custom_latest_hash : str, optional
            Overrides the default hash lookup defined in `IPFS.latest_hash`
        custom_input_path : str, optional
            A path to use for input files
        console_log : bool, optional
            Enable logging `logging.INFO` level and higher statements to console. For more customization, see
            `DatasetManager.log_to_console`
        global_log_level : str, optional
            The root logger `logger.getLogger()` will be set to this level. Recommended to be `logging.DEBUG`, so all
            logging statements will be generated and then logging handlers can decide what to do with them.
        store : str | None
            A string indicating the type of filestore to use (one of, "local", "ipld" or "s3"). A corresponding store
            object will be initialized. If `None`, the store is left unset and the default store interface defined in
            `Attributes.store` (local) is returned when the property is accessed. If using S3, the environment
            variables `AWS_ACCESS_KEY_ID`and `AWS_SECRET_ACCESS_KEY` must be specified in the ~/.aws/credentials file
            or set manually.
        s3_bucket_name : str
            Name of the S3 bucket where this dataset's Zarrs are stored. Only used if "s3" store is used.
        allow_overwrite : bool
            Unless this is set to `True`, inserting or overwriting data for dates before the dataset's current end date
            will fail with a warning message.
        ipfs_host : str
            The URL of the IPFS host
        dask_dashboard_address : str
            The desired URL of the dask dashboard
        dask_worker_memory_target : float
            The desired maximum occupancy of available memory by Dask, expressed as a ratio of one
        dask_worker_memory_spill : float
            The level beyond which Dask will consider spilling additional objects in memory to disk, expressed as a
            ratio of one
        dask_cpu_mem_target_ratio : float
            The desired fraction of the total available memory assigned to each of Dask's workers
            Unless this is set to `True`, inserting or overwriting data for dates before the dataset's current end date
            will fail with a warning message.
        use_local_zarr_jsons: bool, optional
            Write out Zarr JSONs created via Kerchunk to the local file system. For use with remotely kerchunked
            datasets. Defaults to False.
        skip_prepare_input_files: bool, optional
            Skip the `prepare_input_files` method. Useful when restarting a parse that previously prepared input files
        skip_post_parse_qc: bool, optional
            Skip the `post_parse_quality_check` method. Applicable to datasets
            that transform source data before parsing, making source data checks irrelevant.
        skip_post_parse_api_check: bool, optional
            Skip API checks run by the orchestration stack.
            Applicable to datasets that are not set up for API access.
        encryption_key : str, optional
            If provided, data will be encrypted using `encryption_key` with XChaCha20Poly1305. Use
            :func:`.encryption.generate_encryption_key` to generate a random encryption key to be passed in here.
        use_compression: bool, optional
            Data in this dataset will be compressed unless this is set to `False`.
        dry_run: bool, optional
            Run the dataset manager all the way through but never write anything via `to_zarr`.
            Intended for development purposes
        """
        # call IPFS init
        super().__init__(host=ipfs_host)
        # Set member variable defaults
        self.new_files = []
        self.custom_output_path = custom_output_path
        self.custom_latest_hash = custom_latest_hash
        self.custom_input_path = custom_input_path
        self.rebuild_requested = rebuild_requested

        # Create certain paramters for development and debugging of certain dataset. All default to False.
        self.dry_run = dry_run
        self.use_local_zarr_jsons = use_local_zarr_jsons
        self.skip_prepare_input_files = skip_prepare_input_files
        self.skip_post_parse_qc = skip_post_parse_qc
        self.skip_post_parse_api_check = skip_post_parse_api_check

        # Create a store object based on the passed store string. If `None`, treat as "local". If any string other than
        # "local", "ipld", or "s3" is passed, raise a `ValueError`.
        if store is None or store == "local":
            self.store = Local(self)
        elif store == "ipld":
            self.store = IPLD(self)
        elif store == "s3":
            self.store = S3(self, s3_bucket_name)
        else:
            raise ValueError("Store must be one of 'local', 'ipld', or 's3'")

        # Assign the allow overwrite flag. The value should always be either `True` or `False`.
        # Always allow overwrites if IPLD for backwards compatibility
        self.allow_overwrite = allow_overwrite or isinstance(self.store, IPLD)

        # Print log statements to console by default
        if console_log:
            self.log_to_console()

        # Set the logging level of logger.getLogger(), which is the logging module's root logger and will control the
        # level of log statements that are enabled globally. If this is set to `logging.DEBUG`, all log statements will
        # be enabled by default and will be forwarded to handlers set by either `logging.Logger.addHandler`,
        # `DatasetManager.log_to_file`, or `DatasetManager.log_to_console`.
        logging.getLogger().setLevel(global_log_level)

        # Add a custom exception handler that will print the traceback to loggers
        sys.excepthook = self.log_except_hook

        # set chunk sizes (usually specified in the ETL manager class init)
        self.requested_dask_chunks = requested_dask_chunks
        self.requested_zarr_chunks = requested_zarr_chunks
        self.requested_ipfs_chunker = requested_ipfs_chunker

        # set the dask dashboard address. Defaults to 127.0.0.1:8787 so it's only findable on the local machine
        self.dask_dashboard_address = dask_dashboard_address

        # Dask distributed configuration defaults, mostly related to memory usage
        self.dask_scheduler_worker_saturation = 1.2
        self.dask_worker_mem_target = dask_worker_memory_target
        self.dask_worker_mem_spill = dask_worker_memory_spill
        self.dask_worker_mem_pause = 0.92
        self.dask_worker_mem_terminate = 0.98
        self.dask_use_process_scheduler = False
        self.dask_scheduler_protocol = "inproc://"

        # Usually set to 1 to avoid data transfer between workers
        self.dask_num_workers = 1

        # Each thread will use a CPU if self.dask_num_workers is 1. The default target ratio is 4 threads per 32 GB
        # RAM, adjust in the init of your manager if you desire a diffeerent ratio. If there are not enough cores
        # available to use the target number of threads, use the number of available cores.
        total_memory_gb = psutil.virtual_memory().total / 1_000_000_000
        target_thread_count = int(dask_cpu_mem_target_ratio * total_memory_gb)
        if target_thread_count >= multiprocessing.cpu_count():
            target_thread_count = multiprocessing.cpu_count() - 1

        if target_thread_count < 1:
            target_thread_count = 1

        self.dask_num_threads = target_thread_count

        self.info(
            f"Using {self.dask_num_threads} threads on a {multiprocessing.cpu_count()}-core system with "
            f"{total_memory_gb:.2f}GB RAM"
        )

        self.encryption_key = register_encryption_key(encryption_key) if encryption_key else None
        self.use_compression = use_compression

    # SETUP

    @deprecation.deprecated(details="Use Dataset's name attribute directly.")
    def __str__(self) -> str:
        """
        Returns
        -------
        str
            The name of the dataset
        """
        return self.dataset_name

    @deprecation.deprecated(details="Compare Dataset types directly")
    def __eq__(self, other: DatasetManager) -> bool:
        """
        All instances of this class will compare equal to each other.

        Returns
        -------
        bool
            If the other `DatasetManager` instance has the same name, return `True`
        """
        if isinstance(other, type(self)):
            return self.dataset_name == other.dataset_name

        return False

    @deprecation.deprecated(details="Hash Dataset's name attribute directly.")
    def __hash__(self):
        return hash(str(self))

    # MINIMUM ETL METHODS

    @abstractmethod
    def static_metadata(self):
        """
        Placeholder indicating necessity of instantiating static metadata at the top of an ETL manager script
        """

    @property
    @abstractmethod
    def dataset_start_date(self):
        """First date in dataset. Used to populate corresponding encoding and metadata."""

    @abstractmethod
    def extract(self, date_range: tuple[datetime.datetime, datetime.datetime] | None = None):
        """
        Check for updates to local input files (usually by checking a remote location where climate data publishers
        post updated data). Highly customized for every ETL.
        """
        if date_range and date_range[0] < self.dataset_start_date:
            raise ValueError(
                f"First datetime requested {date_range[0]} is before the start of the dataset in question. Please "
                "request a valid datetime."
            )
        self.new_files = []

    def transform_data_on_disk(self):
        """
        Open all raw files in self.local_input_path(). Transform the data contained in them into a virtual Zarr JSON
        conforming to Arbol's standard format for gridded datasets

        This is the core function for transforming data from a provider's raw files and should be standard for all ETLs
        Modify the child methods it calls to tailor the methods to the individual dataset and resolve any issues
        """
        self.info("Transforming raw files to an in-memory dataset")
        # Dynamically adjust metadata based on fields calculated during `extract`, if necessary (usually not)
        self.populate_metadata()
        # Create 1 file per measurement span (hour, day, week, etc.) so Kerchunk has consistently chunked inputs for
        # MultiZarring
        if not self.skip_prepare_input_files:  # in some circumstances it may be useful to skip file prep
            self.prepare_input_files()
        # Create Zarr JSON outside of Dask client so multiprocessing can use all workers / threads without interference
        # from Dask
        self.create_zarr_json()

    def transform_dataset_in_memory(self) -> xr.Dataset:
        """
        Get an `xr.Dataset` that can be passed to the appropriate writing method when writing a new Zarr. Read the
        virtual Zarr JSON at the path returned by `Creation.zarr_json_path` and *lazily* normalize the axes,
        re-chunk the dataset according to this object's chunking parameters,
        and add custom metadata defined by this class.

        This is the core function for transforming a dataset in-memory and should be standard for all ETLs.
        Modify the child methods it calls to tailor the methods to the individual dataset and resolve any issues

        NOTE that rechunking must be done within the Dask cluster to be optimized for a task graph,
        so it's located within the parse step

        Returns
        -------
        publish_dataset : xr.Dataset
            A finalized in-memory dataset ready for rechunking and publication
        """
        # Load the single dataset and perform any necessary transformations of it using Xarray
        self.info("Transforming in-memory dataset to its final format")
        if self.store.has_existing and not self.rebuild_requested:
            publish_dataset = self.update_ds_transform()
        elif not self.store.has_existing or (self.rebuild_requested and self.allow_overwrite):
            publish_dataset = self.initial_ds_transform()
        return publish_dataset

    @abstractmethod
    def prepare_input_files(self, keep_originals: bool = True):
        """
        Convert each of the input files (and associated metadata files) to a collection of daily netCDF4 classic files
        suitable for reading by Kerchunk and intake into Xarray. This allows us to stack data into modern, performant
        N-Dimensional Zarr data.

        Parameters
        ----------

        keep_originals : bool, optional
            An optional flag to preserve the original files for debugging purposes. Defaults to True.
        """

    def populate_metadata(self):  # pragma NO COVER
        """Override point for managers to populate metadata.

        The default implementation simply uses ``self.static_metadata``.
        """
        super().populate_metadata()

    def set_zarr_metadata(self, dataset) -> xr.Dataset:  # pragma NO COVER
        """
        Placeholder indicating necessity of possibly editing Zarr metadata within an ETL manager script
        Method to align Zarr metadata with requirements of Zarr exports and STAC metadata format
        Happens after `populate_metadata` and immediately before data publication.
        """
        return super().set_zarr_metadata(dataset)

    @classmethod
    def get_subclasses(cls) -> typing.Iterator:
        """Create a generator with all the subclasses and sub-subclasses of a parent class"""
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    @classmethod
    def get_subclass(cls, name: str) -> type:
        """
        Method to return the subclass instance corresponding to the name provided when invoking the ETL

        Parameters
        ----------
        name : str
            The str returned by the name() property of the dataset to be parsed. Used to return that subclass's
            manager. For example, 'chirps_final_05' will yield CHIRPSFinal05 if invoked for the CHIRPS manager

        Returns
        -------
        type
            A dataset source class
        """
        for source in cls.get_subclasses():
            if source.dataset_name == name:
                return source

        warnings.warn(f"failed to set manager from name {name}, could not find corresponding class")
