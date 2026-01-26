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
import platform
import pathlib

import psutil

from .utils.encryption import register_encryption_key
from .utils.logging import Logging
from .utils.publish import Publish
from .utils.time import TimeSpan
from .utils.store import Local, S3


class DatasetManager(Logging, Publish, ABC):
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

    # Time span constants for backward compatibility
    SPAN_HALF_HOURLY = TimeSpan.SPAN_HALF_HOURLY
    SPAN_HOURLY = TimeSpan.SPAN_HOURLY
    SPAN_THREE_HOURLY = TimeSpan.SPAN_THREE_HOURLY
    SPAN_SIX_HOURLY = TimeSpan.SPAN_SIX_HOURLY
    SPAN_DAILY = TimeSpan.SPAN_DAILY
    SPAN_WEEKLY = TimeSpan.SPAN_WEEKLY
    SPAN_MONTHLY = TimeSpan.SPAN_MONTHLY
    SPAN_YEARLY = TimeSpan.SPAN_YEARLY
    SPAN_SEASONAL = TimeSpan.SPAN_SEASONAL
    DATE_FORMAT_FOLDER = "%Y%m%d"
    DATE_HOURLY_FORMAT_FOLDER = "%Y%m%d%H"
    DATE_FORMAT_METADATA = "%Y/%m/%d"

    @classmethod
    def from_time_span_string(cls, span_str: str) -> TimeSpan:
        """Convert a string representation of a time span to the corresponding TimeSpan enum.

        Parameters
        ----------
        span_str : str
            String representation of the time span (e.g., "hourly", "daily")

        Returns
        -------
        TimeSpan
            The corresponding TimeSpan enum member

        Raises
        ------
        ValueError
            If the string does not correspond to a valid time span
        """
        return TimeSpan.from_string(span_str.lower())

    def __init__(
        self,
        requested_dask_chunks,
        requested_zarr_chunks,
        rebuild_requested=False,
        custom_output_path: str | pathlib.Path | None = None,
        custom_input_path: str | pathlib.Path | None = None,
        console_log=True,
        global_log_level=logging.DEBUG,
        store=None,
        s3_bucket_name=None,
        allow_overwrite=False,
        dask_dashboard_address: str = "127.0.0.1:8787",
        dask_worker_memory_target: float = 0.65,
        dask_worker_memory_spill: float = 0.65,
        dask_num_workers: int | None = None,
        dask_num_threads: int | None = None,
        dask_cpu_mem_target_ratio: float = 4 / 32,
        dask_scheduler_protocol: str = "inproc://",
        use_local_zarr_jsons: bool = False,
        skip_prepare_input_files: bool = False,
        skip_pre_parse_nan_check: bool = False,
        skip_post_parse_qc: bool = False,
        skip_post_parse_api_check: bool = False,
        encryption_key: str | None = None,
        use_compression: bool = True,
        dry_run: bool = False,
        output_zarr3: bool = False,
        align_update_chunks: bool = False,
        *args,
        **kwargs,
    ):
        """
        Set member variables to defaults. Set up logging to console and any other requested logs.

        Parameters
        ----------
        rebuild_requested : bool, optional
            Sets `DatasetManager.rebuild_requested`. If this parameter is set, the manager requests and parses all
            available data from beginning to end.
        custom_output_path : str | pathlib.Path, optional
            Overrides the default path returned by `StoreInterface.path` for Local and S3 stores.
        custom_input_path : str | pathlib.Path, optional
            A path to use for input files
        console_log : bool, optional
            Enable logging `logging.INFO` level and higher statements to console. For more customization, see
            `DatasetManager.log_to_console`
        global_log_level : str, optional
            The root logger `logger.getLogger()` will be set to this level. Recommended to be `logging.DEBUG`, so all
            logging statements will be generated and then logging handlers can decide what to do with them.
        store : str | None
            A string indicating the type of filestore to use (one of, "local" or "s3"). A corresponding store
            object will be initialized. If `None`, the store is left unset and the default store interface defined in
            `Attributes.store` (local) is returned when the property is accessed. If using S3, the environment
            variables `AWS_ACCESS_KEY_ID`and `AWS_SECRET_ACCESS_KEY` must be specified in the ~/.aws/credentials file
            or set manually.
        s3_bucket_name : str
            Name of the S3 bucket where this dataset's Zarrs are stored. Only used if "s3" store is used.
        allow_overwrite : bool
            Unless this is set to `True`, inserting or overwriting data for dates before the dataset's current end date
            will fail with a warning message.
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
        output_zarr3: bool
            Although the required Zarr library version is 3.0+, Zarrs are still written in the 2.0 format by default,
            for backward compatibility. However, if this parameter is set, the Zarr will be written in 3.0 format.
        align_update_chunks: bool
            When updating an existing Zarr, align incoming time chunks with existing time chunks. If these chunks don't
            align, the ETL may fail because there will be a danger of corrupting data. However, this is disabled by
            default for backward compatibility with ETLs that don't need this behavior. See Aligning_update_chunks.md
            for details.
        """
        super().__init__()

        self.rebuild_requested = rebuild_requested

        # These paths are expected to be pathlib.Path objects in some cases, so make sure they are not str type. The
        # one-line if/else makes this typing work with mypy.
        self.custom_output_path: pathlib.Path | None = (
            custom_output_path if custom_output_path is None else pathlib.Path(custom_output_path)
        )
        self.custom_input_path: pathlib.Path | None = (
            custom_input_path if custom_input_path is None else pathlib.Path(custom_input_path)
        )

        # Create certain paramters for development and debugging of certain dataset. All default to False.
        self.dry_run = dry_run
        self.use_local_zarr_jsons = use_local_zarr_jsons
        self.skip_prepare_input_files = skip_prepare_input_files
        self.skip_pre_parse_nan_check = skip_pre_parse_nan_check
        self.skip_post_parse_qc = skip_post_parse_qc
        self.skip_post_parse_api_check = skip_post_parse_api_check

        # Create a store object based on the passed store string. If `None`, treat as "local". If any string other than
        # "local" or "s3" is passed, raise a `ValueError`.
        if store is None or store == "local":
            self.store = Local(self)
        elif store == "s3":
            self.store = S3(self, s3_bucket_name)
        else:
            raise ValueError("Store must be one of 'local' or 's3'")

        # Assign the allow overwrite flag. The value should always be either `True` or `False`.
        self.allow_overwrite = allow_overwrite

        # Initialize logging and write system info to the log
        self.init_logging(console_log=console_log, global_log_level=global_log_level)

        # set chunk sizes (usually specified in the ETL manager class init)
        self.requested_dask_chunks = requested_dask_chunks
        self.requested_zarr_chunks = requested_zarr_chunks

        # set the dask dashboard address. Defaults to 127.0.0.1:8787 so it's only findable on the local machine
        self.dask_dashboard_address = dask_dashboard_address

        # Setup dask workers and threads
        total_memory_gb = psutil.virtual_memory().total / 1_000_000_000  # needed for info statement below
        # Default path is to set dask_num_workers and dask_num_threads based on the number of cores and memory available
        if dask_num_workers is None and dask_num_threads is None:
            # Usually set to 1 to avoid data transfer between workers
            self.dask_num_workers = 1

            # All threads will be on a single CPU because self.dask_num_workers is 1.
            # By default, we use 4 threads per 32 GB RAM, adjust in the init of your manager
            # if you desire a different ratio. If there are not enough cores
            # available to use the target number of threads, use all available cores as threads.
            target_thread_count = int(dask_cpu_mem_target_ratio * total_memory_gb)
            if target_thread_count >= multiprocessing.cpu_count():
                target_thread_count = multiprocessing.cpu_count() - 1

            if target_thread_count < 1:
                target_thread_count = 1

            self.dask_num_threads = target_thread_count
        # Otherwise, use the values provided. You must provide both!
        elif dask_num_workers is not None and dask_num_threads is not None:
            self.dask_num_workers = dask_num_workers
            self.dask_num_threads = dask_num_threads
        else:
            raise ValueError(
                "Either set both dask_num_workers and dask_num_threads or neither; "
                f"got dask_num_workers={dask_num_workers} and dask_num_threads={dask_num_threads}"
            )

        # Other Dask distributed configuration defaults, mostly related to memory usage
        self.dask_scheduler_worker_saturation = 1.2
        self.dask_worker_mem_target = dask_worker_memory_target
        self.dask_worker_mem_spill = dask_worker_memory_spill
        self.dask_worker_mem_pause = 0.92
        self.dask_worker_mem_terminate = 0.98
        self.dask_use_process_scheduler = self.dask_num_workers > 1
        self.dask_scheduler_protocol = dask_scheduler_protocol

        self.info(
            f"Using {self.dask_num_threads} threads on a {multiprocessing.cpu_count()}-core system with "
            f"{total_memory_gb:.2f}GB RAM"
        )

        self.encryption_key = register_encryption_key(encryption_key) if encryption_key else None
        self.use_compression = use_compression
        self.output_zarr3 = output_zarr3

        # Check output Zarr format versus existing format before moving on with this object
        if self.store.has_existing and not self.rebuild_requested:
            if self.store.has_v3_metadata:
                if not self.output_zarr3:
                    raise RuntimeError("Existing data is Zarr v3, but output_zarr3 is not set.")
            elif self.output_zarr3:
                raise RuntimeError("Existing data is not Zarr v3, but output_zarr3 is set.")

        # Control behavior when appending or inserting data to an existing Zarr
        self.align_update_chunks = align_update_chunks

    # SETUP

    def init_logging(self, console_log: bool = True, global_log_level: int = logging.DEBUG):
        """
        Configure the Python logging module according to the given parameters, and write some system information to the
        log.

        Parameters
        ----------
        console_log : bool, optional
            Enable logging `logging.INFO` level and higher statements to console. For more customization, see
            `DatasetManager.log_to_console`
        global_log_level : str, optional
            The root logger `logger.getLogger()` will be set to this level. Recommended to be `logging.DEBUG`, so all
            logging statements will be generated and then logging handlers can decide what to do with them.
        """
        # Print log statements to console by default
        if console_log:
            self.log_to_console()

        # Set the logging level of logger.getLogger(), which is the logging module's root logger and will control the
        # level of log statements that are enabled globally. If this is set to `logging.DEBUG`, all log statements will
        # be enabled by default and will be forwarded to handlers set by either `logging.Logger.addHandler`,
        # `DatasetManager.log_to_file`, or `DatasetManager.log_to_console`.
        logging.getLogger().setLevel(global_log_level)

        # hide DEBUG spam from h5-to-zarr during kerchunking
        logging.getLogger("h5-to-zarr").setLevel(logging.WARNING)

        # Add a custom exception handler that will print the traceback to loggers
        sys.excepthook = self.log_except_hook

        # Log key system information
        self.info(platform.platform())
        self.info(f"Python {platform.python_version()}")
        try:
            # None of the following keys are guaranteed to be included in the platform information, so build a string
            # with as much information as is available.
            release = platform.freedesktop_os_release()
            log = ""
            if "PRETTY_NAME" in release:
                log += release["PRETTY_NAME"]
            else:
                if "NAME" in release:
                    log += release["NAME"] + " "
                if "VERSION" in release:
                    log += release["VERSION"]
            if log:
                self.info(log)
        except OSError:
            # OK to pass because the platform may not be Linux, in which case, just platform.platform() will print
            pass

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

    # Attributes

    @property
    def initial_metadata(self):
        """
        Property returning a dictionary of metadata fields for this ETL.

        This is inherited from the Metadata mixin and can be extended by subclasses.
        See :py:meth:`gridded_etl_tools.utils.metadata.Metadata.initial_metadata` for details.
        """
        return super().initial_metadata

    @property
    @abstractmethod
    def dataset_start_date(self):
        """First date in dataset. Used to populate corresponding encoding and metadata."""

    # Extraction

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

    # Transformation

    def transform(self) -> xr.Dataset:
        """
        Master convenience function encapsulating all transform steps, on-disk and in-memory. These can optionally
        be broken out and run separately if more useful within an ETL.

        Returns
        -------
        xr.Dataset
            A finalized in-memory dataset ready for rechunking and publication
        """
        self.transform_data_on_disk()
        return self.transform_dataset_in_memory()

    def transform_data_on_disk(self):
        """
        Open all raw files in self.local_input_path(). Transform the data contained in them into a virtual Zarr JSON
        conforming to Arbol's standard format for gridded datasets

        This is the core function for transforming data from a provider's raw files and should be standard for all ETLs
        Modify the child methods it calls to tailor the methods to the individual dataset and resolve any issues
        """
        self.info("Transforming raw files to an in-memory dataset")
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
            publish_dataset = self._update_ds_transform()
        elif not self.store.has_existing or (self.rebuild_requested and self.allow_overwrite):
            publish_dataset = self._initial_ds_transform()
        else:
            raise RuntimeError(
                "There is already a zarr at the specified path and a rebuild is requested, "
                "but overwrites are not allowed. Therefore the appropriate transformation, "
                "not to mention the appropriate eventual write operation, is unclear."
            )
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

    def set_zarr_metadata(self, dataset) -> xr.Dataset:  # pragma NO COVER
        """
        Placeholder indicating necessity of possibly editing Zarr metadata within an ETL manager script
        Method to align Zarr metadata with requirements of Zarr exports and STAC metadata format
        Happens immediately before data publication.
        """
        return super().set_zarr_metadata(dataset)

    # Orchestration

    @classmethod
    def get_subclasses(cls) -> typing.Iterator:
        """Create a generator with all the subclasses and sub-subclasses of a parent class"""
        for subclass in cls.__subclasses__():
            yield subclass
            yield from subclass.get_subclasses()

    @classmethod
    def get_subclass(cls, name: str, time_resolution: str | None = None) -> type | None:
        """
        Method to return the subclass instance corresponding to the name provided when invoking the ETL

        Parameters
        ----------
        name : str
            The str returned by the name() property of the dataset to be parsed. Used to return that subclass's
            manager. For example, 'chirps_final_05' will yield CHIRPSFinal05 if invoked for the CHIRPS manager
        time_resolution : str, optional
            The time resolution of the dataset to be parsed.
            If provided, only subclasses with the same time resolution will be returned.
            This helps when there are multiple subclasses with the same name but different time resolutions.

        Returns
        -------
        type
            A dataset source class
        """

        for subclass in cls.get_subclasses():
            # Weed out abstract classes
            if ABC not in subclass.__bases__:
                # # Find the matching class
                if subclass.dataset_name == name:
                    # Use time resolution, if provided, to differentiate between
                    # otherwise identical classes with different time resolutions
                    if time_resolution and subclass.time_resolution != TimeSpan.from_string(time_resolution):
                        continue
                    return subclass

        warnings.warn(f"failed to set manager from name {name}, could not find corresponding class")
        return None
