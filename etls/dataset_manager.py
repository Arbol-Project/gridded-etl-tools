#
##### DatasetManager.py
# Abstract base classes defining managers for Arbol's climate data sets that can be represented as N-Dimensional data using Xarray and Zarr.
# To be inherited by classes that will implement set-specific update, parse, and/or verification methods

import sys, logging, pathlib, multiprocessing, multiprocessing.pool

from etls.utils.logging import Logging
from etls.utils.zarr_methods import Publish
from etls.utils.ipfs import IPFS

from abc import abstractmethod, ABC

class DatasetManager(Logging, Publish, IPFS, ABC):
    """
    This is a base class for data parsers. It is intended to be inherited and implemented by child classes specific to
    each data source. 

    It is the base class for any climate data set published in a format that is compatible with being opened in `xarray` and 
    transformed into a Zarr. Usable formats so far include netCDF and GRIB2.
    Sets in this category include CHIRPS, CPC, ERA5, VHI, and RTMA.
    
    For example, for data sourced from CHIRPS, there is a CHIRPS general class that implements most of CHIRPS parsing,
    and further inheriting that class is a fully implemented CHIRPS05 class which updates, parses, and verifies CHIRPS .05 data
    """

    SPAN_HOURLY = "hourly"
    SPAN_DAILY = "daily"
    SPAN_WEEKLY = "weekly"
    SPAN_MONTHLY = "monthly"
    SPAN_YEARLY = "yearly"
    SPAN_SEASONAL = "seasonal"
    DATE_FORMAT_FOLDER = "%Y%m%d"
    DATE_HOURLY_FORMAT_FOLDER = "%Y%m%d%H"
    DATE_FORMAT_METADATA = "%Y/%m/%d"
    GATEWAY_IPFS_ID = "/ip4/159.89.40.173/tcp/4001/p2p/12D3KooWChQtvhtqu3KZdPYM4tEG15xjZbT7zcMjQBCkJv4Khanw"

    # paths relative to the script directory
    SOURCE_FILE_PATH = pathlib.Path(__file__).parent.resolve()
    LOCAL_INPUT_ROOT = SOURCE_FILE_PATH.parent / "datasets"
    OUTPUT_ROOT = SOURCE_FILE_PATH.parent / "climate"

    def __init__(self, 
                 requested_dask_chunks, requested_zarr_chunks, requested_ipfs_chunker,
                 rebuild_requested=False, 
                 custom_output_path=None, custom_metadata_head_path=None, custom_latest_hash=None, custom_input_path=None, 
                 console_log=True, global_log_level=logging.DEBUG, 
                 *args, **kwargs):
                 
        """
        Set member variables to defaults. Setup logging to console and any other requested logs. Set the libeccodes lookup path.

        Parameters
        ----------
        rebuild_requested : bool, optional
            A switch to rebuild_requested the dataset from scratch
        custom_output_path : str, optional
            A path to use for outputting finalized data (normally in the `climate` folder)
        custom_metadata_head_path : str, optional
            A CID to use for retrieving metadata
        custom_latest_hash : str, optional
            A hash to use for operations using the latest hash
        custom_input_path : str, optional
            A path to use for input files
        http_root : str, optional
            A URL to use for interacting with the API
        console_log : bool, optional
            Enable logging `logging.INFO` level and higher statements to console. For more customization, see
            `DatasetManager.log_to_console`
        global_log_level : str, optional
            The root logger `logger.getLogger()` will be set to this level. Recommended to be logging.DEBUG, so all logging
            statements will be generated and then logging handlers can decide what to do with them.

        """
        # call IPFS init
        super().__init__()
        # Set member variable defaults
        self.new_files = []
        self.custom_output_path = custom_output_path
        self.custom_metadata_head_path = custom_metadata_head_path
        self.custom_latest_hash = custom_latest_hash
        self.custom_input_path = custom_input_path
        self.rebuild_requested = rebuild_requested

        # Print log statements to console by default
        if console_log:
            self.log_to_console()

        # Set the logging level of logger.getLogger(), which is the logging module's root logger and will control the level of log statements
        # that are enabled globally. If this is set to `logging.DEBUG`, all log statements will be enabled by default and will be forwarded to
        # handlers set by either `logging.Logger.addHandler`, `DatasetManager.log_to_file`, or `DatasetManager.log_to_console`.
        logging.getLogger().setLevel(global_log_level)

        # Add a custom exception handler that will print the traceback to loggers
        sys.excepthook = self.log_except_hook

        # set chunk sizes based on specifications in the ETL manager
        self.requested_dask_chunks = requested_dask_chunks
        self.requested_zarr_chunks = requested_zarr_chunks
        self.requested_ipfs_chunker = requested_ipfs_chunker

        # Dask distributed configuration defaults, mostly related to memory usage
        self.dask_scheduler_worker_saturation = 1.2
        self.dask_worker_mem_target = 0.65
        self.dask_worker_mem_spill = 0.65
        self.dask_worker_mem_pause = 0.92
        self.dask_worker_mem_terminate = 0.98

        # Each thread will use a CPU if self.dask_num_workers is 1. Setting it to use 75% of available CPUs seems to be reasonable.
        self.dask_num_threads = max(1, int(multiprocessing.cpu_count() * 0.75))

        # Usually set to 1 to avoid data transfer between workers
        self.dask_num_workers = 1

    # SETUP

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            The name of the dataset
        """
        return self.name()


    def __eq__(self, other):
        """
        All instances of this class will compare equal to each other
        """
        return str(self) == other


    def __hash__(self):
        return hash(str(self))


    # MINIMUM ETL METHODS

    @abstractmethod
    def static_metadata(self):
        """
        Placeholder indicating necessity of instantiating static metadata at the top of an ETL manager script
        """
        ...


    @abstractmethod
    def update_local_input(self):
        """
        Check for updates to local input files (usually by checking a remote location where climate data publishers post updated
        data)
        """ 
        self.new_files = []


    @abstractmethod
    def prepare_input_files(self, keep_originals=True):
        """
        Convert each of the input files (and associated metadata files) to a collection of daily netCDF4 classic files suitable for 
        reading by Kerchunk and intake into Xarray. This allows us to stack data into modern, performant N-Dimensional Zarr data.
        """
        pass


    def populate_metadata(self):
        """
        Fill the metadata with values describing this set, using the static_metadata as a base template.
        """
        if hasattr(self, "metadata") and self.metadata is not None:
            self.metadata = self.metadata.update(self.static_metadata)
        else:
            self.metadata = self.static_metadata


    def set_zarr_metadata(self, dataset):
        """
        Placeholder indicating necessity of possibly editing Zarr metadata within an ETL manager script
        Method to align Zarr metadata with requirements of Zarr exports and STAC metadata format
        Happens after `populate_metadata` and immediately before data publication.
        """
        dataset = super().set_zarr_metadata(dataset)
        return dataset


