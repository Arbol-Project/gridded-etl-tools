import numpy as np
from abc import ABC, abstractmethod
from .store import StoreInterface, Local


class Attributes(ABC):
    """
    Abstract base class containing default attributes of Zarr ETLs
    These can be overriden in the ETL managers for a given ETL as needed
    """

    @classmethod
    def host_organization(self) -> str:
        """
        Name of the organization (your organization) hosting the data being published. Used in STAC metadata.
        """
        return ""  # e.g. "Arbol"

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        The name of each ETL is built recursively by appending each child class name to the inherited name

        Returns
        -------
        str
           Name of dataset

        """

    @classmethod
    @abstractmethod
    def collection(cls):
        """'
        Placeholder class for collection name
        """

    @property
    def file_type(cls):
        """
        Class method to populate with a string representing the file type of each child class (and edition if
        relevant), e.g. GRIB1 for ERA5 data, GRIB2 for RTMA, or NetCDF for Copernicus Marine Service

        Used to trigger file format-appropriate functions and methods for Kerchunking and Xarray operations.
        """

    @classmethod
    @abstractmethod
    def remote_protocol(cls):
        """
        Remote protocol string for MultiZarrToZarr and Xarray to use when opening input files. 'File' for local, 's3'
        for S3, etc. See fsspec docs for more details.
        """

    @classmethod
    @abstractmethod
    def identical_dims(cls):
        """
        List of dimension(s) whose values are identical in all input datasets. This saves Kerchunk time by having it
        read these dimensions only one time, from the first input file
        """

    @classmethod
    @abstractmethod
    def concat_dims(cls):
        """
        List of dimension(s) by which to concatenate input files' data variable(s) -- usually time, possibly with some
        other relevant dimension
        """

    @classmethod
    @abstractmethod
    def data_var(self) -> str:
        """Name of the relevant data variable in the original dataset"""

    @property
    def data_var_dtype(self) -> str:
        """
        Property specifying the data type of the data variable

        Returns
        -------
        str
            The final data type of the dataset's data variable
        """
        return "<f4"

    def spatial_resolution(self) -> float:
        """
        Property specifying the spatial resolution of a dataset in decimal degrees

        Returns
        -------
        float
            The spatial resolution of a dataset
        """

    def spatial_precision(self) -> float:
        """
        Property specifying the spatial resolution of a dataset in decimal degrees

        Returns
        -------
        float
            The spatial resolution of a dataset
        """

    @classmethod
    @abstractmethod
    def temporal_resolution(cls) -> str:
        """
        Returns the time resolution of the dataset as a string (e.g. "hourly", "daily", "monthly", etc.)

        Returns
        -------
        str
           Temporal resolution of the dataset

        """

    @classmethod
    def update_cadence(self) -> str:
        """
        Property specifying the frequency with which a dataset is updated
        Optional class method, may just be specified directly in the static metadata

        Returns
        -------
        str
            The update frequency of a dataset
        """

    @classmethod
    def missing_value_indicator(cls) -> str:
        """
        Default indicator of a missing value in a dataset

        Returns
        -------
        str
           Stand-in for a missing value

        """
        return ""

    @property
    def tags(cls) -> list[str]:
        """
        Default tag for a dataset. Prevents crashes on parse if no tags assigned.

        Returns
        -------
        list[str]
           Stand-in for a dataset's tags

        """
        return [""]

    @property
    def forecast(self) -> bool:
        """Forecast defaults to False, must override for actual forecast datasets"""
        return False

    @property
    def ensemble(self) -> bool:
        """Ensemble defaults to False, must override for actual ensemble datasets"""
        return False

    @property
    def hindcast(self) -> bool:
        """Hindcast defaults to False, must override for actual hindcast datasets"""
        return False

    @property
    def forecast_hours(self) -> list[int]:
        """To be overwritten by actual forecast datasets"""
        return list(None)

    @property
    def ensemble_numbers(self) -> list[int]:
        """To be overwritten by actual ensemble datasets"""
        return list(None)

    @property
    def hindcast_steps(self) -> list[int]:
        """To be overwritten by actual hindcast datasets"""
        return list(None)

    @property
    def has_nans(self) -> bool:
        """
        If True, disable quality checks for NaN values to prevent wrongful flags
        Default value set as False"""
        return False

    @classmethod
    def irregular_update_cadence(self) -> None | tuple[np.timedelta64, np.timedelta64]:
        """
        If a dataset doesn't update on a monotonic schedule return a tuple noting the lower and upper bounds of
        acceptable updates Intended to prevent time contiguity checks from short-circuiting valid updates for datasets
        with non-monotic update schedules
        """
        return None

    @property
    def bbox_rounding_value(self) -> int:
        """
        Value to round bbox values by. Specify within the dataset for very high resolution datasets
        to prevent mismatches with rounding behavior of old Arbol API.

        Returns
        -------
        int
            The number of decimal places to round bounding box values to.
        """
        return 5

    @property
    def store(self) -> StoreInterface:
        """
        Get the store interface object for the store the output will be written to.

        If it has not been previously set, a `etls.utils.store.Local` will be initialized and returned.

        Returns
        -------
        StoreInterface
            Object for interfacing with a Zarr's data store.
        """
        if not hasattr(self, "_store"):
            self._store = Local(self)
        return self._store

    @store.setter
    def store(self, value: StoreInterface):
        """
        Assign a `StoreInterface` to the store property. If the assigned value is not an instance of `StoreInterface`,
        a `ValueError` will be raised.

        Parameters
        ----------
        value
            An instance of `StoreInterface`

        Raises
        ------
        ValueError
            Raised if anything other than a `StoreInterface` is passed.
        """
        if isinstance(value, StoreInterface):
            self._store = value
        else:
            raise ValueError(f"Store must be an instance of {StoreInterface}")
