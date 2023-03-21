
from abc import ABC, abstractmethod

class Attributes(ABC):
    """
    Abstract base class containing default attributes of Zarr ETLs
    These can be overriden in the ETL managers for a given ETL as needed
    """

    @property
    def data_host(self) -> str:
        """
        The name of the institution, organization, or person publishing this data
        """
        return "Arbol"


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
        ...


    @classmethod
    @abstractmethod
    def collection(cls):
        """'
        Placeholder class for collection name
        """
        ...


    @property
    def file_type(cls):
        """
        Class method to populate with a string representing the file type of each child class (and edition if relevant),
        e.g. GRIB1 for ERA5 data, GRIB2 for RTMA, or NetCDF for Copernicus Marine Service

        Used to trigger file format-appropriate functions and methods for Kerchunking and Xarray operations.
        """
        ...


    @classmethod
    @abstractmethod
    def remote_protocol(cls):
        """
        Remote protocol string for MultiZarrToZarr and Xarray to use when opening input files. 'File' for local, 's3' for S3, etc.
        See fsspec docs for more details.
        """
        ...


    @classmethod
    @abstractmethod
    def identical_dims(cls):
        """
        List of dimension(s) whose values are identical in all input datasets. This saves Kerchunk time by having it read these 
        dimensions only one time, from the first input file
        """
        ...


    @classmethod
    @abstractmethod
    def concat_dims(cls):
        """
        List of dimension(s) by which to concatenate input files' data variable(s) -- usually time, possibly with some other relevant dimension
        """
        ...


    @property
    def data_var_dtype(self) -> str:
        """
        Property specifying the data type of the data variable

        Returns
        -------
        str
            The final data type of the dataset's data variable
        """
        return '<f4'


    def spatial_resolution(self) -> float:
        """
        Property specifying the spatial resolution of a dataset in decimal degrees

        Returns
        -------
        float
            The spatial resolution of a dataset
        """
        ...


    def spatial_precision(self) -> float:
        """
        Property specifying the spatial resolution of a dataset in decimal degrees

        Returns
        -------
        float
            The spatial resolution of a dataset
        """
        ...


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
        ...


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
        ...


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