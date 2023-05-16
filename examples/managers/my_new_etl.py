##### my_new_etl.py
#
# Template classes for creating a new gridded climate data ETL.
# All filled fields are examples that can be replaced; all unfilled fields must be filled by the user.

import datetime
import pathlib

from gridded_etl_tools.dataset_manager import DatasetManager


class MyNewETL(DatasetManager):
    """
    Base class for datasets from a provider. For example's sake assumes that such data are published in NetCDF format
    """

    def __init__(
            self, *args,
            requested_dask_chunks={"time": 1769, "latitude": 24, "longitude": -1},
            requested_zarr_chunks={"time": 1769, "latitude": 24, "longitude": 24},
            requested_ipfs_chunker="size-2304", **kwargs
            ):
        """
        Initialize a new ETL object with appropriate chunking parameters.
        """
        super().__init__(requested_dask_chunks, requested_zarr_chunks, requested_ipfs_chunker, *args, **kwargs)
        self.standard_dims = ["latitude", "longitude", "time"]

    @property
    def static_metadata(self) -> dict:
        """
        dict containing static fields in the metadata
        """
        static_metadata = {
            "coordinate reference system": "EPSG:4326",
            "spatial resolution": self.spatial_resolution,
            "spatial precision": 0.01,
            "temporal resolution": self.temporal_resolution(),
            "update cadence": "daily",
            "provider url": "",
            "data download url": "",
            "publisher": "",
            "title": "",
            "provider description": "",
            "dataset description": "",
            "license": "",
            "terms of service": "",
            "name": self.name(),
            "updated": str(datetime.datetime.now()),
            "missing value": self.missing_value_indicator(),
            "tags": self.tags,
            "standard name": self.standard_name,
            "long name": self.long_name,
            "unit of measurement": self.unit_of_measurement
            }

        return static_metadata

    @classmethod
    def collection(cls) -> str:
        """
        Overall collection of data. Used for filling STAC Catalogue.
        """
        return "Dataset Collection"

    @classmethod
    def name(cls) -> str:
        """
        The name of the dataset.
        Used as a command-line trigger and to populate directory names, so whitespaces must be undesrcored or hyphenated
        """
        return "dataset_name"

    @classmethod
    def temporal_resolution(cls) -> str:
        """Incremental step size for temporal values in the dataset's time dimension"""
        return cls.SPAN_DAILY

    @property
    def dataset_start_date(self) -> datetime.datetime:
        """First date in dataset. Used to populate corresponding encoding and metadata."""
        return datetime.datetime(1979, 1, 1, 0)

    @classmethod
    def missing_value_indicator(cls) -> str:
        """
        What value should be interpreted and masked as NA.
        Failure to specify this correctly may result in values incorrectly entering calculations and/or coordinate values being masked
        """
        return -9.96921e+36  # replace

    def relative_path(self) -> pathlib.Path:
        """Relative path in which to output raw files or a final zarr"""
        return super().relative_path() / self.name()

    @property
    def file_type(cls) -> str:
        """
        File type of raw data. Used to trigger file format-appropriate functions and methods for Kerchunking and Xarray operations.
        """
        return "NetCDF"

    @classmethod
    def remote_protocol(cls) -> str:
        """
        Remote protocol string for MultiZarrToZarr and Xarray to use when opening input files. 'File' for local, 's3' for S3, etc.
        See fsspec docs for more details.
        """
        return "file"

    @classmethod
    def identical_dims(cls) -> str:
        """
        List of dimension(s) whose values are identical in all input datasets. This saves Kerchunk time by having it read these
        dimensions only one time, from the first input file
        """
        return ["latitude", "longitude"]

    @classmethod
    def concat_dims(cls) -> str:
        """
        List of dimension(s) by which to concatenate input files' data variable(s) -- usually time, possibly with some other relevant dimension
        """
        return ["time"]

    def extract(self, rebuild: bool = False, date_range: list[datetime.datetime, datetime.datetime] = None, *args, **kwargs) -> bool:
        """
        Check the remote from the end year of or after our data's end date. Download necessary files. Check
        newest file and return `True` if it has newer data than us or `False` otherwise.

        Pseudocode for the basic logic behind retrievals is provided below

        Returns
        -------
        bool
            A boolean indicating whether to proceed with a parse operation or not
        """
        parsing_should_happen = False
        # Insert custom retrieval code here
        if rebuild:
            # download everything, regardless of what data already exists
            pass  # insert download everything logic here
        elif date_range:
            # only download for the selected date range
            pass  # insert download everything logic here
        else:
            # if this is the first time an ETL is running, download all available data
            # if this is an update to an existing dataset,download whatever new or updated data has been published since the last time the dataset was updated
            new_data_found = False  # set conditions here
            if new_data_found:
                # download only the new data, then trigger a parse
                parsing_should_happen = True

        # Trigger a parse based on whether new data was found / rebuild flag provided
        if parsing_should_happen | rebuild:
            self.info("Conditions met to trigger a parse")
            return True
        else:
            self.info("Conditions not met to trigger a parse")
            return False

    def prepare_input_files(self, keep_originals: bool = False):
        """
        Command line tools converting raw downloaded data to daily / hourly data
        Normally `convert_to_lowest_common_time_denom` and/or `ncs_to_nc4s` are used here,
        perhaps in combinations, perhaps overloaded, perhaps paired with custom processing
        """
        ...


class MyNewETLPrecip(MyNewETL):
    """
    Base class for precip sets
    """

    @classmethod
    def name(cls) -> str:
        return f"{super().name()}_precip"

    def relative_path(self) -> pathlib.Path:
        return super().relative_path() / "precip"

    @property
    def tags(self) -> list[str]:
        """Tags for data to enable filtering"""
        return ["Precipitation"]

    @property
    def standard_name(self) -> str:
        """Short form name, as per the Climate and Forecasting Metadata Convention"""
        return "precipitation_amount"

    @property
    def long_name(self) -> str:
        """Long form name, as per the Climate and Forecasting Metadata Convention"""
        return "Precipitation"

    @property
    def unit_of_measurement(self) -> str:
        return "mm"


class MyNewETLTemp(MyNewETL):
    """
    Base class for gridded temperature data
    """

    @classmethod
    def name(cls) -> str:
        return f"{super().name()}_temp"

    def relative_path(self) -> pathlib.Path:
        return super().relative_path() / "temp"

    @property
    def tags(self) -> list[str]:
        """Tags for data to enable filtering"""
        return ["Temperature"]

    @property
    def unit_of_measurement(self) -> str:
        return "degC"

    @property
    def spatial_resolution(self) -> float:
        return 0.5


class MyNewETLTempMin(MyNewETLTemp):
    """
    Gridded minimum temperature data manager class
    """
    @classmethod
    def name(cls) -> str:
        return f"{super().name()}_min"

    def relative_path(self) -> pathlib.Path:
        return super().relative_path() / "min"

    def data_var(self) -> str:
        """
        Name of the column in the original data
        """
        return "tmin"

    @property
    def standard_name(self) -> str:
        """Short form name, as per the Climate and Forecasting Metadata Convention"""
        return "air_temperature"

    @property
    def long_name(self) -> str:
        """Long form name, as per the Climate and Forecasting Metadata Convention"""
        return "Daily Minimum Near-Surface Air Temperature"


if __name__ == "__main__":
    MyNewETL().run_etl_as_script()
