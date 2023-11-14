"""
Classes for managing CHIRPS global, gridded precipitation data
"""

import glob
import datetime
import pathlib
import re
import requests
import xarray as xr
from gridded_etl_tools.dataset_manager import DatasetManager


class CHIRPS(DatasetManager):
    """
    The base class for any CHIRPS set using Arbol's data architecture. It is a superclass of both CHIRPS Final (monthly
    updates of .05 and .25 resolution from 1981) and CHIRPS Prelim (weekly updates of 0.05 resolution, from 2016 to
    present).
    """

    def __init__(
        self,
        *args,
        # 0.05 dataset size is time: 15000, latitude: 2000, longitude: 7200
        requested_dask_chunks={"time": 200, "latitude": 25, "longitude": -1},  # 144 MB
        requested_zarr_chunks={"time": 200, "latitude": 25, "longitude": 50},  # 1 MB
        requested_ipfs_chunker="size-5000",
        **kwargs,
    ):
        """
        Initialize a new CHIRPS object with appropriate chunking parameters.

        0.05 dataset size is time: 15000, latitude: 2000, longitude: 7200
        """
        super().__init__(requested_dask_chunks, requested_zarr_chunks, requested_ipfs_chunker, *args, **kwargs)
        self.standard_dims = ["latitude", "longitude", "time"]

    @property
    def static_metadata(self):
        """
        Dict containing static fields in the metadata. These will be populated into STAC metadata and Zarr metadata.
        Fields that are static should be manually specified here. Fields that change per child class should be defined
        as properties or class methods under the relevant child class
        """
        static_metadata = {
            "coordinate reference system": "EPSG:4326",
            "update cadence": self.update_cadence,
            "temporal resolution": self.time_resolution,
            "spatial resolution": self.spatial_resolution,
            "spatial precision": 0.00001,
            "provider url": "http://chg.geog.ucsb.edu/",
            "data download url": self.dataset_download_url,
            "publisher": "Climate Hazards Group, University of California at Santa Barbara",
            "title": "CHIRPS Version 2.0",
            "provider description": "The Climate Hazards Center is an alliance of multidisciplinary scientists and "
            "food security analysts utilizing climate and crop models, satellite-based earth observations, and "
            "socioeconomic data sets to predict and monitor droughts and food shortages among the world's most "
            "vulnerable populations. Through partnerships with USAID, USGS, and FEWS NET, the CHC provides early "
            "warning to save lives and secure livelihoods.",
            "dataset description": (
                "Climate Hazards center InfraRed Precipitation with Station data (CHIRPS) is a 30+ year quasi-global "
                "rainfall data set. Spanning 50°S-50°N (and all longitudes), starting in 1981 to near-present, CHIRPS "
                "incorporates 0.05° resolution satellite imagery with in-situ station data to create gridded rainfall "
                "time series for trend analysis and seasonal drought monitoring. For more information about CHIRPS "
                "data, visit http://chg.geog.ucsb.edu/data/chirps/index.html or "
                "http://chg-wiki.geog.ucsb.edu/wiki/CHIRPS_FAQ. "
                "For full technical documentation visit http://pubs.usgs.gov/ds/832/"
            ),
            "license": "Creative Commons Attribution 3.0",
            "terms of service": "To the extent possible under the law, Pete Peterson has waived all copyright and "
            "related or neighboring rights to CHIRPS. CHIRPS data is in the public domain as registered with "
            "Creative Commons.",
            "name": self.dataset_name,
            "updated": str(datetime.datetime.now()),
            "missing value": self.missing_value,
            "tags": self.tags,
            "standard name": self.standard_name,
            "long name": self.long_name,
            "unit of measurement": self.unit_of_measurement,
        }

        return static_metadata

    @classmethod
    def host_organization(cls) -> str:
        return "My Organization"

    dataset_name = "chirps"

    def relative_path(self) -> str:
        return super().relative_path() / "chirps"

    collection_name = "CHIRPS"
    """Overall collection of data. Used for filling and referencing STAC Catalog."""

    time_resolution = DatasetManager.SPAN_DAILY
    """Increment size along the "time" coordinate axis"""

    @classmethod
    def data_var(self) -> str:
        """Name of the relevant data variable in the original dataset"""
        return "precip"

    @property
    def standard_name(self):
        """Short form name, as per the Climate and Forecasting Metadata Convention"""
        return "precipitation_amount"

    @property
    def long_name(self):
        """Long form name, as per the Climate and Forecasting Metadata Convention"""
        return "Precipitation"

    @property
    def tags(self):
        """Tags for data to enable filtering"""
        return ["Precipitation"]

    @property
    def unit_of_measurement(self):
        """Unit of measurement for the component key (data variable)"""
        return "mm"

    @property
    def dataset_start_date(self):
        """First date in dataset. Used to populate corresponding encoding and metadata."""
        return datetime.datetime(1981, 1, 1, 0)

    has_nans: bool = True
    """If True, disable quality checks for NaN values to prevent wrongful flags"""

    missing_value = -9999
    """
    Value within the source data that should be automatically converted to 'nan' by Xarray.
    Cannot be empty/None or Kerchunk will fail, so use -9999 if no NoData value actually exists in the dataset.
    """

    @property
    def dataset_download_url(self) -> str:
        """URL to download location of the dataset. May be an FTP site, API base URL, or otherwise."""
        return "https://data.chc.ucsb.edu/products/CHIRPS-2.0"

    @property
    def file_type(self) -> str:
        """
        File type of raw data. Used to trigger file format-appropriate functions and methods for Kerchunking and Xarray
        operations.
        """
        return "NetCDF"

    protocol = "file"
    """
    Remote protocol string for MultiZarrToZarr and Xarray to use when opening input files. 'File' for local, 's3'
    for S3, etc. See fsspec docs for more details.
    """

    identical_dimensions = ["latitude", "longitude"]
    """
    List of dimension(s) whose values are identical in all input datasets. This saves Kerchunk time by having it
    read these dimensions only one time, from the first input file
    """

    concat_dimensions = ["time"]
    """
    List of dimension(s) by which to concatenate input files' data variable(s) -- usually time, possibly with some
    other relevant dimension
    """

    @property
    def bbox_rounding_value(self) -> int:
        """Value to round bbox values by"""
        return 3

    def extract(self, date_range: list[datetime.datetime, datetime.datetime] = None, *args, **kwargs) -> bool:
        """
        Check CHIRPS HTTP server for files from the end year of or after our data's end date. Download necessary files.
        Check newest file and return `True` if it has newer data than us or `False` otherwise.

        Parameters
        ----------
        date_range: list, optional
            A flag to specify a date range for download (and parsing). Assumes two isoformatted date strings. Defaults
            to None.

        Returns
        -------
        bool
            A boolean indicating whether to proceed with a parse operation or not
        """
        # Find previous end date so the manager can start downloading the day after it
        if not date_range:
            try:
                self.info("Calculating new start date based on end date in STAC metadata")
                end_date = self.get_metadata_date_range()["end"] + datetime.timedelta(days=1)
            except (KeyError, ValueError):
                self.info(
                    "Because no metadata found, starting file search from the dataset beginning of "
                    f"{self.dataset_start_date}"
                )
                end_date = self.dataset_start_date
            download_year_range = range(end_date.year, datetime.datetime.now().year + 1)
        else:
            self.info("Calculating start and end dates based on the provided date range.")
            end_date = date_range[1]
            download_year_range = range(date_range[0].year, end_date.year + 1)
        # find all files in the relevant remote server folder
        url = f"{self.dataset_download_url}/{self.remote_path}"
        self.info(f"connecting to {url}")
        index = requests.get(url).text
        # loop through every year from end date until present year and download any files that are newer than ones we
        # have on our server
        for year in download_year_range:
            pattern = rf"<a.+>(chirps-.+{year}.+\.nc)</a></td><td[^>]+>([^<]+[0-9])\s*</td>"
            matches = re.findall(pattern, index, re.MULTILINE)
            if len(matches) > 0:
                file_name, _ = matches[0]
                local_path = self.local_input_path() / file_name
                self.info(f"downloading remote file {file_name}")
                remote_file = requests.get(f"{url}{file_name}").content
                with open(local_path, "wb") as local_file:
                    local_file.write(remote_file)
        # check if newest file on our server has newer data
        try:
            newest_file_end_date = self.get_newest_file_date_range()[1]
        except IndexError as e:
            self.info(
                f"Date range operation failed due to absence of input files. Exiting script. Full error message: {e}"
            )
            return False
        self.info(f"newest file ends at {newest_file_end_date}")
        if newest_file_end_date >= end_date:
            self.info(f"newest file has newer data than our end date {end_date}, triggering parse")
            return True
        else:
            self.info(f"newest file doesn't have data past our existing end date {end_date}")
            return False

    def prepare_input_files(self, keep_originals: bool = False):
        """
        Convert each of the input files (and associated metadata files) to a collection of daily netCDF4 classic files
        suitable for reading by Kerchunk and intake into Xarray. This allows us to stack data into modern, performant
        N-Dimensional Zarr data.

        Parameters
        ----------
        keep_originals: bool, optional
            A flag to preserve the original files for debugging purposes. Defaults to False.
        """
        input_dir = pathlib.Path(self.local_input_path())
        yearlies = [pathlib.Path(file) for file in glob.glob(str(input_dir / "*.nc"))]
        if len(yearlies) == 0:
            if glob.glob(str(input_dir / "*.nc4")):
                self.info("Only converted NC4s found, skipping preparation step")
            if not glob.glob(str(input_dir / "*.nc4")):
                raise FileNotFoundError(
                    "Neither yearly files nor converted NC4s found in input directory. Please provide data before "
                    "processing"
                )
        else:
            # Convert input files to daily NetCDFs
            self.info(f"Converting {(len(list(yearlies)))} yearly NetCDF file(s) to daily NetCDFs")
            self.convert_to_lowest_common_time_denom(yearlies, keep_originals)
            # Convert all NCs to NC4s
            self.ncs_to_nc4s(keep_originals)

        self.info("Finished preparing input files")

    def remove_unwanted_fields(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Function to append to or update key metadata information to the attributes and encoding of the output Zarr.
        Extends existing class method to create attributes or encoding specific to this dataset.

        Parameters
        ----------
        dataset: xr.Dataset
            The dataset prepared for parsing, before removing unwanted fields specific to the dataset

        Returns
        -------
        dataset: xarray.Dataset dataset
            The dataset prepared for parsing, after removing unwanted fields specific to the dataset
        """
        dataset = super().remove_unwanted_fields(dataset)
        for variable in dataset.variables:
            dataset[variable].encoding["_FillValue"] = self.missing_value
        # Remove extraneous data from the data variable's attributes
        keys_to_remove = [
            "Conventions",
            "history",
            "version",
            "date_created",
            "creator_name",
            "creator_email",
            "institution",
            "documentation",
            "reference",
            "comments",
            "acknowledgements",
            "ftp_url",
            "website",
            "faq",
            "zlib",
            "shuffle",
            "complevel",
            "contiguous",
            "source",
            "original_shape",
            "missing_value",
        ]
        for key in keys_to_remove:
            dataset.attrs.pop(key, None)
            dataset[self.data_var()].attrs.pop(key, None)
            dataset[self.data_var()].encoding.pop(key, None)

        return dataset


class CHIRPSFinal(CHIRPS):
    """
    A class for finalized CHIRPS data
    """

    dataset_name = f"{CHIRPS.dataset_name}_final"

    def relative_path(self) -> pathlib.Path:
        return super().relative_path() / "final"

    @property
    def update_cadence(self) -> str:
        return "monthly"

    def populate_metadata(self):
        super().populate_metadata()
        self.metadata["revision"] = "final"


class CHIRPSFinal05(CHIRPSFinal):
    """
    Finalized CHIRPS data at 0.05 resolution
    """

    dataset_name = f"{CHIRPSFinal.dataset_name}_05"

    def relative_path(self) -> pathlib.Path:
        """Relative path used to store data under 'datasets' and 'climate' folders"""
        return super().relative_path() / "05"

    @property
    def remote_path(self) -> str:
        """path on CHIRPS server to relevant files"""
        return "global_daily/netcdf/p05/"

    @property
    def spatial_resolution(self) -> float:
        """Increment size along the latitude/longitude coordinate axis"""
        return 0.05


class CHIRPSFinal25(CHIRPSFinal):
    """
    Finalized CHIRPS data at 0.25 resolution
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a new CHIRPS object with appropriate chunking parameters.
        """
        # 0.25 dataset size is time: 15000, latitude: 400, longitude: 1440
        chunks = dict(
            requested_dask_chunks={"time": 500, "latitude": 40, "longitude": -1},  # 115 MB
            requested_zarr_chunks={"time": 500, "latitude": 40, "longitude": 40},  # 3.2 MB
            requested_ipfs_chunker="size-6400",
        )
        kwargs.update(chunks)
        super().__init__(*args, **kwargs)

    dataset_name = f"{CHIRPSFinal.dataset_name}_25"

    def relative_path(self) -> pathlib.Path:
        """Relative path used to store data under 'datasets' and 'climate' folders"""
        return super().relative_path() / "25"

    @property
    def remote_path(self) -> str:
        """path on CHIRPS server to relevant files"""
        return "global_daily/netcdf/p25/"

    @property
    def spatial_resolution(self) -> float:
        """Increment size along the latitude/longitude coordinate axis"""
        return 0.25


class CHIRPSPrelim05(CHIRPS):
    """
    Preliminary CHIRPS data at 0.05 resolution
    """

    dataset_name = f"{CHIRPS.dataset_name}_prelim_05"

    def relative_path(self) -> pathlib.Path:
        """Relative path used to store data under 'datasets' and 'climate' folders"""
        return super().relative_path() / "prelim" / "05"

    @property
    def remote_path(self) -> str:
        """path on CHIRPS server to relevant files"""
        return "prelim/global_daily/netcdf/p05/"

    @property
    def spatial_resolution(self) -> float:
        """Increment size along the latitude/longitude coordinate axis"""
        return 0.05

    @property
    def dataset_start_date(self):
        """First date in dataset. Used to populate corresponding encoding and metadata."""
        return datetime.datetime(2019, 1, 1, 0)

    @property
    def update_cadence(self) -> str:
        return "weekly"

    def populate_metadata(self):
        super().populate_metadata()
        self.metadata["revision"] = "preliminary"


if __name__ == "__main__":
    CHIRPS().run_etl_as_script()
