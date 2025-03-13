"""
Classes for managing CHIRPS global, gridded precipitation data
"""

import glob
import datetime
import pathlib
import xarray as xr
from abc import ABC

from gridded_etl_tools.dataset_manager import DatasetManager
from gridded_etl_tools.utils import extractor


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
        **kwargs,
    ):
        """
        Initialize a new CHIRPS object with appropriate chunking parameters.

        0.05 dataset size is time: 15000, latitude: 2000, longitude: 7200
        """
        super().__init__(requested_dask_chunks, requested_zarr_chunks, *args, **kwargs)
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
            "final lag in days": self.final_lag_in_days,
            "preliminary lag in days": self.preliminary_lag_in_days,
            "expected_nan_frequency": self.expected_nan_frequency,
        }

        return static_metadata

    organization = "My Organization"
    dataset_name = "chirps"

    def relative_path(self) -> str:
        return super().relative_path() / "chirps"

    collection_name = "CHIRPS"
    """Overall collection of data. Used for filling and referencing STAC Catalog."""

    time_resolution = DatasetManager.SPAN_DAILY
    """Increment size along the "time" coordinate axis"""

    data_var = "precip"

    standard_name = "precipitation_amount"

    long_name = "Precipitation"

    tags = ["Precipitation"]
    """Tags for data in dClimate"""

    unit_of_measurement = "mm"
    """Unit of measurement for the component key (data variable)"""

    dataset_start_date = datetime.datetime(1981, 1, 1, 0)
    """First date in dataset. Used to populate corresponding encoding and metadata."""

    has_nans: bool = True
    """If True, disable quality checks for NaN values to prevent wrongful flags"""

    missing_value = -9999
    """
    Value within the source data that should be automatically converted to 'nan' by Xarray.
    Cannot be empty/None or Kerchunk will fail, so use -9999 if no NoData value actually exists in the dataset.
    """

    dataset_download_url = "ftp.chc.ucsb.edu"
    """URL to download location of the dataset. May be an FTP site, API base URL, or otherwise."""

    file_type = "NetCDF"
    """
    File type of raw data.
    Used to trigger file format-appropriate functions and methods for Kerchunking and Xarray operations.
    """

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

    bbox_rounding_value = 3
    """Value to round bbox values by"""

    def extract(self, *, date_range: tuple[datetime.datetime, datetime.datetime] | None = None, **kwargs) -> bool:
        """
        Download climate data netCDF files from CHIRPS's FTP server. Files are assumed to be stored in the format
        'chirps-*YEAR*.nc' where year is any year in the given date range. If the date range is unspecified, check
        for an existing dataset and use the end date of the existing dataset as the start date for the date range.

        Check newest file downloaded and return `True` if it has newer data than in the existing Zarr or `False`
        otherwise.

        Parameters
        ----------
        date_range: list, optional
            Optional start and end date to extract. Assumes two isoformatted date strings.
            Defaults to None.

        Returns
        -------
        bool
            `True` if local data has newer data than the existing Zarr or `False` otherwise
        """
        super().extract()
        if not date_range:
            # Find previous end date so the manager can start downloading the day after it
            try:
                self.info("Calculating new start date based on end date in STAC metadata")
                end_date = self.get_metadata_date_range()["end"] + datetime.timedelta(days=1)
            except (KeyError, ValueError):
                self.info("Because no metadata found, starting file search from beginning {self.dataset_start_date}")
                end_date = self.dataset_start_date
            download_year_range = range(end_date.year, datetime.datetime.now().year + 1)
        else:
            self.info("Calculating start and end dates based on the provided date range.")
            end_date = date_range[1]
            download_year_range = range(date_range[0].year, end_date.year + 1)

        # Connect to CHIRPS FTP and request files for all needed years
        with extractor.FTPExtractor(self, self.dataset_download_url) as ftp:
            ftp.cwd = self.remote_path
            requests: list[pathlib.Path] = []
            # Loop through every year in the date range and queue requests
            for year in download_year_range:
                pattern = rf"chirps-.+{year}.+\.nc"
                requests.extend(ftp.find(pattern))
            ftp.pool([{"source": request, "destination": self.local_input_path()} for request in requests])

        # Check if newest local file has newer data
        return self.check_if_new_data(end_date)

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
            self.convert_to_lowest_common_time_denom(raw_files=yearlies, keep_originals=keep_originals)

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
        super().remove_unwanted_fields(dataset)
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
            dataset[self.data_var].attrs.pop(key, None)
            dataset[self.data_var].encoding.pop(key, None)


class CHIRPSFinal(CHIRPS, ABC):
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

    final_lag_in_days = 30


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
        return "/pub/org/chc/products/CHIRPS-2.0/global_daily/netcdf/p05/"

    @property
    def spatial_resolution(self) -> float:
        """Increment size along the latitude/longitude coordinate axis"""
        return 0.05

    expected_nan_frequency = 0.738


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
        )
        kwargs.update(chunks)
        kwargs["console_log"] = False
        super().__init__(*args, **kwargs)

    dataset_name = f"{CHIRPSFinal.dataset_name}_25"

    def relative_path(self) -> pathlib.Path:
        """Relative path used to store data under 'datasets' and 'climate' folders"""
        return super().relative_path() / "25"

    @property
    def remote_path(self) -> str:
        """path on CHIRPS server to relevant files"""
        return "/pub/org/chc/products/CHIRPS-2.0/global_daily/netcdf/p25/"

    @property
    def spatial_resolution(self) -> float:
        """Increment size along the latitude/longitude coordinate axis"""
        return 0.25

    expected_nan_frequency = 0.72


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
        return "/pub/org/chc/products/CHIRPS-2.0/prelim/global_daily/netcdf/p05/"

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

    final_lag_in_days = 30

    preliminary_lag_in_days = 6

    expected_nan_frequency = 0.73872
