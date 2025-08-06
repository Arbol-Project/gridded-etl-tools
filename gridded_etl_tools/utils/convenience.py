import pathlib
import datetime
import io
import json
import random
from typing import Any, Iterator

from dateutil.parser import parse as parse_date
import deprecation
import natsort
import numpy as np
import pandas as pd
import xarray as xr

from .attributes import Attributes


class Convenience(Attributes):
    """
    Base class holding convenience methods for Zarr ETLs
    """

    # BASE DIRECTORIES

    def root_directory(self, refresh: bool = False):
        # ensure this is only calculated one time, at the beginning of the script
        if refresh or not hasattr(self, "_root_directory"):

            # Paths are relative to the working directory of the ETL manager, *not* the scripts
            self._root_directory = pathlib.Path.cwd()

        return self._root_directory

    @property
    def local_input_root(self):
        return self.root_directory() / "datasets"

    @property
    def output_root(self):
        return self.root_directory() / "climate"

    # NAMES

    def zarr_json_path(self) -> pathlib.Path:
        """
        A path to the local final Zarr

        Returns
        -------
        pathlib.Path
            The path to the local final Zarr JSON file
        """
        return self.local_input_root / "merged_zarr_jsons" / f"{self.dataset_name}_zarr.json"

    @classmethod
    def key(cls, alt_time_resolution: str = None) -> str:
        """
        Returns the key value that can identify this set in catalogs, registries, and metadata.
        The key by default takes the form of either name-measurement_span or name-today.

        If `append_date` is True, add today's date to the end of the string

        Parameters
        ----------
        alt_time_resolution : str, optional
            An alternative time resolution to use for the key. If not provided, the default time resolution is used.

        Returns
        -------
        str
            The formatted JSON key

        """
        return f"{cls.dataset_name}-{alt_time_resolution or cls.time_resolution}"

    # PATHS

    def local_input_path(self) -> pathlib.Path:
        """
        The path to local data is built recursively by appending each derivative's relative path to the previous
        derivative's path. If a custom input path is set, force return the custom path.
        """
        if self.custom_input_path:
            return pathlib.Path(self.custom_input_path)
        else:
            path = pathlib.Path(self.local_input_root) / pathlib.Path(self.relative_path())
            # Create directory if necessary
            path.mkdir(parents=True, mode=0o755, exist_ok=True)
            return path

    def relative_path(self) -> str:
        """
        The file folder hierarchy for a set. This should be a relative path so it can be appended to other root paths
        like `self.local_input_path()`

        Returns
        -------
        str
            The relative path that should be used for this set's data
        """
        return pathlib.Path(".")

    def input_files(self) -> Iterator[pathlib.Path]:
        """
        Iterate over the listing of local input files

        Returns
        -------
        Generator[pathlib.Path]
            Files from `self.local_input_path()`

        """
        root = pathlib.Path(self.local_input_path())
        for entry in natsort.natsorted(pathlib.Path(root).iterdir()):
            if not entry.name.startswith(".") and not entry.name.endswith(".idx") and entry.is_file():
                yield pathlib.Path(root / entry.name)

    def get_folder_path_from_date(self, date: datetime.datetime, omit_root: bool = False) -> str:
        """
        Return a folder path inside `self.output_root` with the folder name based on `self.temporal_resolution()`
        and the passed `datetime`. If `omit_root` is set, remove `self.output_root` from the path.

        Parameters
        ----------
        date : datetime.datetime
            datetime.datetime object representing the date to be appended to the folder name
        omit_root : bool, optional
            If False, prepent `self.output_root` to the beginning of the path, otherwise leave it off. Defaults to
            False

        Returns
        -------
        str
            Directory path derived from the date provided

        """
        if self.time_resolution == self.SPAN_HOURLY:
            date_format = self.DATE_HOURLY_FORMAT_FOLDER
        else:
            date_format = self.DATE_FORMAT_FOLDER
        path = pathlib.Path(self.relative_path()) / date.strftime(date_format)
        if not omit_root:
            path = pathlib.Path(self.output_root) / path
        return path

    def output_path(self, omit_root: bool = False) -> pathlib.Path:
        """
        Return the path to a directory where parsed climate data will be written, automatically determining the end
        date and base on that. If `omit_root` is set, remove `self.output_root` from the path. Override with
        `self.custom_output_path` if that member variable is set.

        Parameters
        ----------
        omit_root : bool, optional
            If False, prepend self.output_root to the beginning of the path, otherwise leave it off. Defaults to False.

        Returns
        -------
        str
            string representing the output directory path where climate data will be written
        """
        path = self.relative_path()
        if not omit_root:
            path = self.output_root / path
        return path

    # DATES

    def get_metadata_date_range(self) -> dict[str, datetime.datetime]:
        """
        Returns the date range in the metadata as datetime objects in a dict with `start` and `end` keys.

        On S3 and local, uses an existing Zarr to get the date range.

        Existing dates are assumed to be formatted as "%Y%m%d%H"

        Returns
        -------
        dict
            Two str: datetime.datetimes representing the start and end times in a STAC Item's metadata
        """
        date_format = "%Y%m%d%H"
        # Use existing Zarr attrs or raise an exception if there is no usable date attribute
        if self.store.has_existing:
            dataset = self.store.dataset()
            if "date range" in dataset.attrs:
                # Assume attr format is ['%Y%m%d%H', '%Y%m%d%H'], translate to datetime objects, then transform
                # into a dict with "start" and "end" keys
                return dict(  # pragma NO BRANCH, coverage is confused here for some reason
                    zip(
                        ("start", "end"),
                        (datetime.datetime.strptime(d, date_format) for d in dataset.attrs["date range"]),
                    )
                )
            else:
                raise ValueError(f"Existing date range not found in {dataset} attributes")
        else:
            raise ValueError(f"No existing dataset found to get date range from at {self.store}")

    @deprecation.deprecated("use dateutil.parser.parse")
    def convert_date_range(self, date_range: list) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Convert a JSON text/isoformat date range into a python datetime object range

        Parameters
        ----------
        date_range : list
            A list of length two containing isoformatted start and end date strings

        Returns
        -------
        tuple[datetime.datetime, datetime.datetime]
            A tuple of (datetime.datetime, datetime.datetime) representing a date range's start and end

        """
        start, end = [parse_date(date) for date in date_range]
        return start, end

    @deprecation.deprecated("use dateutil.parser.parse")
    def iso_to_datetime(self, isodate: str) -> datetime.datetime:
        """
        Get a datetime object from an ISO formatted date string

        Parameters
        ----------
        isodate : str
            An Isoformatted string representing a date

        Returns
        -------
        datetime.datetime
            The converted date

        """
        return parse_date(isodate)

    def numpydate_to_py(self, numpy_date: np.datetime64, **kwargs) -> datetime.datetime:
        """
        Convert a numpy datetime object to a python standard library datetime object

        Parameters
        ----------
        np.datetime64
            A numpy.datetime64 object to be converted
        kwargs : dict, optional
            Additional keyword arguments to pass to pd.Timestamp
            Most notably "tz" can be used to set the timezone of the returned datetime

        Returns
        -------
        datetime.datetime
            A datetime.datetime object

        """
        return pd.Timestamp(numpy_date, **kwargs).to_pydatetime()

    @staticmethod
    def today() -> str:
        """
        Convenience method to return today's date in Isoformat

        Returns
        -------
        str
            Today's date in Isoformat
        """
        return datetime.date.today().isoformat()

    # DATE RANGES

    def get_date_range_from_dataset(self, dataset: xr.Dataset) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Return the start and end date in a dataset's "time" dimension

        Parameters
        ----------
        dataset : xr.Dataset
            The xr.Dataset to be evaluated

        Returns
        -------
        tuple[datetime.datetime, datetime.datetime]
            A tuple defining the start and end date of a file's time dimension

        """
        if not hasattr(self, "time_dim"):
            self.set_key_dims()
        # if there is only one date, set start and end to the same value
        if dataset[self.time_dim].size == 1:
            values = dataset[self.time_dim].values
            assert len(values) == 1
            start = end = self.numpydate_to_py(values[0])
        else:
            start = self.numpydate_to_py(dataset[self.time_dim][0].values)
            end = self.numpydate_to_py(dataset[self.time_dim][-1].values)
        return start, end

    def get_date_range_from_file(
        self, path: str, backend_kwargs: dict = None, **kwargs
    ) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Open file and return the start and end date of the data. The dimension name used to store dates should be
        passed as `dimension`.

        Parameters
        ----------
        path : str
            Path to the input dataset file on disk
        backend_kwargs : dict, optional
            Backend arguments for the xr.open_dataset() method
        kwargs : dict, optional
            A dictionary of any additional kwargs that are specific to open_dataset, not the backend

        Returns
        -------
        tuple
            A tuple of datetime.datetime objects defining the start and end date of a file's time dimension
        """
        dataset = xr.open_dataset(path, backend_kwargs=backend_kwargs, **kwargs)
        return self.get_date_range_from_dataset(dataset)

    def date_range_to_string(self, date_range: tuple) -> tuple[str, str]:
        """
        Convert a tuple of datetime objects to a tuple of parseable strings. Necessary for Xarray metadata parsing.

        Parameters
        ----------
        date_range : tuple
            A (datetime.datetime, datetime.datetime) tuple containing the start and end dates of a date range

        Returns
        -------
        tuple
            A tuple of `%Y%m%d%H` formatted start and end dates of a date range

        """
        return (
            datetime.datetime.strftime(date_range[0], "%Y%m%d%H"),
            datetime.datetime.strftime(date_range[1], "%Y%m%d%H"),
        )

    def strings_to_date_range(
        self, date_range: tuple, parse_string: str = "%Y%m%d%H"
    ) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Convert a tuple of parseable strings to datetime objects. Necessary for Xarray metadata parsing.

        Parameters
        ----------
        tuple
            A tuple of `%Y%m%d%H` formatted start and end dates of a date range

        Returns
        -------
        date_range : tuple
            A (datetime.datetime, datetime.datetime) tuple containing the start and end dates of a date range

        """
        return (
            datetime.datetime.strptime(date_range[0], parse_string),
            datetime.datetime.strptime(date_range[1], parse_string),
        )

    def get_newest_file_date_range(self, **kwargs) -> datetime.datetime:
        """
        Return the date range of the newest local file

        Returns
        -------
        datetime.datetime
            The start and end date of the newest local file

        """
        return self.get_date_range_from_file(list(self.input_files())[-1], **kwargs)

    @property
    def next_date(self) -> datetime.datetime:
        """
        Return the next possible date for time_dim after the latest date in the current production dataset

        This function will raise an error for datasets with irregular cadences.

        Returns
        -------
        datetime.datetime
            The next possible date

        Raises
        ------
        ValueError
            Reject this request for any datasets with irregular cadences, as the next possible date
            cannot be programmatically derived
        """
        if self.update_cadence_bounds:
            raise ValueError(
                "Dataset has irregular update cadence, so the next date cannot be derived "
                "programmatically. Please locate the date manually."
            )
        if not hasattr(self, "time_dim"):
            self.set_key_dims()
        dataset = self.store.dataset()
        time_delta = dataset[self.time_dim].values[1] - dataset[self.time_dim].values[0]
        return self.numpydate_to_py(dataset[self.time_dim].values[-1] + time_delta)

    def get_next_date_as_date_range(self) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Return the next possible date for time_dim after the latest date in the current production dataset
        Used to programmatically feed the next possible date, and only that date, to an extract. This is
        useful for development and debugging

        This function will raise an error for datasets with irregular cadences.

        Returns
        -------
        tuple[datetime.datetime, datetime.datetime]
            The next possible date enclosed in a tuple, so it can be interpreted as a single
            datetime date range

        Raises
        ------
        ValueError
            Reject this request for any datasets with irregular cadences, as the next possible date
            cannot be programmatically derived
        """
        if self.update_cadence_bounds:
            raise ValueError(
                "Dataset has irregular update cadence, so the next date cannot be derived "
                "programmatically. Please locate the date manually."
            )
        return (self.next_date, self.next_date)

    # ETC

    def bbox_coords(self, dataset: xr.Dataset) -> tuple[float, float, float, float]:
        """
        Calculate bounding box coordinates from an Xarray dataset

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to use for extent calculations

        Returns
        -------
        tuple[float, float, float, float]
            The minimum X, minimum Y, maximum X, and maximum Y values of the dataset's bounding box extent
        """
        # Shoulc longitude/latitude be hardcoded names?
        return (
            round(float(dataset.longitude.values.min()), self.bbox_rounding_value),
            round(float(dataset.latitude.values.min()), self.bbox_rounding_value),
            round(float(dataset.longitude.values.max()), self.bbox_rounding_value),
            round(float(dataset.latitude.values.max()), self.bbox_rounding_value),
        )

    def json_to_bytes(self, obj: dict) -> bytes:
        """
        Convert a JSON object to a file type object (bytes). Primarily used for passing STAC metadata over HTTP

        Parameters
        ----------
        obj : dict
            The object (JSON) to be converted

        Returns
        -------
        bytes
            The json encoded as a file type object
        """
        return io.BytesIO(json.dumps(obj).encode("utf-8"))

    def check_if_new_data(self, compare_date: datetime.datetime) -> bool:
        """
        Check if the downloaded data contains any new records relative to the existing dataset.
        Return a boolean indicating whether to proceed with a transform/parse based on the presence of new records.

        Parameters
        ==========
        compare_date : datetime.datetime
            A cutoff date to compare against downloaded data; if any downloaded data is newer, move ahead with the
            parse. When updating, refers to the last datetime available in the existing dataset.

        Returns
        =======
        bool
            An indication of whether to proceed with a parse (True) or not (False)
        """

        # check if newest file on our server has newer data
        try:
            newest_file_end_date = self.get_newest_file_date_range()[1]
        except IndexError as e:
            self.info(
                f"Date range operation failed due to absence of input files. Exiting script. Full error message: {e}"
            )
            return False
        self.info(f"newest file ends at {newest_file_end_date}")
        if newest_file_end_date >= compare_date:
            self.info(f"newest file has newer data than our end date {compare_date}, triggering parse")
            return True
        else:
            self.info(f"newest file doesn't have data past our existing end date {compare_date}.")
            return False

    @classmethod
    def standardize_longitudes(cls, dataset: xr.Dataset) -> xr.Dataset:
        """
        Convert the longitude coordinates of a dataset from 0 - 360 to -180 to 180.

        Parameters
        ----------
        xr.Dataset
            A dataset with longitudes from 0 to 360
        Returns
        -------
        xr.Dataset
            A dataset with longitudes from -180 to 180

        """
        # Convert longitudes from 0 - 360 to -180 to 180
        standard_lon_coords = ((dataset.longitude.values + 180) % 360) - 180
        dataset = dataset.assign_coords(longitude=(dataset.longitude.dims, standard_lon_coords))
        return dataset.sortby(cls.spatial_dims)

    def get_random_coords(self, dataset: xr.Dataset) -> dict[str, Any]:
        """
        Derive a dictionary of random coordinates, one for each dimension in the input dataset

        Parameters
        ----------
        dataset
            An Xarray dataset

        Returns
        -------
            A dict of {str: Any} pairing each dimension to a randomly selected coordinate value
        """
        coords_dict = {}
        # We select from dims, not coords because inserts drop all non-time_dim coords
        for dim in dataset.dims:
            coords_dict.update({dim: random.choice(dataset[dim].values)})
        return coords_dict

    @property
    @deprecation.deprecated("Use the constant DatasetManager.EXTREME_VALUES_BY_UNIT")
    def extreme_values_by_unit(self):
        """
        Define minimum and maximum permissible values for common units

        Returns
        -------
        dict
            A dict of {str : (float, float)} representing the unit name
            and corresponding lower/upper value limits
        """
        return self.EXTREME_VALUES_BY_UNIT  # pragma NO COVER, this is a constant
