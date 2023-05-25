import os
import gc
import pathlib
import natsort
import datetime
import re
import ftplib
import io
import json

import pandas as pd
import numpy as np
import xarray as xr

from .attributes import Attributes
from .store import IPLD


class Convenience(Attributes):
    """
    Base class holding convenience methods for Zarr ETLs
    """

    # BASE DIRECTORIES

    def root_directory(self, refresh: bool = False):
        if refresh or not hasattr(self, "_root_directory"):  # ensure this is only calculated one time, at the beginning of the script
            self._root_directory = pathlib.Path.cwd()  # Paths are relative to the working directory of the ETL manager, *not* the scripts
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
        A path to the virtual Zarr

        Returns
        -------
        pathlib.Path
            The path to the virtual Zarr JSON file
        """
        return self.root_directory() / f"{self.name()}_zarr.json"

    @classmethod
    def json_key(cls, append_date: bool = False) -> str:
        """
        Returns the key value that can identify this set in a JSON file. JSON key takes the form of either name-measurement_span or name-today.
        If `append_date` is True, add today's date to the end of the string

        Parameters
        ----------
        append_date : bool, optional
            Whether to add today's date to the end of the json_key string

        Returns
        -------
        str
            The formatted JSON key

        """
        key = f"{cls.name()}-{cls.temporal_resolution()}"
        if append_date:
            key = f"{key}-{datetime.datetime.now().strftime(cls.DATE_FORMAT_FOLDER)}"
        return key

    # PATHS

    def local_input_path(self) -> str:
        """
        The path to local data is built recursively by appending each derivative's relative path to the previous derivative's
        path. If a custom input path is set, force return the custom path.
        """
        if self.custom_input_path:
            return pathlib.Path(self.custom_input_path)
        else:
            path = pathlib.Path(self.local_input_root) / pathlib.Path(
                self.relative_path()
            )
            # Create directory if necessary
            path.mkdir(parents=True, mode=0o755, exist_ok=True)
            return path

    def relative_path(self) -> str:
        """
        The file folder hierarchy for a set. This should be a relative path so it can be appended to other root paths like
        `self.local_input_path()` and `self.output_path()`

        Returns
        -------
        str
            The relative path that should be used for this set's data

        """
        return pathlib.Path(".")

    def input_files(self) -> list:
        """
        Iterator for iterating through the list of local input files

        Returns
        -------
        list
            List of input files from `self.local_input_path()`

        """
        root = pathlib.Path(self.local_input_path())
        for entry in natsort.natsorted(pathlib.Path(root).iterdir()):
            if (
                not entry.name.startswith(".")
                and not entry.name.endswith(".idx")
                and entry.is_file()
            ):
                yield pathlib.Path(root / entry.name)

    def get_folder_path_from_date(
        self, date: datetime.datetime, omit_root: bool = False
    ) -> str:
        """
        Return a folder path inside `self.output_root` with the folder name based on `self.temporal_resolution()`
        and the passed `datetime`. If `omit_root` is set, remove `self.output_root` from the path.

        Parameters
        ----------
        date : datetime.datetime
            datetime.datetime object representing the date to be appended to the folder name
        omit_root : bool, optional
            If False, prepent `self.output_root` to the beginning of the path, otherwise leave it off. Defaults to False

        Returns
        -------
        str
            Directory path derived from the date provided

        """
        if self.temporal_resolution() == self.SPAN_HOURLY:
            date_format = self.DATE_HOURLY_FORMAT_FOLDER
        else:
            date_format = self.DATE_FORMAT_FOLDER
        path = pathlib.Path(self.relative_path()) / date.strftime(date_format)
        if not omit_root:
            path = pathlib.Path(self.output_root) / path
        return path

    def output_path(self, omit_root: bool = False) -> str:
        """
        Return the path to a directory where parsed climate data will be written, automatically determining the end date and
        base on that. If `omit_root` is set, remove `self.output_root` from the path. Override with `self.custom_output_path`
        if that member variable is set.

        Parameters
        ----------
        omit_root : bool, optional
            If False, prepend self.output_root to the beginning of the path, otherwise leave it off. Defaults to False.

        Returns
        -------
        str
            string representing the output directory path where climate data will be written
        """
        if self.custom_output_path is not None:
            return self.custom_output_path
        else:
            path = self.relative_path()
            if not omit_root:
                path = self.output_root / path
            return path

    def create_output_path(self):
        """
        Make output directory
        """
        os.makedirs(self.output_path(), 0o755, exist_ok=True)

    # DATES

    def get_metadata_date_range(self) -> dict:
        """
        Returns the date range in the metadata as datetime objects in a dict with `start` and `end` keys.

        On IPLD, uses STAC to get the date. On S3 and local, uses an existing Zarr.

        Existing dates are assumed to be formatted as "%Y%m%d%H"

        Returns
        -------
        dict
            Two str: datetime.datetimes representing the start and end times in a STAC Item's metadata
        """
        date_format = "%Y%m%d%H"
        if isinstance(self.store, IPLD):
            # Use STAC
            metadata = self.load_stac_metadata()
            return {
                "start": datetime.datetime.strptime(
                    metadata["properties"]["date range"][0], date_format
                ),
                "end": datetime.datetime.strptime(
                    metadata["properties"]["date range"][1], date_format
                ),
            }
        else:
            # Use existing Zarr attrs or raise an exception if there is no usable date attribute
            if self.store.has_existing:
                dataset = self.store.dataset()
                if "date range" in dataset.attrs:
                    # Assume attr format is ['%Y%m%d%H', '%Y%m%d%H'], translate to datetime objects, then transform into a dict with "start" and "end" keys
                    return dict(
                        zip(
                            ("start", "end"),
                            (
                                datetime.datetime.strptime(d, date_format)
                                for d in dataset.attrs["date range"]
                            ),
                        )
                    )
                else:
                    raise ValueError(
                        f"Existing date range not found in {dataset} attributes"
                    )
            else:
                raise ValueError(
                    f"No existing dataset found to get date range from at {self.store}"
                )

    def convert_date_range(
        self, date_range: list
    ) -> tuple[datetime.datetime, datetime.datetime]:
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
        if re.match(".+/.+/.+", date_range[0]):
            start, end = [
                datetime.datetime.strptime(d, self.DATE_FORMAT_METADATA)
                for d in date_range
            ]
        else:
            start, end = [datetime.datetime.fromisoformat(d) for d in date_range]
        return start, end

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
        return datetime.datetime.fromisoformat(isodate)

    def numpydate_to_py(self, numpy_date: np.datetime64) -> datetime.datetime:
        """
        Convert a numpy datetime object to a python standard library datetime object

        Parameters
        ----------
        np.datetime64
            A numpy.datetime64 object to be converted

        Returns
        -------
        datetime.datetime
            A datetime.datetime object

        """
        return pd.Timestamp(numpy_date).to_pydatetime()

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

    def get_date_range_from_dataset(
        self, dataset: xr.Dataset
    ) -> tuple[datetime.datetime, datetime.datetime]:
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
        # if there is only one date, set start and end to the same value
        if dataset["time"].size == 1:
            value = dataset["time"].values
            if isinstance(value, np.ndarray):
                value = value[0]
            start = self.numpydate_to_py(value)
            end = start
        else:
            start = self.numpydate_to_py(dataset["time"][0].values)
            end = self.numpydate_to_py(dataset["time"][-1].values)
        return start, end

    def get_date_range_from_file(
        self, path: str, backend_kwargs: dict = None
    ) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Open file and return the start and end date of the data. The dimension name used to store dates should be passed as `dimension`.

        Parameters
        ----------
        path : str
            Path to the input dataset file on disk
        backend_kwargs : dict, optional
            Backend arguments for the xr.open_dataset() method

        Returns
        -------
        tuple
            A tuple of datetime.datetime objects defining the start and end date of a file's time dimension

        """
        dataset = xr.open_dataset(path, backend_kwargs=backend_kwargs)
        date_range = self.get_date_range_from_dataset(dataset)
        dataset.close()
        del dataset
        gc.collect()
        return date_range

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

    def get_newest_file_date_range(self) -> datetime.datetime:
        """
        Return the date range of the newest local file

        Returns
        -------
        datetime.datetime
            The start and end date of the newest local file

        """
        return self.get_date_range_from_file(list(self.input_files())[-1])

    # STRING TRANSFORMATIONS

    def _coord_reformat(self, *args, pretty: bool = False, padding: int = 0) -> str:
        """
        Return coordinates (individually or pair) as a single string value with 3 decimal places of precision. With `pretty` set
        to False, return a string that can be used for a file name. With `pretty` set to True, return formatted coordinate string

        Parameters
        ----------
        args : list
            A list of (index, coordinate) tuples
        pretty : bool, optional
            A switch indicating whether to add a separator to the returned coordinates
        padding : int
            The number of zero padding spaces, in integer form, to add to returned coordinates

        Returns
        -------
        str
            Coordinates reformatted as specified

        """
        if not pretty:
            separator = "_"
            coords = ""
        else:
            separator = ", "
            coords = "("
        for ii, coord in enumerate(args):
            if ii > 0:
                coords += separator
            coords += f"{float(coord):0{padding}.3f}"
        if pretty:
            coords += ")"
        return coords

    def coord_str(self, *args, pretty: bool = False, padding=0) -> str:
        """
        Return coordinates (individually or pair) as a single string value with 3 decimal places of precision. With `pretty` set
        to False, return a string that can be used for a file name. With `pretty` set to True, return formatted coordinate string


        Parameters
        ----------
        args : list
            A list of (index, coordinate) tuples
        pretty : bool, optional
            A switch indicating whether to add a separator to the returned coordinates
        padding : int
            The number of zero padding spaces, in integer form, to add to returned coordinates

        Returns
        -------
        str
            Coordinates reformatted as specified

        """
        translated_args = []
        for coord in args:
            if isinstance(coord, xr.DataArray):
                translated_args.append(coord.values)
            else:
                translated_args.append(coord)
        return self._coord_reformat(*translated_args, pretty=pretty, padding=padding)

    # FTP

    def sync_ftp_files(
        self,
        server: str,
        directory_path: str,
        file_match_pattern: str,
        include_size_check: bool = False,
    ):
        """
        Connect to `server` (currently only supports anonymous login), change to `directory_path`, pull new and updated files
        that match `file_match_pattern` in that directory into `self.local_input_path()`. Store a list of newly downloaded
        files in a member variable.

        Parameters
        ----------
        server : str
            The URL of the FTP server to check
        directory_path: str
            The path to the directory holding the desired FTP files on the server
        file_match_pattern : str
            A regex string to match file names (in directory_path) against
        include_size_check : bool, optional
            Switch to check (or not) the size of files against a maximum. Defaults to False.

        """
        # Login to remote FTP server
        with ftplib.FTP(server) as ftp:
            self.info(
                "checking {}:{} for files that match {}".format(
                    server, directory_path, file_match_pattern
                )
            )
            ftp.login()
            ftp.cwd(directory_path)
            # Loop through directory listing
            for file_name in ftp.nlst():
                if re.match(file_match_pattern, file_name):
                    # path on our local filesystem
                    local_file_path = pathlib.Path(self.local_input_path()) / file_name
                    modification_timestamp = ftp.sendcmd("MDTM {}".format(file_name))[
                        4:
                    ].strip()
                    modification_time = datetime.datetime.strptime(
                        modification_timestamp, "%Y%m%d%H%M%S"
                    )
                    # Retrieve this file unless we find conditions not to
                    retrieve = True
                    # Compare to local file of same name
                    if local_file_path.exists():
                        local_file_attributes = os.stat(local_file_path)
                        local_file_mtime = datetime.datetime.fromtimestamp(
                            local_file_attributes.st_mtime
                        )
                        local_file_size = local_file_attributes.st_size
                        # Set to binary transfer mode
                        ftp.sendcmd("TYPE I")
                        remote_file_size = ftp.size(file_name)
                        if modification_time <= local_file_mtime and (
                            not include_size_check
                            or remote_file_size == local_file_size
                        ):
                            self.debug(
                                "local file {} does not need updating".format(
                                    local_file_path
                                )
                            )
                            retrieve = False
                        elif modification_time > local_file_mtime:
                            self.debug(
                                "file {} local modification time {} less than remote modification time {}".format(
                                    local_file_path,
                                    local_file_mtime.strftime("%Y/%m/%d"),
                                    modification_time.strftime("%Y/%m/%d"),
                                )
                            )
                        else:
                            self.debug(
                                "mismatch between local and remote size for file {}".format(
                                    local_file_path
                                )
                            )
                    else:
                        self.debug("new remote file found {}".format(file_name))
                    # Write this file locally
                    if retrieve:
                        self.new_files.append(self.local_input_path() / file_name)
                        self.info(
                            "downloading remote file {} to {}".format(
                                file_name, local_file_path
                            )
                        )
                        with open(local_file_path, "wb") as fp:
                            ftp.retrbinary("RETR {}".format(file_name), fp.write)

    # ETC

    def array_has_data(self, array: np.ndarray) -> bool | np.ndarray:
        """
        Convenience method to determine if an array has any data

        Parameters
        ----------
        array : np.array
            A numpy array to assess

        Returns
        -------
        bool | np.array
            Either a boolean indicating whether an array has data,
            or an array containing multiple booleans indicating which arrays have data

        """
        return not np.all(np.isnan(array))

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
        return (
            round(float(dataset.longitude.values.min()), self.bbox_rounding_value),
            round(float(dataset.latitude.values.min()), self.bbox_rounding_value),
            round(float(dataset.longitude.values.max()), self.bbox_rounding_value),
            round(float(dataset.latitude.values.max()), self.bbox_rounding_value),
        )

    def json_to_bytes(self, obj: dict) -> bytes:
        """
        Convert a JSON object to a file type object (bytes). Primarily used for passing STAC metadata to IPFS

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

    def check_if_new_data(self, end_date: datetime.datetime) -> bool:
        """
        Check if the downloaded data contains any new records relative to the existing dataset.
        Return a boolean indicating whether to proceed with a transform/parse based on the presence of new records.

        Returns
        =======
        bool
            An indication of whether to proceed with a parse (True) or not (False)
        """

        # check if newest file on our server has newer data
        try:
            newest_file_end_date = self.get_newest_file_date_range()[1]
        except IndexError as e:
            self.info(f"Date range operation failed due to absence of input files. Exiting script. Full error message: {e}")
            return False
        self.info(f"newest file ends at {newest_file_end_date}")
        if newest_file_end_date >= end_date:
            self.info(f"newest file has newer data than our end date {end_date}, triggering parse")
            return True
        else:
            self.info(f"newest file doesn't have data past our existing end date {end_date}")
            return False
