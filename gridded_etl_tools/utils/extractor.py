"""
Consolidation of functions useful during the extract step of an ETL cycle for a dataset manager.
"""

# The annotations dict and TYPE_CHECKING var are necessary for referencing types that aren't fully imported yet. See
# https://peps.python.org/pep-0563/
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma NO COVER
    from .. import dataset_manager

from abc import ABC, abstractmethod

from multiprocess.pool import ThreadPool
import pathlib
import typing
import ftplib
import re
import time
import logging


log = logging.getLogger("extraction_logs")


class Extractor(ABC):

    def __init__(self, dm: dataset_manager.DatasetManager, concurrency_limit: int = 8):
        self.dm = dm
        self._concurrency_limit = concurrency_limit

    def pool(self, batch: typing.Sequence[typing.Dict]) -> bool:
        """
        Executes a batch of jobs concurrently using asyncio.

        Args:
            batch (typing.Sequence[typing.Sequence]): A sequence of job arguments.

        Returns:
            bool: True if all of the jobs succeeded, False otherwise.
        """

        with ThreadPool(self._concurrency_limit) as pool:
            results = pool.map(self._request_helper, batch)
            all_successful = all(results)

        if all_successful:
            log.info("All requests succeeded.")
            return True
        else:
            log.info("One or more requests returned no data or failed.")
            return False

    def _request_helper(self, dict_arg: dict) -> bool:
        """
        Helper function to unpack the arguments for the request method.

        Args:
            dict_arg (dict): A dictionary of arguments to be passed to the request method.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        return self.request(**dict_arg)

    @abstractmethod
    def request(self, *args, **kwargs) -> bool:
        """
        Abstract method to be implemented by subclasses. This method should perform an extraction operation
        and return True if data is retrieved, False otherwise
        """


class HTTPExtractor(Extractor):

    def __init__(self, dm: dataset_manager.DatasetManager, concurrency_limit: int = 8):
        """
        Set the host parameter when initializing an HTTPExtractor object
        Initializes a session within the dataset manager if it hasn't yet been initialized,
        so that the `request` method works as intended

        Note that `get_session` is therefore a required method for any DatasetManager using
        the HTTPExtractor class.

        Parameters
        ----------
        host
            Address to connect to for source data
        """
        super().__init__(dm, concurrency_limit=concurrency_limit)
        if not hasattr(dm, "session"):
            dm.get_session()

    def request(self, remote_file_path: str, local_file_path: str) -> bool:
        """
        Request a file from an HTTP Server and save it to disk
        Requires an active session within the dataset manager
        """
        self.dm.info(f"Downloading {local_file_path}")
        fil_in_mem = self.dm.session.get(remote_file_path)
        with open(self.dm.local_input_path() / local_file_path, "wb") as outfile:
            outfile.write(fil_in_mem.content)
        return True


class S3Extractor(Extractor):
    """
    Create an object that can be used to request remote kerchunking of S3 files in parallel. The kerchunked files will
    be added to the given `DatasetManager`'s list of Zarr JSONs at `DatasetManager.zarr_jsons`.
    """

    def __init__(self, dm: dataset_manager.DatasetManager):
        """
        Create a new Extractor object by associating a Dataset Manager with it.

        Parameters
        ----------
        dm
            Source data for this dataset manager will be extracted
        """
        super().__init__(dm)

    def request(
        self,
        remote_file_path: str,
        scan_indices: int | tuple[int, int] = 0,
        tries: int = 5,
        local_file_path: pathlib.Path | None = None,
        informative_id: str | None = None,
    ) -> bool:
        """
        Transform a remote S3 climate file into a JSON file and add it to the given `DatasetManager` object's internal
        list of Zarr JSONs. The list can then be processed by `DatasetManager.create_zarr_json` to create a Zarr that
        can be opened remotely in `xarray`.

        Parameters
        ----------
        remote_file_path
            An S3 file URL path to the climate file to be transformed and added to `DatasetManager.zarr_jsons`
        scan_indices
            Indices of the raw climate data to be read. This is a quirk particular to certain GRIB files that package
            many datasets w/in one GRIB file. The index tells Kerchunk which to pull out and process.
        tries
            Allow a number of failed requests before failing permanently
        local_file_path
            An optional local file path to save the kerchunked Zarr JSON to
        informative_id
            A string to identify the request in logs. Defaults to just the given remote file path

        Returns
        -------
        bool
            Returns a boolean indicating the success of the operation, if successful. Errors out if not.

        Raises
        ------
        FileNotFoundError
            If the request fails more than the given amount of tries
        """
        if not remote_file_path.lower().startswith("s3://"):
            raise ValueError(f"Given path {remote_file_path} is not an S3 path")

        # Default to using the raw file name to identify the request in the log message
        if informative_id is None:
            informative_id = remote_file_path

        log.info(f"Beginning to download {informative_id}")

        # Count failed requests and fail with an exception if allowed amount of tries is exceeded
        counter = 1
        while counter <= tries:
            try:
                # Remote kerchunk the requested file
                self.dm.kerchunkify(
                    file_path=remote_file_path, scan_indices=scan_indices, local_file_path=local_file_path
                )
                log.info(f"Finished downloading {informative_id}")
                return True
            except Exception as e:
                # Increase delay time after each failure
                retry_delay = counter * 30
                log.info(
                    f"Encountered exception {e} for {informative_id}, retrying after {retry_delay} seconds"
                    f" , attempt {counter}"
                )
                counter += 1
                time.sleep(retry_delay)
        else:
            log.info(f"Couldn't find or download a remote file for {informative_id}")
            raise FileNotFoundError(f"Too many ({counter}) failed download attempts from server")


class FTPExtractor(Extractor):
    """
    Create an object that provides an interface to a climate data source's FTP server. The object is able to open its
    connection within a context manager, navigate to specific working directory, match files located in subdirectories,
    and fetch files to a given destination folder.

    Currently only anonymous FTP access is supported.
    """

    ftp: ftplib.FTP
    host: str

    def __init__(self, dm: dataset_manager.DatasetManager, host: str, concurrency_limit: int = 1):
        """
        Set the host parameter when initializing an FTPEXtractor object

        Parameters
        ----------
        host
            Address to connect to for source data
        """
        super().__init__(dm, concurrency_limit=concurrency_limit)
        self.host = host

    def __enter__(self) -> FTPExtractor:
        """
        Open a connection to the FTP server at this object's given source from within a context manager. When creating
        a context, pass the host address unless `FTPExtractor.host` was already set.

        Example
        -------
        my_extractor = FTPExtractor(my_dataset_manager)
        with my_extractor("ftp.cdc.noaa.gov") as extractor:
            # get source files

        Returns
        -------
        FTPConnection
            this object

        Raises
        ------
        ValueError
            If host parameter hasn't been set
        """
        log.info(f"Opening a connection to {self.host}")
        self.ftp = ftplib.FTP(self.host)
        self.ftp.login()
        return self

    def __exit__(self, *exception):
        """
        Close the connection with the FTP server at this object's given source. This will be called automatically when
        exiting the connection context manager.

        Parameters
        ----------
        *exception
            Exception information passed automatically by Python
        """
        self.ftp.close()
        log.info(f"Closed connection to {self.host}")

    @property
    def cwd(self) -> pathlib.PurePosixPath:
        """
        Returns
        -------
        pathlib.PurePosixPath
            The object's working directory on the FTP server

        Raises
        ------
        RuntimeError
            If the FTP connection is not open yet
        """
        try:
            return pathlib.PurePosixPath(self.ftp.pwd())
        except ftplib.error_perm:
            raise RuntimeError(
                "FTP connection must be opened from a context manager before getting the working directory."
            )

    @cwd.setter
    def cwd(self, path: pathlib.PurePosixPath):
        """
        Change working directory on the FTP server to the given path. The connection must already be opened using
        `FTPExtractor.__enter__`.

        Parameters
        ----------
        path
            Directory path to change to

        Raises
        ------
        RuntimeError
            If there is an error during changing directories. This can be caused by the directory not existing, or
            the connection being closed.
        """
        if not self.ftp.nlst(str(path)):
            raise RuntimeError(f'Could not find path "{path}" on FTP server.')
        try:
            self.ftp.cwd(str(path))
        except ftplib.error_perm:
            raise RuntimeError("Error changing directory. Is the FTP connection open?")

    def request(
        self, source: pathlib.PurePosixPath, destination: pathlib.PurePosixPath = pathlib.PurePosixPath()
    ) -> bool:
        """
        Download the given source path within the FTP server's current working directory to the given destination.

        If the destination is a full path or the destination doesn't exist yet, that will be the path used for the
        output. If the destination is an existing folder, the output path will use the source's file name with any
        subdirectories omitted. By default, the destination is the current working directory.

        Parameters
        ----------
        source
            Path to a file to retrieve within the FTP server's current working directory

        Returns
        -------
        bool
            True after file is successfully retrieved. This is used for compatibilty with `Extractor.request` and
            `Extractor.pool`.

        Raises
        ------
        RuntimeError
            If an error occurs during the FTP retrieval call
        """
        # Build a file name using the source name if an existing directory was given as the destination. Otherwise, use
        # the destination as the full path to the output file.
        if pathlib.Path(destination).is_dir():
            output = destination / source.name
        else:
            output = destination

        # Open the output file and write the contents of the remote file to it using the FTP library
        log.info(f"Downloading remote file {source} to {output}")
        with open(output, "wb") as fp:
            try:
                # need separate FTP for each download to take advantage of concurrency
                with ftplib.FTP(self.host) as download_ftp:
                    download_ftp.login()
                    download_ftp.cwd(str(self.cwd))
                    download_ftp.retrbinary(f"RETR {source}", fp.write)
            except ftplib.error_perm:
                raise RuntimeError(f"Error retrieving {source} from {self.host} in {self.cwd}")

        # If the exception wasn't raised, the file was downloaded successfully
        return True

    def find(self, pattern: str) -> typing.Iterator[pathlib.PurePosixPath]:
        """
        Create an generator over all files in the FTP server's current working directory matching the given regex
        pattern. The FTP connection must already be opened using `FTPExtractor.__enter__`.

        Parameters
        ----------
        pattern
            Regular expression pattern to match against the current working directory listing

        Yields
        ------
        pathlib.PurePosixPath
            The next file matched
        """
        for file_name in self.ftp.nlst():
            if re.search(pattern, file_name):
                yield pathlib.PurePosixPath(file_name)

    def batch_requests(self, pattern: str = ".*") -> list[pathlib.PurePosixPath]:
        """
        Get a list of paths in the current working directory to download, optionally matching a given pattern. If no
        pattern is given, match all files. The result of this function can be passed to `FTPExtractor.pool` along with
        `FTPExtractor.request` to process download requests in parallel.

        Parameters
        ----------
        pattern
            Optional pattern to filter files from the current working directory

        Returns
        -------
        list[pathlib.PurePosixPath]
            List of paths to files in the current working directory matching the pattern
        """
        return list(self.find(pattern))
