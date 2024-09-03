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

from multiprocessing.pool import ThreadPool
import pathlib
import typing
import ftplib
import re
import time
import logging
import requests
from requests.adapters import HTTPAdapter, Retry
from urllib.parse import urlparse, urljoin
import os
import collections

from bs4 import BeautifulSoup

log = logging.getLogger("extraction_logs")


class Extractor(ABC):

    def __init__(self, dm: dataset_manager.DatasetManager, concurrency_limit: int = 8):
        """
        Create an instance of `Extrator`. `Extractor` is an abstract base class, so this should not be called directly.
        Use a specific type of extractor below instead.

        Parameters
        ----------
        dm
            Source data for this dataset manager will be extracted
        concurrency_limit
            Number of simultaneous request threads to run
        """
        self.dm = dm
        self._concurrency_limit = concurrency_limit

    def pool(self, batch: typing.Sequence) -> bool:
        """
        Run the `Extractor.request` function multiple times in parallel using `ThreadPool`. Wait for all requests to
        complete, then return a boolean indicating whether all requests were successful or not.

        The batch is a sequence of arguments to be passed to `Extractor.request`. Each entry in the batch can be either
        a single argument, in which case it is passed on its own, or a sequence of arguments to be passed.

        - If a batch entry is an instance of `collections.abc.Mapping`, like `dict`, the resulting call to
        `Extractor.request` will be `Extractor.request(**entry)`.

        - If a batch entry is an instance of `list`, `tuple`, or `set`, the resulting call will be
        `Extractor.request(*entry)`.

        - Otherwise, the call will be `Extractor(entry)`.

        Parameters
        ----------
        batch
            A sequence of job arguments

        Returns
        -------
        bool
            True if all of the jobs succeeded, False otherwise.
        """
        if not batch:
            log.info("No jobs submitted for downloading, exiting ETL.")
            return False
        with ThreadPool(self._concurrency_limit) as pool:
            results = pool.map(self._request_helper, batch)
            all_successful = all(results)

        if all_successful:
            log.info("All requests succeeded.")
            return True
        else:
            log.info("One or more requests returned no data or failed.")
            return False

    def _request_helper(self, arg: typing.Any) -> bool:
        """
        Helper function to unpack the arguments for the request method.

        - A `collection.abc.Mapping` (dict for example) becomes `Extractor.request(**arg)`
        - An instance of `list`, `tuple`, or `set` becomes `Extractor.request(*arg)`
        - Anything else becomes `Extractor.request(arg)`

        Parameters
        ----------
        arg
            A single argument or list/dict of arguments to be passed to the request method

        Returns
        -------
        bool
            True if the request was successful, False otherwise.
        """
        print(arg)
        if isinstance(arg, collections.abc.Mapping):
            return self.request(**arg)
        elif isinstance(arg, list) or isinstance(arg, tuple) or isinstance(arg, set):
            return self.request(*arg)
        else:
            return self.request(arg)

    @abstractmethod
    def request(self, *args, **kwargs) -> bool:
        """
        Abstract method to be implemented by subclasses. This method should perform an extraction operation
        and return True if data is retrieved, False otherwise
        """


class HTTPExtractor(Extractor):
    """
    Request data from given URLs over HTTP from within a context manager. The context manager creates a session, from
    which all requests are made.

    On 500, 502, 503, and 504 failures, requests are automatically retried a given amount of times, by a given backoff
    factor (defaults to 10 retries with a 10s backoff factor).

    URLs can be scraped from a given URL using `HTTPExtractor.get_links`. The resulting list of URLs can then be passed
    to `HTTPExtractor.pool` for download using multiple threads.

    Example
    -------
    with HTTPExtractor(my_dataset_manager) as extractor:
        links = extractor.get_links("https://climate.data.gov/usa/rainfall", my_filter_function)
        extractor.pool(links) # download all links found from get_links in parallel
    """

    session: requests.Session
    backoff_factor: float
    retries: int

    def __init__(
        self,
        dm: dataset_manager.DatasetManager,
        concurrency_limit: int = 8,
        retries: int = 10,
        backoff_factor: float = 10.0,
    ):
        """
        Create a new HTTPExtractor object for a given dataset manager.

        Parameters
        ----------
        dm
            Source data for this dataset manager will be extracted
        concurrency_limit
            Number of simultaneous threads to run while requesting data
        retries
            Number of times to retry a failed URL
        backoff_factor
            Number of seconds to wait between each request. Scales by a factor of 2**n per failed request.
        """
        super().__init__(dm, concurrency_limit)
        self.retries = retries
        self.backoff_factor = backoff_factor

    def __enter__(self) -> HTTPExtractor:
        """
        Open a new HTTP requests session. All URL requests will be made within this session.

        Returns
        -------
        HTTPConnection
            this object
        """
        retry_strategy = Retry(
            total=self.retries, status_forcelist=[500, 502, 503, 504], backoff_factor=self.backoff_factor
        )
        self.session = requests.Session()
        self.session.mount(prefix="https://", adapter=HTTPAdapter(max_retries=retry_strategy))
        self.session.mount(prefix="http://", adapter=HTTPAdapter(max_retries=retry_strategy))
        log.info(f"Opened a new HTTP requests session with {self.retries} automatic retries per request")
        return self

    def __exit__(self, *exception):
        """
        Close the HTTP requests session. This will be called automatically when exiting the connection context manager.

        Parameters
        ----------
        *exception
            Exception information passed automatically by Python
        """
        self.session.close()
        log.info("Closed HTTP requests session")

    def get_links(
        self, url: str, filter_func: typing.Callable[[str], bool] | bool = True, timeout: int = 10
    ) -> set[str]:
        """
        Return a set of links parsed from an HTML page at a given url. Relative URLs are converted to absolute using
        `urllib.parse.urljoin`. Duplicates are removed, and a `set` type is returned.

        `filter_func` is used to filter the list of links. If `filter_func` is given, the href value of each parsed link
        is passed to the given function. If the function returns `True`, the link is kept, otherwise it is discarded.
        The function can be used to match against a regular expression using `re.compile`, for example, or it can be any
        arbitrary function. The BeautifulSoup documentation contains examples of filter functions
        https://beautiful-soup-4.readthedocs.io/en/latest/#a-function.

        An active session is required, so this must be called within a context manager.

        Parameters
        ----------
        url
            A URL to scrape links from.
        filter_func
            A function that accepts a link's href string and returns True if the link should be kept, or False otherwise
        timeout
            Number of seconds to wait for a response before a request fails

        Returns
        -------
        set[str]
            A list of links, in string format, from the specified URL.
        """
        if not hasattr(self, "session"):
            raise RuntimeError(
                "Extractor object does not have a session to run the request from. Create the extractor"
                " using the 'with' operator to create a session."
            )

        log.info(f"Getting links from {url}")

        # Get a response from a given url, and raise an exception on error
        response = self.session.get(url, timeout=10)
        response.raise_for_status()

        # Parse the returned HTML webpage with BeautifulSoup and build a list of all links on the page, filtered by a
        # function if one was given. Relative URLs are converted to absolute URLs.
        soup = BeautifulSoup(response.content, "html.parser")
        href_links = set(urljoin(url, link.get("href")) for link in soup.find_all("a", href=filter_func))

        return href_links

    def request(self, remote_file_path: str, destination_path: pathlib.Path | None = None) -> bool:
        """
        Request a file from an HTTP server and save it to disk, optionally at a given destination. If no destination is
        given, the file will be saved to `self.dm.local_input_path()` with the same name it has on the server.

        An active session is required to make the request, so this must be called from within a context manager for this
        object.

        Parameters
        ----------
        remote_file_path
            URL to a file to be downloaded
        destination_path
            Local path to write file to

        Raises
        ------
        RuntimeError
            If this is not run from within a context manager
        """
        if not hasattr(self, "session"):
            raise RuntimeError(
                "Extractor object does not have a session to run the request from. Create the extractor"
                " using the 'with' operator to create a session."
            )

        log.info(f"Downloading {remote_file_path}")

        if destination_path is None:
            # Extract the file name from the end of the URL
            destination_path = pathlib.Path(os.path.basename(urlparse(remote_file_path).path))

        # Open the remote file, and write it locally
        with open(self.dm.local_input_path() / destination_path, "wb") as outfile:
            outfile.write(self.session.get(remote_file_path).content)

        # If no exceptions were raised, the file was downloaded successfully
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
        self.dm.zarr_jsons = []

    def request(
        self,
        remote_file_path: str,
        scan_indices: int | tuple[int, int] = 0,
        tries: int = 5,
        local_file_path: pathlib.Path | None = None,
        informative_id: str | None = None,
    ) -> bool:
        """
        Extract a remote S3 climate file into a JSON and add it to the given `DatasetManager` object's internal
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
    Create an object that provides an interface to a climate data source's FTP server by passing the target FTP server's
    host address.

    Use a context manager to open the connection (@see FTPExtractor.__enter__). Once connected, the object is able to
    navigate to specific working directory, match files located in subdirectories, and fetch files to a given
    destination folder.

    Currently only anonymous FTP access is supported.
    """

    ftp: ftplib.FTP
    host: str

    def __init__(self, dm: dataset_manager.DatasetManager, host: str, concurrency_limit: int = 1):
        """
        Set the host parameter when initializing an FTPExtractor object

        Parameters
        ----------
        dm
            Source data for this dataset manager will be extracted
        host
            Address to connect to for source data
        concurrency_limit
            Number of simultaneous requests. If greater than 1, multiple connections will be opened because an FTP
            connection only supports synchronous requests.
        """
        super().__init__(dm, concurrency_limit=concurrency_limit)
        self.host = host

    def __enter__(self) -> FTPExtractor:
        """
        Open a connection to the FTP server at `FTPExtractor.host` from within a context manager.

        Example
        -------
        with FTPExtractor(my_dataset_manager, "ftp.cdc.noaa.gov") as extractor:
            # now connected to ftp.cdc.noaa.gov

        Returns
        -------
        FTPConnection
            this object
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
