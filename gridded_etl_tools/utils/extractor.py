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
import s3fs

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

    def pool(self, batch: typing.Sequence, accept_any_success: bool = False) -> bool:
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
        accept_any_success
            If True, the function will return True if any of the jobs succeed,
            otherwise it will return True only if ALL jobs succeed.
            Defaults to False

        Returns
        -------
        bool
            True if any/all of the jobs succeeded (see `accept_any_success`), False otherwise.
        """
        if not batch:
            log.info("No jobs submitted for downloading, exiting ETL.")
            return False
        with ThreadPool(self._concurrency_limit) as pool:
            results = pool.map(self._request_helper, batch)
            any_successful = any(results)
            all_successful = all(results)

        if accept_any_success and any_successful:
            log.warning(
                "Some requests succeeded, some did not. Check the logs for details of which. "
                "This dataset is flagged to proceed as long as any requests succeed."
            )
            return True
        elif all_successful:
            log.info("All requests succeeded.")
            return True
        else:
            log.info("All requests returned no data or failed. Extraction was not successful.")
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


class RetryingExtractor(Extractor):
    """
    Abstract base class for extractors that implement retry logic with configurable backoff.

    This class provides a template method pattern for retry logic, allowing subclasses to customize
    the extraction operation while sharing common retry behavior.

    Subclasses must implement:
    - perform_extraction(): The actual extraction operation
    - _get_failure_exception(): Returns the exception to raise on final failure

    Optionally override:
    - _calculate_retry_delay(): To customize backoff strategy (default is exponential)
    """

    retries: int
    backoff_factor: float
    ignorable_extraction_errors: tuple[type[Exception], ...]
    unsupported_extraction_errors: tuple[type[Exception], ...]

    def __init__(
        self,
        dm: dataset_manager.DatasetManager,
        concurrency_limit: int = 8,
        retries: int = 5,
        backoff_factor: float = 1.0,
        ignorable_extraction_errors: tuple[type[Exception], ...] = (),
        unsupported_extraction_errors: tuple[type[Exception], ...] = (),
    ):
        """
        Create a new RetryingExtractor object.

        Parameters
        ----------
        dm
            Source data for this dataset manager will be extracted
        concurrency_limit
            Number of simultaneous threads to run while requesting data
        retries
            Number of times to retry a failed extraction
        backoff_factor
            Base number of seconds used in retry delay calculation
        ignorable_extraction_errors
            A tuple of exception types that can be safely ignored during extraction.
            These will trigger skipping the current file and returning False.
        unsupported_extraction_errors
            A tuple of exception types that should trigger immediate failure of the extraction.
        """
        super().__init__(dm, concurrency_limit)
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.ignorable_extraction_errors: tuple[type[Exception], ...] = tuple(ignorable_extraction_errors)
        self.unsupported_extraction_errors: tuple[type[Exception], ...] = tuple(unsupported_extraction_errors)

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate the delay before the next retry attempt.

        Default implementation uses exponential backoff: backoff_factor * (2 ** (attempt - 1))

        Subclasses can override this for different strategies:
        - Linear: attempt * backoff_factor
        - Constant: backoff_factor

        Parameters
        ----------
        attempt
            The current attempt number (1-indexed)

        Returns
        -------
        float
            Number of seconds to wait before the next attempt
        """
        return self.backoff_factor * (2 ** (attempt - 1))

    @abstractmethod
    def perform_extraction(self, *args, **kwargs) -> bool:
        """
        Perform the actual extraction operation.

        This method is called by the retry loop and should perform a single
        extraction attempt. It should return True on success.

        Raises any exception on failure - the retry loop will catch and handle
        according to ignorable_extraction_errors and unsupported_extraction_errors.
        """

    @abstractmethod
    def _get_failure_exception(self, attempts: int, identifier: str) -> Exception:
        """
        Return the exception to raise when all retry attempts have been exhausted.

        Parameters
        ----------
        attempts
            The number of attempts made
        identifier
            A string identifying the resource that failed (e.g., URL or S3 path)

        Returns
        -------
        Exception
            The exception instance to raise
        """

    def retry_with_backoff(
        self,
        identifier: str,
        *args,
        retries: int | None = None,
        **kwargs,
    ) -> bool:
        """
        Execute the extraction operation with retry logic and backoff.

        This is the template method that orchestrates the retry loop.

        Parameters
        ----------
        identifier
            A string identifying the resource (used for logging)
        *args, **kwargs
            Arguments passed through to perform_extraction
        retries
            Override the instance's retry count for this call (optional)

        Returns
        -------
        bool
            True if extraction succeeded, False if an ignorable exception occurred

        Raises
        ------
        Exception
            The result of _get_failure_exception() if all retries exhausted,
            or any unsupported_extraction_errors immediately
        """
        max_retries = retries if retries is not None else self.retries

        counter = 1
        while counter <= max_retries:
            try:
                result = self.perform_extraction(*args, **kwargs)
                return result
            except self.ignorable_extraction_errors as e:
                log.info(f"Encountered permitted exception {e} for {identifier}, skipping")
                return False
            except self.unsupported_extraction_errors as e:
                log.info(f"Encountered unpermitted exception {e} for {identifier}, failing immediately")
                raise
            except Exception as e:
                retry_delay = self._calculate_retry_delay(counter)
                log.info(
                    f"Encountered retryable exception {e} for {identifier}, "
                    f"retrying after {retry_delay} seconds, attempt {counter}/{max_retries}"
                )
                counter += 1
                time.sleep(retry_delay)

        raise self._get_failure_exception(counter - 1, identifier)


class HTTPExtractor(RetryingExtractor):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # RetryError is raised by urllib3.Retry when HTTP status code retries are exhausted.
        # Don't retry these - they already went through their own retry logic.
        self.unsupported_extraction_errors = tuple(self.unsupported_extraction_errors) + (
            requests.exceptions.RetryError,
        )

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

        Raises
        ------
        RuntimeError
            If the object has not opened a session
        RetryError
            If the requests could not be completed after all retries
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

    def perform_extraction(self, remote_file_path: str, local_path: pathlib.Path) -> bool:
        """
        Perform the HTTP download operation.

        Parameters
        ----------
        remote_file_path
            URL to download from
        local_path
            Local path to write file to

        Returns
        -------
        bool
            True if successful
        """
        response = self.session.get(remote_file_path)
        response.raise_for_status()
        with open(local_path, "wb") as outfile:
            outfile.write(response.content)
        return True

    def _get_failure_exception(self, attempts: int, identifier: str) -> Exception:
        """Return RequestException for HTTP failures."""
        return requests.exceptions.RequestException(
            f"Too many ({attempts}) failed download attempts requesting {identifier}"
        )

    def request(self, remote_file_path: str, local_path: pathlib.Path | None = None) -> bool:
        """
        Request a file from an HTTP server and save it to disk, optionally at a given destination. If no destination is
        given, the file will be saved to the working directory with the same name it has on the server.

        If an existing directory is given as the destination path, the file will be written to the given directory with
        the same file name as on the server. If a destination path is given and is not a directory, the file will be
        written to the given path.

        An active session is required to make the request, so this must be called from within a context manager for this
        object.

        Parameters
        ----------
        remote_file_path
            URL to a file to be downloaded
        local_path
            Local path to write file to

        Returns
        -------
        bool
            True if the file was downloaded successfully, False if an ignorable exception occurred.

        Raises
        ------
        RuntimeError
            If this is not run from within a context manager
        RetryError
            If the requests could not be completed after all retries
        requests.exceptions.RequestException
            If the request fails after all retries or encounters an unsupported exception
        """
        if not hasattr(self, "session"):
            raise RuntimeError(
                "Extractor object does not have a session to run the request from. Create the extractor"
                " using the 'with' operator to create a session."
            )

        log.info(f"Downloading {remote_file_path}")

        # Build a dynamic path if a full destination path hasn't been given
        if local_path is None or local_path.is_dir():
            # Extract the file name from the end of the URL
            file_name = pathlib.Path(os.path.basename(urlparse(remote_file_path).path))

            if local_path is None:
                local_path = file_name
            else:
                local_path /= file_name

        return self.retry_with_backoff(
            identifier=remote_file_path,
            remote_file_path=remote_file_path,
            local_path=local_path,
        )


class S3ExtractorBase(RetryingExtractor):
    """
    Base class for S3 extraction operations.
    """

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Linear backoff: attempt * backoff_factor (default 30 seconds per attempt).

        Maintains backwards compatibility with existing S3 retry behavior.
        """
        return attempt * self.backoff_factor

    def _get_failure_exception(self, attempts: int, identifier: str) -> Exception:
        """Return FileNotFoundError for S3 failures."""
        return FileNotFoundError(f"Too many ({attempts}) failed download attempts from server requesting {identifier}")

    def request(
        self,
        remote_file_path: str,
        scan_indices: int | tuple[int, int] = 0,
        tries: int | None = None,
        local_file_path: pathlib.Path | None = None,
        informative_id: str | None = None,
    ) -> bool:
        """
        Extract/download a remote S3 file with retry logic and error handling.

        Parameters
        ----------
        remote_file_path
            An S3 file URL path to the file to be processed
        scan_indices
            Indices of the raw climate data to be read (used by kerchunk, ignored by download)
        tries
            Allow a number of failed requests before failing permanently. If None, uses instance's retries value.
        local_file_path
            An optional local file path
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
        identifier = informative_id if informative_id is not None else remote_file_path

        log.info(f"Beginning to download {identifier}")

        return self.retry_with_backoff(
            identifier=identifier,
            retries=tries,
            remote_file_path=remote_file_path,
            scan_indices=scan_indices,
            local_file_path=local_file_path,
        )


class S3ExtractorKerchunk(S3ExtractorBase):
    """
    Create an object that can be used to request remote kerchunking of S3 files in parallel. The kerchunked files will
    be added to the given `DatasetManager`'s list of Zarr JSONs at `DatasetManager.zarr_jsons`.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dm.zarr_jsons = []

    def perform_extraction(
        self,
        remote_file_path: str,
        scan_indices: int | tuple[int, int] = 0,
        local_file_path: pathlib.Path | None = None,
    ) -> bool:
        """
        Extract a remote S3 climate file into a JSON using kerchunk.

        Parameters
        ----------
        remote_file_path
            An S3 file URL path to the climate file to be transformed
        scan_indices
            Indices of the raw climate data to be read
        local_file_path
            An optional local file path to save the kerchunked Zarr JSON to
        """
        self.dm.kerchunkify(file_path=remote_file_path, scan_indices=scan_indices, local_file_path=local_file_path)
        return True


class S3ExtractorDownload(S3ExtractorBase):
    """
    Create an object that can be used to download S3 files directly using s3fs.
    """

    def __init__(
        self,
        *args,
        fs: s3fs.S3FileSystem | None = None,
        **kwargs,
    ):
        # ValueError can be raised by perform_extraction and should not retry
        super().__init__(*args, **kwargs)
        self.unsupported_extraction_errors = tuple(self.unsupported_extraction_errors) + (ValueError,)
        if fs:
            self.s3_fs = fs
        else:
            self.s3_fs = s3fs.S3FileSystem()

    def perform_extraction(
        self,
        remote_file_path: str,
        scan_indices: int | tuple[int, int] = 0,
        local_file_path: pathlib.Path | None = None,
    ) -> True:
        """
        Download S3 file directly to local path using s3fs.

        Parameters
        ----------
        remote_file_path
            An S3 file URL path to download
        scan_indices
            Ignored for direct downloads
        local_file_path
            Local file path where file should be saved

        Raises
        ------
        ValueError
            If local_file_path is None (required for downloads)
        ValueError
            If scan_indices is not 0 (unsupported for downloads)
        """
        if local_file_path is None:
            raise ValueError(f"local_file_path is required for {self.__class__.__name__}")

        if scan_indices != 0:
            raise ValueError(f"scan_indices not supported for {self.__class__.__name__}")

        # Ensure local directory exists
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the file
        self.s3_fs.download(remote_file_path, str(local_file_path))
        return True


class S3Extractor(S3ExtractorKerchunk):
    """
    Deprecated: Use S3ExtractorKerchunk instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log.warning(
            "S3Extractor is deprecated. Use S3ExtractorKerchunk for kerchunking operations "
            "or S3ExtractorDownload for direct file downloads."
        )


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
