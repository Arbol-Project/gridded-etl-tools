"""
Consolidation of functions useful during the extract step of an ETL cycle for a dataset manager.
"""

# The annotations dict and TYPE_CHECKING var are necessary for referencing types that aren't fully imported yet. See
# https://peps.python.org/pep-0563/
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma NO COVER
    from .. import dataset_manager

import pathlib
import typing
import ftplib
import re
import time
import multiprocessing


class Extractor:
    """
    Base class for common extraction functions. Associate this class with a `DatasetManager` object by passing the
    object at initialization.

    Extractor's functions are used to request source data, using multiprocessing when possible. `Extractor.pool` is
    used to launch the requests using a thread pool.

    If more specific types of extraction are available in a child class, consider using the child class instead. For
    example, if extracting from FTP, use FTPExtractor.
    """

    def __init__(self, dm: dataset_manager.DatasetManager):
        """
        Create a new Extractor object by associating a Dataset Manager with it.

        Parameters
        ----------
        dm
            Source data for this dataset manager will be extracted
        """
        self.dm = dm

    def pool(self, batch_processor: typing.Callable[..., bool], batch: typing.Sequence[typing.Sequence]) -> bool:
        """
        Launch a batch of requests simultaneously, wait for them all to complete, then return a boolean indicating
        whether any of the requests were successful.

        The batch processor is any generic callable that accepts the arguments provided in each entry of the batch.
        Each entry in the batch is a list of arguments to be forwarded to the batch processor. The batch is submitted
        to a thread pool which will call the batch processor on every entry of the batch, launching all the calls in
        parallel and waiting for them to complete.

        If any of the calls returns `True`, this will return `True`, otherwise it will return `False`.

        Parameters
        ----------
        batch_processor
            Function which will be called repeatedly, being passed a new entry in the batch each time it's called
        batch
            A 2D list of lists (or any sequential type) containing arguments to be forwarded to the given batch
            processor

        Returns
        -------
        bool
            `True` if any of the batch calls returns `True`, `False` otherwise
        """
        # Success remains false until the first successful request is completed
        success = False

        # Time the download
        start_downloading = time.time()

        # Thread count is one less than the amount of CPU available, unless there is only 1 CPU
        thread_count = max(1, multiprocessing.cpu_count() - 1)

        # run downloads
        if len(batch) > 0:
            self.dm.info(f"Submitting request for {len(batch)} datasets in parallel using {thread_count} threads.")

            # download in parallel
            with multiprocessing.pool.ThreadPool(processes=thread_count) as pool:
                for result in pool.starmap(batch_processor, batch):
                    if result:
                        success = True

            self.dm.info(f"Downloading took {(time.time() - start_downloading) / 60:.2f} minutes")
        else:
            self.dm.info("No requests prepared to be submitted, please check batch_processor is functioning properly.")

        # If every request returned False, log a message
        if not success:
            self.dm.info("No requests for data were successful.")

        return success


class S3Extractor(Extractor):
    """
    Create an object that can be used to request remote kerchunking of S3 files in parallel. The kerchunked files will
    be added to the given `DatasetManager`'s list of Zarr JSONs at `DatasetManager.zarr_jsons`.
    """

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

        self.dm.info(f"Beginning to download {informative_id}")

        # Count failed requests and fail with an exception if allowed amount of tries is exceeded
        counter = 1
        while counter <= tries:
            try:
                # Remote kerchunk the requested file
                self.dm.kerchunkify(
                    file_path=remote_file_path, scan_indices=scan_indices, local_file_path=local_file_path
                )
                self.dm.info(f"Finished downloading {informative_id}")
                return True
            except Exception as e:
                # Increase delay time after each failure
                retry_delay = counter * 30
                self.dm.info(
                    f"Encountered exception {e} for {informative_id}, retrying after {retry_delay} seconds"
                    f" , attempt {counter}"
                )
                counter += 1
                time.sleep(retry_delay)
        else:
            self.dm.info(f"Couldn't find or download a remote file for {informative_id}")
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

    # Used to limit retrieval to a single thread at a time
    semaphore: multiprocessing.synchronize.Semaphore = multiprocessing.Semaphore()

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
        if not hasattr(self, "host"):
            raise ValueError("FTPExtractor must have a host parameter to open connection")
        else:
            self.dm.info(f"Opening a connection to {self.host}")
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
        self.dm.info(f"Closed connection to {self.host}")

    def __call__(self, host: str):
        """
        When called like a function, set the host parameter. See `FTPExtractor.__enter__`.

        Parameters
        ----------
        host
            Address to connect to for source data
        """
        self.host = host
        return self

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
        self.dm.info(f"Downloading remote file {source} to {output}")
        with open(output, "wb") as fp:
            try:
                # Use a semaphore to limit the number of simultaneous downloads to 1 even in multithreaded
                # environments. This is either a requirement of ftplib or a common requirement of FTP servers.
                with self.semaphore:
                    self.dm.info(
                        "Using a single thread as a requirement of FTP even if multiple threads are available"
                    )
                    self.ftp.retrbinary(f"RETR {source}", fp.write)
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
