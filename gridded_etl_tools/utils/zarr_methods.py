import datetime
import multiprocessing
import time
import json
import re
import fsspec
import pprint
import dask
import pathlib
import glob
import os

import pandas as pd
import numpy as np
import xarray as xr

from typing import Optional
from tqdm import tqdm
from subprocess import Popen
from contextlib import nullcontext
from itertools import starmap, repeat, chain
from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.grib2 import scan_grib
from kerchunk.combine import MultiZarrToZarr
from dask.distributed import Client, LocalCluster

from .convenience import Convenience
from .metadata import Metadata
from .store import IPLD


class Transform(Convenience):
    """
    Base class for transforming a collection of downloaded input files in NetCDF4 Classic format into
    (sequentially) kerchunk JSONs, a MultiZarr Kerchunk JSON, and finally an Xarray Dataset based on that MultiZarr.
    """

    # KERCHUNKING

    def create_zarr_json(self, force_overwrite: bool = True, file_filter: str|None = None, outfile_path: str|None = None):
        """
        Convert list of local input files (MultiZarr) to a single JSON representing a "virtual" Zarr

        Read each file in the local input directory and create an in-memory JSON object representing it as a Zarr,
        then read that collection of JSONs (MultiZarr) into one master JSON formatted as a Zarr and hence openable as a single file

        Note that MultiZarrToZarr will fail if chunk sizes are inconsistent due to inconsistently sized data inputs (e.g. different
        numbers of steps in input datasets)

        Parameters
        ----------
        force_overwrite : bool, optional
            Switch to use (or not) an existing MultiZarr JSON at `DatasetManager.zarr_json_path()`.
            Defaults to ovewriting any existing JSON under the assumption new data has been found.
        file_filter
            A string used to further filter down input files for kerchunkifying.
            Useful if you want to kerchunkify only a subset of available files.
            Defaults to None.
        outfile_path
            A custom string path for the final, merged Zarr JSON.
            Defaults to None, in which case Zarr JSONs are output to self.zarr_json_path.
        """
        self.zarr_json_path().parent.mkdir(mode=0o755, exist_ok=True)
        # Generate a multizarr if it doesn't exist. If one exists, overwrite it unless directed otherwise.
        if not self.zarr_json_path().exists() or force_overwrite:
            start_kerchunking = time.time()
            # Prepapre a list of zarr_jsons and feed that to MultiZarrtoZarr
            if not hasattr(self, "zarr_jsons"):
                input_files_list = [
                    str(fil)
                    for fil in self.input_files()
                    if any(fil.suffix in file_ext for file_ext in [".nc", ".nc4", ".grib", ".grib1", ".grib2", "grb1", ".grb2"])
                ]
                # Further filter down which files are processsed using an optional file filter string or integer
                if file_filter:
                    input_files_list = [fil for fil in input_files_list if file_filter in str(fil)]
                # Now prepare the MultiZarr
                self.info(f"Generating Zarr JSON for {len(input_files_list)} files with {multiprocessing.cpu_count()} processors")
                self.zarr_jsons = list(map(self.kerchunkify, tqdm(input_files_list)))
                mzz = MultiZarrToZarr(path=input_files_list, indicts=self.zarr_jsons, **self.mzz_opts())
            # if remotely extracting JSONs from S3, self.zarr_jsons should already be prepared during the `extract` step
            else:
                self.info(f"Generating Zarr JSON for {len(self.zarr_jsons)} files with {multiprocessing.cpu_count()} processors")
                mzz = MultiZarrToZarr(path=self.zarr_jsons, **self.mzz_opts())  # There are no file names to pass `path` if reading remotely
            # Translate the MultiZarr to a master JSON and save that out locally. Will fail if the input JSONs are misspecified.
            if not outfile_path:
                outfile_path = self.zarr_json_path()
            mzz.translate(filename=outfile_path)
            self.info(
                f"Kerchunking to Zarr JSON --- {round((time.time() - start_kerchunking)/60,2)} minutes"
            )
        else:
            self.info("Existing Zarr found, using that")

    def kerchunkify(self, file_path: str, scan_indices: int = 0, local_file_path: Optional[pathlib.Path] = None):
        """
        Transform input NetCDF or GRIB into a JSON representing it as a Zarr. These JSONs can be merged into a MultiZarr that Xarray can open natively as a Zarr.

        Read the input file either locally or remotely from S3, depending on whether an s3 bucket is specified in the file path.

        NOTE under the hood there are several versions of GRIB files -- GRIB1 and GRIB2 -- and NetCDF files -- classic, netCDF-4 classic, 64-bit offset, etc.
        Kerchunk will fail on some versions in undocumented ways. We have found consistent success with netCDF-4 classic files so test against using those.

        The command line tool `nccopy -k 'netCDF-4 classic model' infile.nc outfile.nc` can convert between formats

        Parameters
        ----------
        file_path : str
            A file path to an input GRIB or NetCDF-4 Classic file. Can be local or on a remote S3 bucket that accepts anonymous access.
        scan_indices : int, slice(int)
            One or many indices to filter the JSONS returned by `scan_grib` when scanning remotely.
            When multiple options are returned that usually means the provider prepares this data variable at multiple depth / surface layers.
            We currently default to the 1st (index=0), as we tend to use the shallowest depth / surface layer in ETLs we've written.
        local_file_path : Optional[str], optional
            An optional local file path to save the Kerchunked Zarr JSON to

        Returns
        -------
        scanned_zarr_json : dict
            A JSON representation of a local/remote NetCDF or GRIB file produced by Kerchunk and readable by Xarray as a lazy Dataset.
        """
        if not file_path.lower().startswith('s3://'):
            scanned_zarr_json = self.local_kerchunk(file_path, scan_indices)
        elif file_path.lower().startswith('s3://'):
            scanned_zarr_json = self.remote_kerchunk(file_path, scan_indices)
        # output individual JSONs for re-reading locally. This guards against crashes for long Extracts and speeds up dev. work.
        if self.use_local_zarr_jsons:
            if not local_file_path:
                raise NameError("Writing out local JSONS specified but no `local_file_path` variable was provided.")
            if isinstance(scanned_zarr_json, list):  # presumes lists are not nested more than one level deep
                memory_write_args = zip(scanned_zarr_json, repeat(local_file_path))
                list(starmap(self.zarr_json_in_memory_to_file, memory_write_args))
            else:
                self.zarr_json_in_memory_to_file(scanned_zarr_json, local_file_path)

        return scanned_zarr_json

    def local_kerchunk(self, file_path: str, scan_indices: int | tuple[int] = 0) -> dict:
        """
        Use Kerchunk to scan a file on the local file system

        Parameters
        ----------
        file_path : str
            A file path to an input GRIB or NetCDF-4 Classic file on a local file system
        scan_indices : int, slice(int)
            One or many indices to filter the JSONS returned by `scan_grib`
            When multiple options are returned that usually means the provider prepares this data variable at multiple depth / surface layers.
            We currently default to the 1st (index=0), as we tend to use the shallowest depth / surface layer in ETLs we've written.

        Returns
        -------
        scanned_zarr_json : dict
            A JSON representation of a NetCDF or GRIB file produced by Kerchunk and readable by Xarray as a lazy Dataset.
        """
        try:
            if self.file_type == 'NetCDF':
                fs = fsspec.filesystem("file")
                with fs.open(file_path) as infile:
                    scanned_zarr_json = SingleHdf5ToZarr(h5f=infile, url=file_path, inline_threshold=5000).translate()
            elif self.file_type == 'GRIB':
                scanned_zarr_json = scan_grib(url=file_path, filter = self.grib_filter, inline_threshold=20)[scan_indices]
        except OSError as e:
            raise ValueError(
                f"Error found with {file_path}, likely due to incomplete file. Full error message is {e}"
            )
        return scanned_zarr_json
    
    def remote_kerchunk(self, file_path: str, scan_indices: int | tuple[int] = 0) -> dict:
        """
        Use Kerchunk to scan a file on a remote S3 file system

        Parameters
        ----------
        file_path : str
            A file path to an input GRIB or NetCDF-4 Classic file on a remote S3 bucket that accepts anonymous access.
        scan_indices : int, slice(int)
            One or many indices to filter the JSONS returned by `scan_grib` when scanning remotely.
            When multiple options are returned that usually means the provider prepares this data variable at multiple depth / surface layers.
            We currently default to the 1st (index=0), as we tend to use the shallowest depth / surface layer in ETLs we've written.

        Returns
        -------
        scanned_zarr_json : dict
            A JSON representation of a NetCDF or GRIB file produced by Kerchunk and readable by Xarray as a lazy Dataset.
        """
        s3_so = {
            'anon': True,
            "default_cache_type": "readahead"
            }
        # Scan based on file type
        if self.file_type == 'NetCDF':
            with self.store.fs().open(file_path, **s3_so) as infile:
                scanned_zarr_json = SingleHdf5ToZarr(h5f=infile, url=file_path).translate()
        elif 'GRIB' in self.file_type:
            scanned_zarr_json = scan_grib(url=file_path, storage_options=s3_so, filter=self.grib_filter, inline_threshold=20)[scan_indices]
        # append/extend to self.zarr_jsons for later use in an ETL's `transform` step
        if type(scanned_zarr_json) == dict:
            self.zarr_jsons.append(scanned_zarr_json)
        elif type(scanned_zarr_json) == list:  # some remote scans will return a list of GRIBs
            self.zarr_jsons.extend(scanned_zarr_json)
        return scanned_zarr_json

    def zarr_json_in_memory_to_file(self, scanned_zarr_json: str, local_file_path: pathlib.Path):
        """
        Export a Kerchunked Zarr JSON to file. 
        If necessary, create a file name for that JSON in situ based on its attributes.

        Parameters
        ----------
        scanned_zarr_json
            The in-memory Zarr JSON returned by Kerchunk
        local_file_path
            The existing local file path specified by the user
        """
        local_file_path = self.file_path_from_zarr_json_attrs(scanned_zarr_json=scanned_zarr_json, local_file_path=local_file_path)
        with open(local_file_path, "w") as file:
            json.dump(scanned_zarr_json, file, sort_keys=False, indent=4)
            self.info(f"Wrote local JSON to {local_file_path}")

    def file_path_from_zarr_json_attrs(self, scanned_zarr_json: dict, local_file_path: pathlib.Path) -> pathlib.Path:
        """
        Create a local file path based on attributes of the input Zarr JSON. 
        Necessary for some datasets that package many forecasts into one single extract, preventing
        us from passing in a local file path for each forecasts

        Defaults to returning the local_file_path, i.e. doing nothing. 
        Implement any code creating a new file path within child implementations of this method.

        Parameters
        ----------
        scanned_zarr_json
            The in-memory Zarr JSON returned by Kerchunk
        local_file_path
            The existing local file path specified by the user
        
        Returns
        -------
        str
            The existing local file path specified by the user
        """
        return local_file_path

    @classmethod
    def mzz_opts(cls) -> dict:
        """
        Class method to populate with options to be passed to MultiZarrToZarr.
        The options dict is by default populated with class variables instantiated above;
        optional additional parameters can be added as per the needs of the input dataset

        Returns
        -------
        opts : dict
            Kwargs for kerchunk's MultiZarrToZarr method
        """
        opts = dict(
            remote_protocol=cls.remote_protocol(),
            remote_options={'anon' : True},
            identical_dims=cls.identical_dims(),
            concat_dims=cls.concat_dims(),
            preprocess=cls.preprocess_kerchunk,
        )
        return opts

    # PRE AND POST PROCESSING

    @classmethod
    def preprocess_kerchunk(cls, refs: dict) -> dict:
        """
        Class method to populate with the specific preprocessing routine of each child class (if relevant), whilst the file is being read by Kerchunk.
        Note this function usually works by manipulating Kerchunk's internal "refs" -- the zarr dictionary generated by Kerchunk.

        If no preprocessing is happening, return the dataset untouched

        Parameters
        ----------
        refs : dict
            Dataset attributes and information automatically supplied by Kerchunk

        Returns
        -------
        refs : dict
            Dataset attributes and information, transformed as needed
        """
        ref_names = set()
        file_match_pattern = "(.*?)/"
        for ref in refs:
            if re.match(file_match_pattern, ref) is not None:
                ref_names.add(re.match(file_match_pattern, ref).group(1))
        for ref in ref_names:
            fill_value_fix = json.loads(refs[f"{ref}/.zarray"])
            fill_value_fix["fill_value"] = str(cls.missing_value_indicator())
            refs[f"{ref}/.zarray"] = json.dumps(fill_value_fix)
        return refs

    # CONVERT FILES

    def parallel_subprocess_files(
            self,
            input_files: list[pathlib.Path],
            command_text: list[str],
            replacement_suffix: str,
            keep_originals: bool = False,
            invert_file_order: bool = False
    ):
        """
        Run a command line operation on a set of input files. In most cases, replace each file with an alternative file.

        Optionally, keep the original files for development and testing purposes.

        Parameters
        ----------
        raw_files : list
            A list of pathlib.Path objects referencing the original files prior to processing
        command_text : list[str]
            A list of strings to reconstruct into a shell command
        replacement_suffix : str
            The desired extension of the file(s) created by the shell routine. Replaces the old extension.
        keep_originals : bool, optional
            An optional flag to preserve the original files for debugging purposes. Defaults to False.
        """
        # set up and run conversion subprocess on command line
        commands = []
        for existing_file in input_files:
            new_file = existing_file.with_suffix(replacement_suffix)
            if invert_file_order:
                filenames = [new_file, existing_file]
            else:
                filenames = [existing_file, new_file]
            commands.append(  # map will convert the file names to strings because some command line tools (e.g. gdal) don't like Pathlib objects
                    list(map(str, command_text + filenames))
                )
        # Convert each comment to a Popen call b/c Popen doesn't block, hence processes will run in parallel
        # Only run 100 processes at a time to prevent BlockingIOErrors
        for index in range(0, len(commands), 100):
            commands_slice = [ Popen(cmd) for cmd in commands[index:index+100]]
            for command in commands_slice:
                command.wait()
                if not keep_originals:
                    if not invert_file_order:
                        os.remove(command.args[-2])
                    else:
                        os.remove(command.args[-1])
        self.info(
            f"{(len(list(input_files)))} conversions finished, cleaning up original files"
        )
        # Get rid of original files that were converted
        if keep_originals:
            self.archive_original_files(input_files)
        self.info(
            f"Cleanup finished"
        )

    def convert_to_lowest_common_time_denom(
        self, raw_files: list, keep_originals: bool = False
    ):
        """
        Decompose a set of raw files aggregated by week, month, year, or other irregular time denominator
        into a set of smaller files, one per the lowest common time denominator -- hour, day, etc.

        Parameters
        ----------
        raw_files : list
            A list of file path strings referencing the original files prior to processing
        originals_dir : pathlib.Path
            A path to a directory to hold the original files
        keep_originals : bool, optional
            An optional flag to preserve the original files for debugging purposes. Defaults to False.
        """
        if len(list(raw_files)) == 0:
            raise FileNotFoundError("No files found to convert, exiting script")
        command_text = ["cdo", "-f", "nc4", "splitsel,1"]
        self.parallel_subprocess_files(raw_files, command_text, '', keep_originals)

    def ncs_to_nc4s(self, keep_originals: bool = False):
        """
        Find all NetCDF files in the input folder and batch convert them
        in parallel to NetCDF4-Classic files that play nicely with Kerchunk

        NOTE There are many versions of NetCDFs and some others seem to play nicely with Kerchunk.
        NOTE To be safe we convert to NetCDF4 Classic as these are reliable and no data is lost.

        Parameters
        ----------
        keep_originals : bool
            An optional flag to preserve the original files for debugging purposes. Defaults to False.
        """
        # Build a list of files for manipulation
        raw_files = [pathlib.Path(file) for file in glob.glob(str(self.local_input_path() / "*.nc"))]
        if len(list(raw_files)) == 0:
            raise FileNotFoundError("No files found to convert, exiting script")
        # convert raw NetCDFs to NetCDF4-Classics in parallel
        self.info(
            f"Converting {(len(list(raw_files)))} NetCDFs to NetCDF4 Classic files"
        )
        command_text = ["nccopy", "-k", "netCDF-4 classic model"]
        self.parallel_subprocess_files(raw_files, command_text, '.nc4', keep_originals)

    def archive_original_files(self, files: list):
        """
        Move each original file to a "<dataset_name>_originals" folder for reference

        Parameters
        ----------
        files : list
            A list of original files to delete or save
        """
        # use the first file to define the originals_dir path
        first_file = files[0]
        originals_dir = first_file.parents[1] / (first_file.stem + "_originals")
        # move original files to the originals_dir
        for file in files:
            pathlib.Path.mkdir(originals_dir, mode=0o755, parents=True, exist_ok=True)
            file.rename(originals_dir / file.name)


class Publish(Transform, Metadata):
    """
    Base class for publishing methods -- both initial publication and updates
    """

    # PARSING

    def parse(self, *args, **kwargs) -> bool:
        """
        Open all raw files in self.local_input_path(). Transform the data contained in them into Zarr format and write to the store specified
        by `Attributes.store`.

        If the store is IPLD or S3, an existing Zarr will be searched for to be opened and appended to by default. This can be overridden to force
        writing the entire input data to a new Zarr by setting `Convenience.rebuild_requested` to `True`. If existing data is found,
        `DatasetManager.overwrite_allowed` must also be `True`.

        This is the core function for transforming and writing data (to disk, S3, or IPLD) and should be standard for all ETLs. Modify the
        child methods it calls or the dask configuration settings to resolve any performance or parsing issues.

        Parameters
        ----------
        args : list
            Additional arguments specified by the user
        kwargs : dict
            Keyword arguments specified by the user

        Returns
        -------
        bool
            Flag indicating if new data was / was not parsed
        """
        self.info("Running parse routine")
        # adjust default dask configuration parameters as needed and spin up a LocalCluster
        self.dask_configuration()
        with LocalCluster(
            processes=self.dask_use_process_scheduler,
            dashboard_address=self.dask_dashboard_address,  # specify local IP to prevent exposing the dashboard
            protocol=self.dask_scheduler_protocol,  # otherwise Dask may default to tcp or tls protocols and choke
            threads_per_worker=self.dask_num_threads,
            n_workers=self.dask_num_workers,
        ) as cluster:
            # IPLD objects can't pickle successfully in Dask distributed schedulers so we remove the distributed client in these cases
            with Client(cluster) if not isinstance(self.store, IPLD) else nullcontext() as client:
                self.info(f"Dask Dashboard for this parse can be found at {cluster.dashboard_link}")
                try:
                    # Attempt to find an existing Zarr, using the appropriate method for the store. If there is existing data and there is no
                    # rebuild requested, start an update. If there is no existing data, start an initial parse. If rebuild is requested and there is
                    # no existing data or allow overwrite has been set, write a new Zarr, overwriting (or in the case of IPLD, not using) any existing
                    # data. If rebuild is requested and there is existing data, but allow overwrite is not set, do not start parsing and issue a warning.
                    if self.store.has_existing and not self.rebuild_requested:
                        self.info(f"Updating existing data at {self.store}")
                        self.update_zarr()
                    elif not self.store.has_existing or (
                        self.rebuild_requested and self.overwrite_allowed
                    ):
                        if not self.store.has_existing:
                            self.info(
                                f"No existing data found. Creating new Zarr at {self.store}."
                            )
                        else:
                            self.info(f"Data at {self.store} will be replaced.")
                        self.write_initial_zarr()
                    else:
                        raise RuntimeError(
                            "There is already a zarr at the specified path and a rebuild is requested, "
                            "but overwrites are not allowed."
                        )
                except KeyboardInterrupt:
                    self.info(
                        "CTRL-C Keyboard Interrupt detected, exiting Dask client before script terminates"
                    )

        if hasattr(self, "dataset_hash") and self.dataset_hash:
            self.info("Published dataset's IPFS hash is " + str(self.dataset_hash))

        return True

    def publish_metadata(self):
        """
        Publishes STAC metadata to the backing store
        """
        current_zarr = self.store.dataset()
        if not current_zarr:
            raise RuntimeError("Attempting to write STAC metadata, but no zarr written yet")

        if not hasattr(self, "metadata"):
            # This will occur when user is only updating metadata and has not parsed
            self.populate_metadata()
        if not hasattr(self, "time_dims"):
            # ditto above; in some cases metadata will be populated but not time_dims
            self.set_key_dims()

        # This will do nothing if catalog already exists
        self.create_root_stac_catalog()

        # This will update the stac collection if it already exists
        self.create_stac_collection(current_zarr)

        # Create and publish metadata as a STAC Item
        self.create_stac_item(current_zarr)

    def to_zarr(self, dataset: xr.Dataset, *args, **kwargs):
        """
        Wrapper around `xr.Dataset.to_zarr`. `*args` and `**kwargs` are forwarded to `to_zarr`. The dataset to write to Zarr must be the first argument.

        On S3 and local, pre and post update metadata edits are saved to the Zarr attrs at `Dataset.update_in_progress` to indicate during writing that
        the data is being edited.

        The time dimension of the given dataset must be in contiguous order. The time step of the order will be determined by taking the difference
        between the first two time entries in the dataset, so the dataset must also have at least two time steps worth of data.

        Parameters
        ----------
        dataset
            Dataset to write to Zarr format
        *args
            Arguments to forward to `xr.Dataset.to_zarr`
        **kwargs
            Keyword arguments to forward to `xr.Dataset.to_zarr`
        """
        # Aggressively assert that the data is in contiguous time order. Only valid if the dataset's time dimension is longer than 1 element
        # This is to protect against data corruption, especially during insert and append operations, but it should probably be replaced with 
        # a more sophisticated set of flags that let the user decide how to handle time data at their own risk.
        times = dataset[self.time_dim].values
        if len(times) >= 2:
            # Check is only valid if we have 2 or more values with which to calculate the delta
            expected_delta = times[1] - times[0]
            if not self.are_times_in_expected_order(times=times, expected_delta=expected_delta):
                raise ValueError("Dataset does not contain contiguous time data")

        # Don't use update-in-progress metadata flag on IPLD
        if not isinstance(self.store, IPLD):
            # Create an empty dataset that will be used to just write the metadata (there's probably a better way to do this? compute=False or
            # zarr.consolidate_metadata?).
            dataset.attrs["update_in_progress"] = True
            empty_dataset = dataset
            for coord in chain(dataset.coords, dataset.data_vars):
                empty_dataset = empty_dataset.drop(coord)
            # If there is an existing Zarr, indicate in the metadata that an update is in progress, and write the metadata before starting the real write.
            # Note that update_is_append_only is also written here because it was set outside of to_zarr.
            if self.store.has_existing:
                self.info("Pre-writing metadata to indicate an update is in progress")
                empty_dataset.to_zarr(
                    self.store.mapper(refresh=True), append_dim=self.time_dim
                )

        # Write data to Zarr and log duration.
        start_writing = time.perf_counter()

        dataset.to_zarr(*args, **kwargs)
        self.info(
            f"Writing Zarr took {datetime.timedelta(seconds=time.perf_counter() - start_writing)}"
        )

        # Don't use update-in-progress metadata flag on IPLD
        if not isinstance(self.store, IPLD):
            # Indicate in metadata that update is complete.
            empty_dataset.attrs["update_in_progress"] = False
            self.info(
                "Re-writing Zarr to indicate in the metadata that update is no longer in process."
            )
            empty_dataset.to_zarr(self.store.mapper(), append_dim=self.time_dim)

    # SETUP

    def dask_configuration(self):
        """
        Convenience method to implement changes to the configuration of the dask client after instantiation

        NOTE Some relevant paramters and console print statements we found useful during testing have been left
        commented out at the bottom of this function. Consider activating them if you encounter trouble parsing
        """
        self.info("Configuring Dask")
        dask.config.set(
            {
                "distributed.scheduler.worker-saturation": self.dask_scheduler_worker_saturation
            }
        )  # toggle upwards or downwards (minimum 1.0) depending on memory mgmt performance
        dask.config.set(
            {"distributed.scheduler.worker-ttl": None}
        )  # will timeout on big tasks otherwise
        dask.config.set(
            {"distributed.worker.memory.target": self.dask_worker_mem_target}
        )
        dask.config.set({"distributed.worker.memory.spill": self.dask_worker_mem_spill})
        dask.config.set({"distributed.worker.memory.pause": self.dask_worker_mem_pause})
        dask.config.set(
            {"distributed.worker.memory.terminate": self.dask_worker_mem_terminate}
        )
        # IPLD should use the threads scheduler to work around pickling issues with IPLD objects like CIDs
        if isinstance(self.store, IPLD):
            dask.config.set({"scheduler" : "threads"})

        # OTHER USEFUL SETTINGS, USE IF ENCOUNTERING PROBLEMS WITH PARSES
        # dask.config.set({'scheduler' : 'threads'}) # default distributed scheduler does not allocate memory correctly for some parses
        # dask.config.set({'nanny.environ.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_' : 0}) # helps clear out unused memory
        # dask.config.set({"distributed.worker.memory.recent-to-old-time": "300s"}) #???

        # DEBUGGING
        self.info(f"dask.config.config is {pprint.pformat(dask.config.config)}")

    # PREPARATION

    def pre_initial_dataset(self) -> xr.Dataset:
        """
        Get an `xr.Dataset` that can be passed to the appropriate writing method when writing a new Zarr. Read the virtual Zarr JSON at the
        path returned by `Creation.zarr_json_path`, normalize the axes, re-chunk the dataset according to this object's chunking parameters, and
        add custom metadata defined by this class.

        Returns
        -------
        xr.Dataset
            The dataset from `Creation.zarr_json_to_dataset` with custom metadata, normalized axes, and rechunked
        """
        # Transform the JSON Zarr into an xarray Dataset
        dataset = self.transformed_dataset()

        # Reset standard_dims to Arbol's standard now that loading + preprocessing on the original names is done
        self.set_key_dims()
        dataset = dataset.transpose(*self.standard_dims)

        # Re-chunk
        self.info(f"Re-chunking dataset to {self.requested_dask_chunks}")
        dataset = dataset.chunk(self.requested_dask_chunks)
        self.info(f"Chunks after rechunk are {dataset.chunks}")

        # Add metadata to dataset
        dataset = self.set_zarr_metadata(dataset)

        # Log the state of the dataset before writing
        self.info(f"Initial dataset\n{dataset}")

        return dataset

    def transformed_dataset(self, custom: bool = False):
        """
        Overall method to return the fully processed and transformed dataset
        Defaults to returning zarr_json_to_datset but can be overridden to return a custom transformation instead
        """
        return self.zarr_json_to_dataset()

    def zarr_hash_to_dataset(self, ipfs_hash: str) -> xr.Dataset:
        """
        Open a Zarr on IPLD at `ipfs_hash` as an `xr.Dataset` object

        Parameters
        ----------
        ipfs_hash : str
            The CID of the dataset

        Returns
        -------
        dataset : xr.Dataset
            Object representing the dataset described by the CID at `self.latest_hash()`

        """
        mapper = self.store.mapper(set_root=False)
        mapper.set_root(ipfs_hash)
        dataset = xr.open_zarr(mapper)
        return dataset

    def zarr_json_to_dataset(self, zarr_json_path: str = None, decode_times: bool = True) -> xr.Dataset:
        """
        Open the virtual zarr at `self.zarr_json_path()` and return as a xr.Dataset object after applying
        any desired postprocessing steps

        Parameters
        ----------
        zarr_json_path : str, optional
            A path to a specific Zarr JSON prepared by Kerchunk. Primarily intended for debugging.
            Defaults to None, which will trigger using the `zarr_json_path` for the dataset in question.
        decode_times : bool, optional
            Choose whether to decode the times in inputs file using the CF conventions.
            In most cases this is desirable and necessary, therefore this defaults to True.

        Returns
        -------
        xr.Dataset
            Object representing the dataset described by the Zarr JSON file at `self.zarr_json_path()`

        """
        if not zarr_json_path:
            zarr_json_path = str(self.zarr_json_path())

        dataset = xr.open_dataset(
            "reference://",
            engine="zarr",
            chunks={},
            backend_kwargs={
                "storage_options": {
                    "fo": zarr_json_path,
                    "remote_protocol": self.remote_protocol(),
                    "skip_instance_cache": True,
                    "default_cache_type": "readahead",
                },
                "consolidated": False,
            },
            decode_times = decode_times
        )
        # Apply any further postprocessing on the way out
        return self.postprocess_zarr(dataset)

    def postprocess_zarr(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Method to populate with the specific postprocessing routine of each child class (if relevant)

        If no preprocessing is happening, return the dataset untouched

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being processed

        Returns
        -------
        dataset : xr.Dataset
            The dataset being processed

        """
        return dataset

    def set_key_dims(self):
        """
        Set the standard and time dimensions based on a dataset's type. Valid types are an ensemble dataset,
        a forecast (ensemble mean) dataset, or a "normal" observational dataset.

        The self.forecast and self.ensemble instance variables are set in the `init` of a dataset and default to False.
        """
        if not self.forecast and not self.ensemble:
            self.standard_dims = ["time", "latitude", "longitude"]
            self.time_dim = "time"
        elif self.hindcast:
            self.standard_dims = ["hindcast_reference_time", "forecast_reference_offset", "step", "ensemble", "latitude", "longitude"]
            self.time_dim = "hindcast_reference_time"
        elif self.ensemble:
            self.standard_dims = ["forecast_reference_time", "step", "ensemble", "latitude", "longitude"]
            self.time_dim = "forecast_reference_time"
        elif self.forecast:
            self.standard_dims = ["forecast_reference_time", "step", "latitude", "longitude"]
            self.time_dim = "forecast_reference_time"

    # INITIAL

    def write_initial_zarr(self):
        """
        Writes the first iteration of zarr for the dataset to the store specified at
        initialization. If the store is `IPLD`, does some additional metadata processing
        """
        # Transform the JSON Zar
        dataset = self.pre_initial_dataset()
        mapper = self.store.mapper(set_root=False)

        self.to_zarr(dataset, mapper, consolidated=True, mode="w")
        if isinstance(self.store, IPLD):
            self.dataset_hash = str(mapper.freeze())

    # UPDATES

    def update_zarr(self):
        """
        Update discrete regions of an N-D dataset saved to disk as a Zarr. If updates span multiple date ranges, pushes separate updates to each region.
        If the IPLD store is in use, after updating the dataset, this function updates the corresponding STAC Item and summaries in the parent
        STAC Collection.
        """
        original_dataset = self.store.dataset()
        update_dataset = self.transformed_dataset()

        # Reset standard_dims to Arbol's standard now that loading + preprocessing on the original names is done
        self.set_key_dims()
        self.info(f"Original dataset\n{original_dataset}")

        # Prepare inputs for the update operation
        insert_times, append_times = self.update_setup(original_dataset, update_dataset)

        # Conduct update operations
        self.update_parse_operations(
            original_dataset, update_dataset, insert_times, append_times
        )

    def update_setup(
        self, original_dataset: xr.Dataset, update_dataset: xr.Dataset
    ) -> tuple[list, list]:
        """
        Create needed inputs for the actual update parses: a variable to hold the hash and lists of any times to insert and/or append.

        Parameters
        ----------
        original_dataset : xr.Dataset
            The existing xr.Dataset
        update_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records

        Returns
        -------
        insert_times : list
            Datetimes corresponding to existing records to be replaced in the original dataset
        append_times : list
            Datetimes corresponding to all new records to append to the original dataset
        """
        original_times = set(original_dataset[self.time_dim].values)
        if (
            type(update_dataset[self.time_dim].values) == np.datetime64
        ):  # cannot perform iterative (set) operations on a single numpy.datetime64 value
            update_times = set([update_dataset[self.time_dim].values])
        else:  # many values will come as an iterable numpy.ndarray
            update_times = set(update_dataset[self.time_dim].values)
        insert_times = sorted(update_times.intersection(original_times))
        append_times = sorted(update_times - original_times)

        return insert_times, append_times

    def update_parse_operations(
        self,
        original_dataset: xr.Dataset,
        update_dataset: xr.Dataset,
        insert_times: tuple[datetime.datetime],
        append_times: tuple[datetime.datetime],
    ):
        """
        An enclosing method triggering insert and/or append operations based on the presence of valid records for either.

        Parameters
        ----------
        original_dataset : xr.Dataset
            The existing dataset
        update_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records
        insert_times : tuple
            Datetimes corresponding to existing records to be replaced in the original dataset
        append_times : tuple
            Datetimes corresponding to all new records to append to the original dataset
        """
        # First check that the data is not obviously wrong
        self.update_quality_check(original_dataset, insert_times, append_times)
        # Then write out updates to existing data using the 'region=' command...
        original_chunks = {
            dim: val_tuple[0] for dim, val_tuple in original_dataset.chunks.items()
        }
        if len(insert_times) > 0:
            if not self.overwrite_allowed:
                self.warn(
                    "Not inserting records despite historical data detected. 'allow_overwrite'"
                    "flag has not been set and store is not IPLD"
                )
            else:
                self.insert_into_dataset(
                    original_dataset, update_dataset, insert_times, original_chunks
                )
        else:
            self.info("No modified records to insert into original zarr")
        # ...then write new data (appends) using the 'append_dim=' command
        if len(append_times) > 0:
            self.append_to_dataset(update_dataset, append_times, original_chunks)
        else:
            self.info("No new records to append to original zarr")

    def insert_into_dataset(
        self,
        original_dataset: xr.Dataset,
        update_dataset: xr.Dataset,
        insert_times: list,
        original_chunks: list,
    ):
        """
        Insert new records to an existing dataset along its time dimension using the `append_dim=` flag.

        Parameters
        ----------
        original_dataset : xr.Dataset
            The existing xr.Dataset
        update_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records
        insert_times : list
            Datetimes corresponding to existing records to be replaced in the original dataset
        originaL_chunks : dict
            A Dict containing the dimension:size parameters for the original dataset
        """
        mapper = self.store.mapper()

        insert_dataset = self.prep_update_dataset(
            update_dataset, insert_times, original_chunks
        )
        date_ranges, regions = self.calculate_update_time_ranges(
            original_dataset, insert_dataset
        )
        for dates, region in zip(date_ranges, regions):
            insert_slice = insert_dataset.sel(**{self.time_dim : slice(dates[0], dates[1])})
            insert_dataset.attrs["update_is_append_only"] = False
            self.info("Indicating the dataset is not appending data only.")
            self.to_zarr(
                insert_slice.drop(self.standard_dims[1:]),
                mapper,
                region={self.time_dim: slice(region[0], region[1])},
            )

        self.info(
            f"Inserted records for {len(insert_dataset[self.time_dim].values)} times from {len(regions)} date range(s) to original zarr"
        )
        # In the case of IPLD, store the hash for later use
        if isinstance(self.store, IPLD):
            self.dataset_hash = str(mapper.freeze())

    def append_to_dataset(
        self, update_dataset: xr.Dataset, append_times: list, original_chunks: dict
    ):
        """
        Append new records to an existing dataset along its time dimension using the `append_dim=` flag.

        Parameters
        ----------
        update_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records
        append_times : list
            Datetimes corresponding to all new records to append to the original dataset
        originaL_chunks : dict
            The dimension:size parameters for the original dataset
        """
        append_dataset = self.prep_update_dataset(
            update_dataset, append_times, original_chunks
        )
        mapper = self.store.mapper()

        # Write the Zarr
        append_dataset.attrs["update_is_append_only"] = True
        self.info("Indicating the dataset is appending data only.")
        self.to_zarr(append_dataset, mapper, consolidated=True, append_dim=self.time_dim)

        self.info(
            f"Appended records for {len(append_dataset[self.time_dim].values)} datetimes to original zarr"
        )
        # In the case of IPLD, store the hash for later use
        if isinstance(self.store, IPLD):
            self.dataset_hash = str(mapper.freeze())

    def prep_update_dataset(
        self, update_dataset: xr.Dataset, time_filter_vals: list, new_chunks: dict
    ) -> xr.Dataset:
        """
        Select out and format time ranges you wish to insert or append into the original dataset based on specified time range(s) and chunks

        Parameters
        ----------
        update_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records
        time_filter_vals : list
            Datetimes corresponding to all new records to insert or append
        new_chunks : dict
            A Dict containing the dimension:size parameters for the original dataset

        Returns
        -------
        update_dataset : xr.Dataset
            An xr.Dataset filtered to only the time values in `time_filter_vals`, with correct metadata
        """
        # Xarray will automatically drop dimensions of size 1. A missing time dimension causes all manner of update failures.
        if self.time_dim in update_dataset.dims:
            update_dataset = update_dataset.sel(**{self.time_dim : time_filter_vals}).transpose(*self.standard_dims)
        else:
            update_dataset = update_dataset.expand_dims(self.time_dim).transpose(*self.standard_dims)
        update_dataset = update_dataset.chunk(new_chunks)
        update_dataset = self.set_zarr_metadata(update_dataset)

        self.info(f"Update dataset\n{update_dataset}")

        return update_dataset

    def calculate_update_time_ranges(
        self, original_dataset: xr.Dataset, update_dataset: xr.Dataset
    ) -> tuple[list[datetime.datetime], list[str]]:
        """
        Calculate the start/end dates and index values for contiguous time ranges of updates.
        Used by `update_zarr` to specify the location(s) of updates in a target Zarr dataset.

        Parameters
        ----------
        original_dataset : xr.Dataset
            The existing xr.Dataset
        update_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records

        Returns
        -------
        datetime_ranges : list[datetime.datetime, datetime.datetime]
            A List of (Datetime, Datetime) tuples defining the time ranges of records to insert
        regions_indices: list[int, int]
             A List of (int, int) tuples defining the indices of records to insert

        """
        dataset_time_span = f"1{self.temporal_resolution()[0]}"  # NOTE this won't work for months (returns 1 minute), we could define a more precise method with if/else statements if needed.
        complete_time_series = pd.Series(update_dataset[self.time_dim].values)
        # Define datetime range starts as anything with > 1 unit diff with the previous value,
        # and ends as > 1 unit diff with the following. First/Last will return NAs we must fill.
        starts = (complete_time_series - complete_time_series.shift(1)).abs().fillna(
            pd.Timedelta(dataset_time_span * 100)
        ) > pd.Timedelta(dataset_time_span)
        ends = (complete_time_series - complete_time_series.shift(-1)).abs().fillna(
            pd.Timedelta(dataset_time_span * 100)
        ) > pd.Timedelta(dataset_time_span)
        # Filter down the update time series to just the range starts/ends
        insert_datetimes = complete_time_series[starts + ends]
        single_datetime_inserts = complete_time_series[starts & ends]
        # Add single day insert datetimes once more so they can be represented as ranges, then sort for the correct order.
        # Divide the result into a collection of start/end range arrays
        insert_datetimes = np.sort(
            pd.concat(
                [insert_datetimes, single_datetime_inserts], ignore_index=True
            ).values
        )
        datetime_ranges = np.array_split(insert_datetimes, (len(insert_datetimes) / 2))
        # Calculate a tuple of the start/end indices for each datetime range
        regions_indices = []
        for date_pair in datetime_ranges:
            start_int = list(original_dataset[self.time_dim].values).index(\
                original_dataset.sel(**{self.time_dim : date_pair[0], 'method' : 'nearest'})[self.time_dim]
            )
            end_int = (
                list(original_dataset[self.time_dim].values).index(
                    original_dataset.sel(**{self.time_dim : date_pair[1], 'method' : 'nearest'})[self.time_dim]
                )
                + 1
            )
            regions_indices.append((start_int, end_int))

        return datetime_ranges, regions_indices

    # CHECKS

    def update_quality_check(self,
                             original_dataset: xr.Dataset,
                             insert_times: tuple[datetime.datetime],
                             append_times: tuple[datetime.datetime]
                             ):
        """
        Master function containing update-specific quality checks to run on a dataset prior to parsing
        If successful passes without comment. If unsuccessful raises a descriptive error message.

        Parameters
        ----------
        original_dataset : xr.Dataset
            The existing dataset
        insert_times : tuple
            Datetimes corresponding to existing records to be replaced in the original dataset
        append_times : tuple
            Datetimes corresponding to all new records to append to the original dataset
        """
        # Check that the first value of the append times and the last value of the original dataset are contiguous
        if append_times:
            original_append_bridge_times = [original_dataset[self.time_dim].values[-1], append_times[0]]
            expected_delta = original_dataset[self.time_dim][1] - original_dataset[self.time_dim][0]
            # Check these two values against the expected delta. All append times will be checked later in the stand
            if not self.are_times_in_expected_order(times=original_append_bridge_times, expected_delta=expected_delta):
                raise ValueError(
                "Append would create out of order or incomplete dataset, aborting"
            )
        # Raise an exception if there is no writable data
        if not any(insert_times) and not any(append_times):
            raise ValueError(
                "Update started with no new records to insert or append to original zarr."
            )

    def are_times_in_expected_order(self, times: tuple[datetime.datetime], expected_delta: np.timedelta64) -> bool:
        """
        Return false if a given iterable of times is out of order and/or does not follow the previous time,
        or falls outside of an acceptable range of timedeltas

        Parameters
        ----------
        times
            A datetime.datetime object representing the timestamp being checked
        expected_delta
            Amount of time expected to be between each time

        Returns
        -------
        bool
            Returns False for any unacceptable timestamp order, otherwise True
        """
        # Warn if not using expected delta
        if self.irregular_update_cadence():
            self.warn(f"Because dataset has irregular cadence {self.irregular_update_cadence} expected delta {expected_delta} is not being used for checking time contiguity")
        # Check if times meet expected_delta or fall within the anticipated range. 
        # Raise a warning and return false if so. Raise a descriptive error message in the enclosing function describing the specific operation this failed on.
        previous_time = times[0]
        for time in times[1:]:
            if self.irregular_update_cadence():
                if not self.irregular_update_cadence()[0] <= (time - previous_time) <= self.irregular_update_cadence()[1]:
                    self.warn(f"Time value {time} and previous time {previous_time} do not fit within anticipated update cadence {self.irregular_update_cadence()}")
                    return False
            elif time - previous_time != expected_delta:
                self.warn(f"Time value {time} and previous time {previous_time} do not fit within expected time delta {expected_delta}")
                return False
            previous_time = time
        # Return True if no problems found
        return True
