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
import s3fs
import zarr

from contextlib import nullcontext
from subprocess import Popen
from typing import Any, Union
from collections.abc import MutableMapping

import pandas as pd
import numpy as np
import xarray as xr

from dask.distributed import Client, LocalCluster
from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.grib2 import scan_grib
from kerchunk.combine import MultiZarrToZarr
from tqdm import tqdm

from .convenience import Convenience
from .metadata import Metadata
from .store import IPLD


class Transform(Convenience):
    """
    Base class for transforming a collection of downloaded input files in NetCDF4 Classic format into
    (sequentially) kerchunk JSONs, a MultiZarr Kerchunk JSON, and finally an Xarray Dataset based on that MultiZarr.
    """

    # KERCHUNKING

    def create_zarr_json(
        self, force_overwrite: bool = True, file_filters: list[str] | None = None, outfile_path: str | None = None
    ):
        """
        Convert list of local input files (MultiZarr) to a single JSON representing a "virtual" Zarr

        Read each file in the local input directory and create an in-memory JSON object representing it as a Zarr, then
        read that collection of JSONs (MultiZarr) into one master JSON formatted as a Zarr and hence openable as a
        single file

        Note that MultiZarrToZarr will fail if chunk sizes are inconsistent due to inconsistently sized data inputs
        (e.g. different numbers of steps in input datasets)

        Parameters
        ----------
        force_overwrite : bool, optional
            Switch to use (or not) an existing MultiZarr JSON at `DatasetManager.zarr_json_path()`.
            Defaults to ovewriting any existing JSON under the assumption new data has been found.
        file_filters
            A list of strings used to further filter down input files for kerchunkifying.
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
                    if any(
                        fil.suffix in file_ext
                        for file_ext in [".nc", ".nc4", ".grib", ".grib1", ".grib2", ".grb1", ".grb2"]
                    )
                ]
                # Further filter down which files are processsed using an optional file filter string or integer
                if file_filters:
                    input_files_list = [
                        fil for fil in input_files_list if any(file_filter in fil for file_filter in file_filters)
                    ]
                # Now prepare the MultiZarr
                self.info(
                    f"Generating Zarr JSON for {len(input_files_list)} files with {multiprocessing.cpu_count()} "
                    "processors"
                )
                self.zarr_jsons = list(map(self.kerchunkify, tqdm(input_files_list)))
                mzz = MultiZarrToZarr(path=input_files_list, indicts=self.zarr_jsons, **self.mzz_opts())
            # if remotely extracting JSONs from S3, self.zarr_jsons should already be prepared during the `extract`
            # step
            else:
                self.info(
                    f"Generating Zarr JSON for {len(self.zarr_jsons)} files with {multiprocessing.cpu_count()} "
                    "processors"
                )
                mzz = MultiZarrToZarr(
                    path=self.zarr_jsons, **self.mzz_opts()
                )  # There are no file names to pass `path` if reading remotely
            # Translate the MultiZarr to a master JSON and save that out locally. Will fail if the input JSONs are
            # misspecified.
            if not outfile_path:
                outfile_path = self.zarr_json_path()
            mzz.translate(filename=outfile_path)
            self.info(f"Kerchunking to Zarr JSON --- {round((time.time() - start_kerchunking)/60,2)} minutes")
        else:
            self.info("Existing Zarr found, using that")

    def kerchunkify(
        self,
        file_path: str,
        scan_indices: int = 0,
        local_file_path: pathlib.Path | None = None,
    ) -> dict:
        """
        Transform input NetCDF or GRIB into a JSON representing it as a Zarr. These JSONs can be merged into a
        MultiZarr that Xarray can open natively as a Zarr.

        Read the input file either locally or remotely from S3, depending on whether an s3 bucket is specified in the
        file path.

        NOTE under the hood there are several versions of GRIB files -- GRIB1 and GRIB2 -- and NetCDF files -- classic,
        netCDF-4 classic, 64-bit offset, etc. Kerchunk will fail on some versions in undocumented ways. We have found
        consistent success with netCDF-4 classic files so test against using those.

        The command line tool `nccopy -k 'netCDF-4 classic model' infile.nc outfile.nc` can convert between formats

        Parameters
        ----------
        file_path : str
            A file path to an input GRIB or NetCDF-4 Classic file. Can be local or on a remote S3 bucket that accepts
            anonymous access.
        scan_indices : int, slice(int)
            One or many indices to filter the JSONS returned by `scan_grib` when scanning remotely. When multiple
            options are returned that usually means the provider prepares this data variable at multiple depth /
            surface layers. We currently default to the 1st (index=0), as we tend to use the shallowest depth / surface
            layer in ETLs we've written.
        local_file_path : pathlib.Path  | None, optional
            An optional local file path to save the Kerchunked Zarr JSON to

        Returns
        -------
        scanned_zarr_json : dict
            A JSON representation of a local/remote NetCDF or GRIB file produced by Kerchunk and readable by Xarray as
            a lazy Dataset.
        """
        if file_path.lower().startswith("s3://"):
            scanned_zarr_json = self.remote_kerchunk(file_path, scan_indices)
        else:
            scanned_zarr_json = self.local_kerchunk(file_path, scan_indices)

        # output individual JSONs for re-reading locally. This guards against crashes for long Extracts and speeds up
        # dev. work.
        if self.use_local_zarr_jsons:
            if not local_file_path:
                if self.protocol == "file":
                    local_file_path = os.path.splitext(file_path)[0] + ".json"
                else:
                    raise ValueError(
                        "Writing out local JSONS specified but no `local_file_path` variable was provided."
                    )

            if isinstance(scanned_zarr_json, list):  # presumes lists are not nested more than one level deep
                for zarr_json in scanned_zarr_json:
                    self.zarr_json_in_memory_to_file(zarr_json, local_file_path)
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
            One or many indices to filter the JSONS returned by `scan_grib` When multiple options are returned that
            usually means the provider prepares this data variable at multiple depth / surface layers. We currently
            default to the 1st (index=0), as we tend to use the shallowest depth / surface layer in ETLs we've written.

        Returns
        -------
        scanned_zarr_json : dict
            A JSON representation of a NetCDF or GRIB file produced by Kerchunk and readable by Xarray as a lazy
            Dataset.
        """
        try:
            if self.file_type == "NetCDF":
                fs = fsspec.filesystem("file")
                with fs.open(file_path) as infile:
                    scanned_zarr_json = SingleHdf5ToZarr(h5f=infile, url=file_path, inline_threshold=5000).translate()

            elif self.file_type == "GRIB":
                scanned_zarr_json = scan_grib(url=file_path, filter=self.grib_filter, inline_threshold=20)[
                    scan_indices
                ]
            else:
                raise ValueError(f"Invalid value for file_type. Expected 'NetCDF' or 'GRIB', got {self.file_type}")
        except OSError as e:
            raise ValueError(f"Error found with {file_path}, likely due to incomplete file. Full error message is {e}")
        return scanned_zarr_json

    def remote_kerchunk(self, file_path: str, scan_indices: int | tuple[int] = 0) -> dict:
        """
        Use Kerchunk to scan a file on a remote S3 file system

        Parameters
        ----------
        file_path : str
            A file path to an input GRIB or NetCDF-4 Classic file on a remote S3 bucket that accepts anonymous access.
        scan_indices : int, slice(int)
            One or many indices to filter the JSONS returned by `scan_grib` when scanning remotely. When multiple
            options are returned that usually means the provider prepares this data variable at multiple depth /
            surface layers. We currently default to the 1st (index=0), as we tend to use the shallowest depth / surface
            layer in ETLs we've written.

        Returns
        -------
        scanned_zarr_json : dict
            A JSON representation of a NetCDF or GRIB file produced by Kerchunk and readable by Xarray as a lazy
            Dataset.
        """
        s3_so = {"anon": True, "default_cache_type": "readahead"}
        # Scan based on file type
        if self.file_type == "NetCDF":
            with s3fs.S3FileSystem().open(file_path, **s3_so) as infile:
                scanned_zarr_json = SingleHdf5ToZarr(h5f=infile, url=file_path).translate()

        elif "GRIB" in self.file_type:
            scanned_zarr_json = scan_grib(
                url=file_path, storage_options=s3_so, filter=self.grib_filter, inline_threshold=20
            )[scan_indices]

        else:
            raise ValueError(f"Expected NetCDF or GRIB, got {type(self.file_type)}")

        # TODO: Code smell -- this code assumes zarr_jsons is already present on the data manager, but it's not obvious
        # how that would come to pass. In this gridded_etl_tools the only time this attribute is assigned is in
        # create_zarr_json, which happens to be the method that calls the method that calls this method, so any results
        # of modifying zarr_jsons would presumably be obliterated at that point, anyway. In the meantime, the only way
        # for this code to run without error is for the manager code to intialize zarr_jsons, which is fairly obscure.

        # append/extend to self.zarr_jsons for later use in an ETL's `transform` step
        if isinstance(scanned_zarr_json, dict):
            self.zarr_jsons.append(scanned_zarr_json)

        elif isinstance(scanned_zarr_json, list):  # some remote scans will return a list of GRIBs
            self.zarr_jsons.extend(scanned_zarr_json)

        else:
            raise ValueError(f"Expected dict or list, got {type(scanned_zarr_json)}")

        return scanned_zarr_json

    def zarr_json_in_memory_to_file(self, scanned_zarr_json: dict, local_file_path: pathlib.Path):
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
        local_file_path = self.file_path_from_zarr_json_attrs(
            scanned_zarr_json=scanned_zarr_json, local_file_path=local_file_path
        )
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
            remote_protocol=cls.protocol,
            remote_options={"anon": True},
            identical_dims=cls.identical_dimensions,
            concat_dims=cls.concat_dimensions,
            preprocess=cls.preprocess_kerchunk,
            postprocess=cls.postprocess_kerchunk,
        )
        return opts

    # PRE AND POST PROCESSING

    @classmethod
    def preprocess_kerchunk(cls, refs: dict) -> dict:
        """
        Class method to populate with the specific preprocessing routine of each child class (if relevant), whilst the
        file is being read by Kerchunk. Note this function usually works by manipulating Kerchunk's internal "refs" --
        the zarr dictionary generated by Kerchunk.

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
            fill_value_fix["fill_value"] = cls.missing_value
            refs[f"{ref}/.zarray"] = json.dumps(fill_value_fix)
        return refs

    @classmethod
    def postprocess_kerchunk(
        cls, out_zarr: Union[zarr._storage.store.BaseStore, MutableMapping]
    ) -> Union[zarr._storage.store.BaseStore, MutableMapping]:
        """
        Class method to modify the in-memory Zarr created by Kerchunk for each file
        using Zarr methods. Useful where manipulating individual files via the reference dictionary in
        'preprocess_kerchunk' is either clumsy or impossible.

        Meant to be inherited and manipulated by child dataset managers as appropriate

        Parameters
        ----------
        out_zarr
            The Zarr returned by a kerchunk read of an individual Kerchunk JSON

        Returns
        -------
        out_zarr
            A modified version of the Zarr returned by a kerchunk read of an individual Kerchunk JSON
        """
        return out_zarr

    # CONVERT FILES

    def parallel_subprocess_files(
        self,
        input_files: list[pathlib.Path],
        command_text: list[str],
        replacement_suffix: str,
        keep_originals: bool = False,
        invert_file_order: bool = False,
    ):
        """
        Run a command line operation on a set of input files. In most cases, replace each file with an alternative
        file.

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
            # CDO specifies the extension via an environment variable, not an argument...
            if "cdo" in command_text:
                new_file = existing_file.with_suffix("")
                os.environ["CDO_FILE_SUFFIX"] = replacement_suffix
            # ...but other tools use arguments
            else:
                new_file = existing_file.with_suffix(replacement_suffix)
            if invert_file_order:
                filenames = [new_file, existing_file]
            else:
                filenames = [existing_file, new_file]
            # map will convert the file names to strings because some command line tools (e.g. gdal) don't like Pathlib
            # objects
            commands.append(list(map(str, command_text + filenames)))
        # CDO responds to an environment variable when assigning file suffixes
        # Convert each command to a Popen call b/c Popen doesn't block, hence processes will run in parallel
        # Only run 100 processes at a time to prevent BlockingIOErrors
        for index in range(0, len(commands), 100):
            commands_slice = [Popen(cmd) for cmd in commands[index : index + 100]]
            for command in commands_slice:
                command.wait()
                if not keep_originals:
                    if not invert_file_order:
                        os.remove(command.args[-2])
                    else:
                        os.remove(command.args[-1])

        self.info(f"{(len(input_files))} conversions finished, cleaning up original files")
        # Get rid of original files that were converted
        if keep_originals:
            self.archive_original_files(input_files)
        self.info("Cleanup finished")

    def convert_to_lowest_common_time_denom(self, raw_files: list, keep_originals: bool = False):
        """
        Decompose a set of raw files aggregated by week, month, year, or other irregular time denominator
        into a set of smaller files, one per the lowest common time denominator -- hour, day, etc.

        Converts to a NetCDF4 Classic file as this has shown consistently performance for parsing

        Parameters
        ----------
        raw_files : list
            A list of file path strings referencing the original files prior to processing
        originals_dir : pathlib.Path
            A path to a directory to hold the original files
        keep_originals : bool, optional
            An optional flag to preserve the original files for debugging purposes. Defaults to False.
        """
        if len(raw_files) == 0:
            raise ValueError("No files found to convert, exiting script")
        command_text = ["cdo", "-f", "nc4c", "splitsel,1"]
        self.parallel_subprocess_files(
            input_files=raw_files,
            command_text=command_text,
            replacement_suffix=".nc4",
            keep_originals=keep_originals,
        )

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
        # Build a list of files for manipulation, sorted so unit tests can have a consistent expected value
        raw_files = sorted([pathlib.Path(file) for file in glob.glob(str(self.local_input_path() / "*.nc"))])
        if len(raw_files) == 0:
            raise FileNotFoundError("No files found to convert, exiting script")
        # convert raw NetCDFs to NetCDF4-Classics in parallel
        self.info(f"Converting {(len(raw_files))} NetCDFs to NetCDF4 Classic files")
        command_text = ["nccopy", "-k", "netCDF-4 classic model"]
        self.parallel_subprocess_files(
            input_files=raw_files, command_text=command_text, replacement_suffix=".nc4", keep_originals=keep_originals
        )

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
        originals_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        for file in files:
            file.rename(originals_dir / file.name)


class Publish(Transform, Metadata):
    """
    Base class for publishing methods -- both initial publication and updates
    """

    # PARSING

    def parse(self) -> bool:
        """
        Open all raw files in self.local_input_path(). Transform the data contained in them into Zarr format and write
        to the store specified by `Attributes.store`.

        If the store is IPLD or S3, an existing Zarr will be searched for to be opened and appended to by default. This
        can be overridden to force writing the entire input data to a new Zarr by setting
        `Convenience.rebuild_requested` to `True`. If existing data is found, `DatasetManager.overwrite_allowed` must
        also be `True`.

        This is the core function for transforming and writing data (to disk, S3, or IPLD) and should be standard for
        all ETLs. Modify the child methods it calls or the dask configuration settings to resolve any performance or
        parsing issues.

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
            # IPLD objects can't pickle successfully in Dask distributed schedulers so we remove the distributed client
            # in these cases
            with Client(cluster) if not isinstance(self.store, IPLD) else nullcontext():
                self.info(f"Dask Dashboard for this parse can be found at {cluster.dashboard_link}")
                try:
                    # Attempt to find an existing Zarr, using the appropriate method for the store. If there is
                    # existing data and there is no rebuild requested, start an update. If there is no existing data,
                    # start an initial parse. If rebuild is requested and there is no existing data or allow overwrite
                    # has been set, write a new Zarr, overwriting (or in the case of IPLD, not using) any existing
                    # data. If rebuild is requested and there is existing data, but allow overwrite is not set, do not
                    # start parsing and issue a warning.
                    if self.store.has_existing and not self.rebuild_requested:
                        self.info(f"Updating existing data at {self.store}")
                        self.update_zarr()
                    elif not self.store.has_existing or (self.rebuild_requested and self.overwrite_allowed):
                        if not self.store.has_existing:
                            self.info(f"No existing data found. Creating new Zarr at {self.store}.")
                        else:
                            self.info(f"Data at {self.store} will be replaced.")
                        self.write_initial_zarr()
                    else:
                        raise RuntimeError(
                            "There is already a zarr at the specified path and a rebuild is requested, "
                            "but overwrites are not allowed."
                        )
                except KeyboardInterrupt:
                    self.info("CTRL-C Keyboard Interrupt detected, exiting Dask client before script terminates")

        if hasattr(self, "dataset_hash") and self.dataset_hash and not self.dry_run:
            self.info("Published dataset's IPFS hash is " + str(self.dataset_hash))

        self.info("Parse run successful")

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
        Wrapper around `xr.Dataset.to_zarr`. `*args` and `**kwargs` are forwarded to `to_zarr`. The dataset to write to
        Zarr must be the first argument.

        On S3 and local, pre and post update metadata edits are saved to the Zarr attrs at `Dataset.update_in_progress`
        to indicate during writing that the data is being edited.

        The time dimension of the given dataset must be in contiguous order. The time step of the order will be
        determined by taking the difference between the first two time entries in the dataset, so the dataset must also
        have at least two time steps worth of data.

        Parameters
        ----------
        dataset
            Dataset to write to Zarr format
        *args
            Arguments to forward to `xr.Dataset.to_zarr`
        **kwargs
            Keyword arguments to forward to `xr.Dataset.to_zarr`
        """
        # First check that the data makes sense
        self.pre_parse_quality_check(dataset)

        # Exit script if dry_run specified
        if self.dry_run:
            self.info("Exiting without parsing since the dataset manager was instantiated as a dry run")
            self.info(f"Dataset final state pre-parse:\n{dataset}")
        else:
            # Don't use update-in-progress metadata flag on IPLD or on a dataset that doesn't have existing data stored
            if not isinstance(self.store, IPLD):
                # Update metadata on disk with new values for update_in_progress and update_is_append_only, so that if
                # a Zarr is opened during writing, there will be indicators that show the data is being edited.
                self.info("Writing metadata before writing data to indicate write is in progress.")
                if self.store.has_existing:
                    self.store.write_metadata_only(
                        update_attrs={
                            "update_in_progress": True,
                            "update_is_append_only": dataset.get("update_is_append_only"),
                            "initial_parse": False,
                        }
                    )
                else:
                    dataset.attrs.update({"update_in_progress": True, "initial_parse": True})
                # Remove update attributes from the dataset putting them in a dictionary to be written post-parse
                dataset, post_parse_attrs = self.move_post_parse_attrs_to_dict(dataset=dataset)

            # Write data to Zarr and log duration.
            start_writing = time.perf_counter()
            dataset.to_zarr(*args, **kwargs)
            self.info(f"Writing Zarr took {datetime.timedelta(seconds=time.perf_counter() - start_writing)}")

            # Don't use update-in-progress metadata flag on IPLD
            if not isinstance(self.store, IPLD):
                # Indicate in metadata that update is complete.
                self.info("Writing metadata after writing data to indicate write is finished.")
                self.store.write_metadata_only(update_attrs=post_parse_attrs)

    def move_post_parse_attrs_to_dict(self, dataset: xr.Dataset) -> tuple[xr.Dataset, dict[str, Any]]:
        """
        Build a dictionary of attributes that should only be populated to a Zarr after parsing finishes
        While building this dict, remove these attributes from the dataset to be written.

        Parameters
        ----------
        dataset
            The xr.Dataset about to be written

        Returns
        -------
        dataset
            The xr.Dataset about to be written, with `self.update_attributes` keys removed from attributes
        update_attrs
            A dictionary of [str, Any] keypairs to be written to a Zarr only after a successful parse has finished
        """
        dataset = dataset.copy()
        update_attrs = {"update_in_progress": False, "initial_parse": False}
        # Build a dictionary of attributes to update post-parse
        for attr in self.update_attributes:
            if attr in dataset.attrs:
                # Remove update attribute fields from the dataset so they aren't written with the dataset
                # For example "date range" should only be updated after a successful parse
                update_attrs[attr] = dataset.attrs.pop(attr, None)

        return dataset, update_attrs

    # SETUP

    def dask_configuration(self):
        """
        Convenience method to implement changes to the configuration of the dask client after instantiation

        NOTE Some relevant paramters and console print statements we found useful during testing have been left
        commented out at the bottom of this function. Consider activating them if you encounter trouble parsing
        """
        self.info("Configuring Dask")
        dask.config.set(
            {"distributed.scheduler.worker-saturation": self.dask_scheduler_worker_saturation}
        )  # toggle upwards or downwards (minimum 1.0) depending on memory mgmt performance
        dask.config.set({"distributed.scheduler.worker-ttl": None})  # will timeout on big tasks otherwise
        dask.config.set({"distributed.worker.memory.target": self.dask_worker_mem_target})
        dask.config.set({"distributed.worker.memory.spill": self.dask_worker_mem_spill})
        dask.config.set({"distributed.worker.memory.pause": self.dask_worker_mem_pause})
        dask.config.set({"distributed.worker.memory.terminate": self.dask_worker_mem_terminate})
        # IPLD should use the threads scheduler to work around pickling issues with IPLD objects like CIDs
        if isinstance(self.store, IPLD):
            dask.config.set({"scheduler": "threads"})

        # OTHER USEFUL SETTINGS, USE IF ENCOUNTERING PROBLEMS WITH PARSES
        # default distributed scheduler does not allocate memory correctly for some parses
        # dask.config.set({'scheduler' : 'threads'})
        # helps clear out unused memory
        # dask.config.set({'nanny.environ.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_' : 0})
        # dask.config.set({"distributed.worker.memory.recent-to-old-time": "300s"}) #???

        # DEBUGGING
        self.info(f"dask.config.config is {pprint.pformat(dask.config.config)}")

    # PREPARATION

    def pre_initial_dataset(self) -> xr.Dataset:
        """
        Get an `xr.Dataset` that can be passed to the appropriate writing method when writing a new Zarr. Read the
        virtual Zarr JSON at the path returned by `Creation.zarr_json_path`, normalize the axes, re-chunk the dataset
        according to this object's chunking parameters, and add custom metadata defined by this class.

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

        # Add metadata to dataset
        dataset = self.set_zarr_metadata(dataset)

        # Re-chunk
        self.info(f"Re-chunking dataset to {self.requested_dask_chunks}")
        # store a version of the dataset that is not re-chunked for use in the pre-parse quality check
        # this is necessary for performance reasons (rechunking for every point comparison is slow)
        self.pre_chunk_dataset = dataset.copy()
        dataset = dataset.chunk(self.requested_dask_chunks)
        self.info(f"Chunks after rechunk are {dataset.chunks}")

        # Log the state of the dataset before writing
        self.info(f"Initial dataset\n{dataset}")

        return dataset

    def transformed_dataset(self):
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
                    "remote_protocol": self.protocol,
                    "skip_instance_cache": True,
                    "default_cache_type": "readahead",
                },
                "consolidated": False,
            },
            decode_times=decode_times,
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

        The self.dataset_category property defaults to "observation". If a dataset provides a different type of data,
         the property should be specific in that dataset's manager; otherwise the default value suffices.

        Raises
        ------
        ValueError
            Return a ValueError if `dataset_category` is misspecified
        """
        if self.dataset_category == "observation":
            self.standard_dims = ["time", "latitude", "longitude"]
            self.time_dim = "time"
        elif self.dataset_category == "hindcast":
            self.standard_dims = [
                "hindcast_reference_time",
                "forecast_reference_offset",
                "step",
                "ensemble",
                "latitude",
                "longitude",
            ]
            self.time_dim = "hindcast_reference_time"
        elif self.dataset_category == "ensemble":
            self.standard_dims = ["forecast_reference_time", "step", "ensemble", "latitude", "longitude"]
            self.time_dim = "forecast_reference_time"
        elif self.dataset_category == "forecast":
            self.standard_dims = ["forecast_reference_time", "step", "latitude", "longitude"]
            self.time_dim = "forecast_reference_time"
        else:
            raise ValueError(
                "Dataset is not correctly specified as an observation, forecast, ensemble, or hindcast, "
                "preventing the correct assignment of standard_dims and the time_dim. Please revise the "
                "dataset's ETL manager to correctly specify one of these properties."
            )

    # INITIAL

    def write_initial_zarr(self):
        """
        Writes the first iteration of zarr for the dataset to the store specified at initialization. If the store is
        `IPLD`, does some additional metadata processing
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
        Update discrete regions of an N-D dataset saved to disk as a Zarr. If updates span multiple date ranges, pushes
        separate updates to each region. If the IPLD store is in use, after updating the dataset, this function updates
        the corresponding STAC Item and summaries in the parent STAC Collection.
        """
        original_dataset = self.store.dataset()
        update_dataset = self.transformed_dataset()

        # Reset standard_dims to Arbol's standard now that loading + preprocessing on the original names is done
        self.set_key_dims()
        self.info(f"Original dataset\n{original_dataset}")

        # Prepare inputs for the update operation
        insert_times, append_times = self.update_setup(original_dataset, update_dataset)

        # Conduct update operations
        self.update_parse_operations(original_dataset, update_dataset, insert_times, append_times)

    def update_setup(self, original_dataset: xr.Dataset, update_dataset: xr.Dataset) -> tuple[list, list]:
        """
        Create needed inputs for the actual update parses: a variable to hold the hash and lists of any times to insert
        and/or append.

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
        # cannot perform iterative (set) operations on a single numpy.datetime64 value
        if type(update_dataset[self.time_dim].values) == np.datetime64:  # noqa: E721
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
        An enclosing method triggering insert and/or append operations based on the presence of valid records for
        either.

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
        original_chunks = {dim: val_tuple[0] for dim, val_tuple in original_dataset.chunks.items()}
        if len(insert_times) > 0:
            if not self.overwrite_allowed:
                self.warn(
                    "Not inserting records despite historical data detected. 'allow_overwrite'"
                    "flag has not been set and store is not IPLD"
                )
            else:
                self.insert_into_dataset(original_dataset, update_dataset, insert_times, original_chunks)
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

        insert_dataset = self.prep_update_dataset(update_dataset, insert_times, original_chunks)
        date_ranges, regions = self.calculate_update_time_ranges(original_dataset, insert_dataset)
        for dates, region in zip(date_ranges, regions):
            insert_slice = insert_dataset.sel(**{self.time_dim: slice(*dates)})
            insert_dataset.attrs["update_is_append_only"] = False
            self.info("Indicating the dataset is not appending data only.")
            self.to_zarr(
                insert_slice.drop(self.standard_dims[1:]),
                mapper,
                region={self.time_dim: slice(*region)},
            )

        if not self.dry_run:
            self.info(
                f"Inserted records for {len(insert_dataset[self.time_dim].values)} times from {len(regions)} date "
                "range(s) to original zarr"
            )
        # In the case of IPLD, store the hash for later use
        if isinstance(self.store, IPLD):
            self.dataset_hash = str(mapper.freeze())

    def append_to_dataset(self, update_dataset: xr.Dataset, append_times: list, original_chunks: dict):
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
        append_dataset = self.prep_update_dataset(update_dataset, append_times, original_chunks)
        mapper = self.store.mapper()

        # Write the Zarr
        append_dataset.attrs["update_is_append_only"] = True
        self.info("Indicating the dataset is appending data only.")

        self.to_zarr(append_dataset, mapper, consolidated=True, append_dim=self.time_dim)

        if not self.dry_run:
            self.info(f"Appended records for {len(append_dataset[self.time_dim].values)} datetimes to original zarr")
        # In the case of IPLD, store the hash for later use
        if isinstance(self.store, IPLD):
            self.dataset_hash = str(mapper.freeze())

    def prep_update_dataset(self, update_dataset: xr.Dataset, time_filter_vals: list, new_chunks: dict) -> xr.Dataset:
        """
        Select out and format time ranges you wish to insert or append into the original dataset based on specified
        time range(s) and chunks

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
        # Xarray will automatically drop dimensions of size 1. A missing time dimension causes all manner of update
        # failures.
        if self.time_dim in update_dataset.dims:
            update_dataset = update_dataset.sel(**{self.time_dim: time_filter_vals}).transpose(*self.standard_dims)
        else:
            update_dataset = update_dataset.expand_dims(self.time_dim).transpose(*self.standard_dims)

        # Add metadata to dataset
        update_dataset = self.set_zarr_metadata(update_dataset)
        # Rechunk, storing a non-rechunked version for pre-parse quality checks
        self.pre_chunk_dataset = update_dataset.copy()
        update_dataset = update_dataset.chunk(new_chunks)

        self.info(f"Update dataset\n{update_dataset}")
        return update_dataset

    def calculate_update_time_ranges(
        self, original_dataset: xr.Dataset, update_dataset: xr.Dataset
    ) -> tuple[list[datetime.datetime], list[str]]:
        """
        Calculate the start/end dates and index values for contiguous time ranges of updates.
        Used by `update_zarr` to specify the location(s) of updates in a target Zarr dataset.

        Algorithm given here due to complexity of function:
        1. Find the time coordinates of the beginnings and ends of contiguous sections of the update dataset
        2. Combine these coordinates into a single array such that
            start and end points of ranges of length 1 are repeated
        3. Create a list of tuples of length 2, where the entries represent the endpoints of these ranges
        4. Find the indices within the orginal dataset of these coordinates
        5. Return the results of 3 and 4

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
        # NOTE this won't work for months (returns 1 minute), we could define a more precise method with if/else
        # statements if needed.
        dataset_time_span = f"1{self.time_resolution[0]}"
        complete_time_series = pd.Series(update_dataset[self.time_dim].values)
        # Define datetime range starts as anything with > 1 unit diff with the previous value,
        # and ends as > 1 unit diff with the following. First/Last will return NAs we must fill.
        starts = (complete_time_series - complete_time_series.shift(1)).abs().fillna(pd.Timedelta.max) > pd.Timedelta(
            dataset_time_span
        )
        ends = (complete_time_series - complete_time_series.shift(-1)).abs().fillna(pd.Timedelta.max) > pd.Timedelta(
            dataset_time_span
        )
        # Filter down the update time series to just the range starts/ends
        insert_datetimes = complete_time_series[starts + ends]
        single_datetime_inserts = complete_time_series[starts & ends]
        # Add single day insert datetimes once more so they can be represented as ranges, then sort for the correct
        # order. Divide the result into a collection of start/end range arrays
        insert_datetimes = np.sort(pd.concat([insert_datetimes, single_datetime_inserts], ignore_index=True).values)
        datetime_ranges = np.array_split(insert_datetimes, (len(insert_datetimes) / 2))
        # Calculate a tuple of the start/end indices for each datetime range
        regions_indices = []
        for date_pair in datetime_ranges:
            start_int = list(original_dataset[self.time_dim].values).index(
                original_dataset.sel(**{self.time_dim: date_pair[0], "method": "nearest"})[self.time_dim]
            )
            end_int = (
                list(original_dataset[self.time_dim].values).index(
                    original_dataset.sel(**{self.time_dim: date_pair[1], "method": "nearest"})[self.time_dim]
                )
                + 1
            )
            regions_indices.append((start_int, end_int))

        return datetime_ranges, regions_indices

    # CHECKS

    def pre_parse_quality_check(self, dataset: xr.Dataset):
        """
        Function containing quality checks applicable to all datasets we parse, initial or update
        Intended to be run on a dataset prior to parsing.

        If successful passes without comment. If unsuccessful raises a descriptive error message.

        Parameters
        ----------
        dataset : xr.Dataset
            The final dataset to be parsed
        """
        self.info("Beginning pre-parse quality check of prepared dataset")
        start_checking = time.perf_counter()
        # TIME CHECK
        # Aggressively assert that the time dimension of the data is in the anticipated order.
        # Only valid if the dataset's time dimension is longer than 1 element
        # This is to protect against data corruption, especially during insert and append operations,
        # but it should probably be replaced with a more sophisticated set of flags
        # that let the user decide how to handle time data at their own risk.
        times = dataset[self.time_dim].values
        if len(times) >= 2:
            # Check is only valid if we have 2 or more values with which to calculate the delta
            expected_delta = times[1] - times[0]
            if not self.are_times_in_expected_order(times=times, expected_delta=expected_delta):
                raise IndexError("Dataset does not contain contiguous time data")

        # VALUES CHECK
        # Check 100 values for NAs and extreme values
        self.check_random_values(dataset.copy(), checks=100)

        # ENCODING CHECK
        # Check that data is stored in a space efficient format
        if not dataset[self.data_var()].encoding["dtype"] == self.data_var_dtype:
            raise TypeError(
                f"Dtype for data variable {self.data_var()} is "
                f"{dataset[self.data_var()].dtype} when it should be {self.data_var_dtype}"
            )
        self.info(f"Checking dataset took {datetime.timedelta(seconds=time.perf_counter() - start_checking)}")

    def check_random_values(self, dataset: xr.Dataset, checks: int = 100) -> dict[str, dict[str:Any]]:
        """
        Check N random values from the finalized dataset for any obviously wrong data points,
        either unanticipated NaNs or extreme values

        Returns
        -------
        random_values
            A dictionary of randomly selected values with their corresponding coordinates.
            Intended for later reuse checking the same coordinates after a dataset is parsed.
        """
        random_vals = {}
        # insert operations will create datasets w/ only time coordinates and index values for other coords
        # this will cause comparison w/ the `pre_chunk_dataset` below to fail as index values != actual vals
        # therefore we repopulate the original values to enable comparisons
        if len(dataset.coords) == 1:
            orig_coords = {
                coord: self.pre_chunk_dataset.coords[coord].values
                for coord in self.pre_chunk_dataset.drop(self.time_dim).coords
            }
            dataset = dataset.assign_coords(**orig_coords)
        for i in range(checks):
            random_coords = self.get_random_coords(dataset)
            random_val = self.pre_chunk_dataset[self.data_var()].sel(**random_coords).values
            # Check for unanticipated NaNs
            if np.isnan(random_val) and not self.has_nans:
                raise ValueError(f"NaN value found for random point at coordinates {random_coords}")
            # Check extreme values if they are defined
            if not np.isnan(random_val):
                unit = dataset[self.data_var()].encoding["units"]
                if unit in self.extreme_values_by_unit.keys():
                    limit_vals = self.extreme_values_by_unit[unit]
                    if not limit_vals[0] <= random_val <= limit_vals[1]:
                        raise ValueError(
                            f"Value {random_val} falls outside acceptable range "
                            f"{limit_vals} for data in units {unit}. Found at {random_coords}"
                        )
            # Build a dictionary of checked values to compare against after parsing
            random_vals.update({i: {"coords": random_coords, "value": random_val}})

        return random_vals

    def update_quality_check(
        self,
        original_dataset: xr.Dataset,
        insert_times: tuple[datetime.datetime],
        append_times: tuple[datetime.datetime],
    ):
        """
        Function containing quality checks specific to update parses, either insert or append.
        Intended to be run on an update dataset prior to parsing.

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
        # Check that the update data isn't before the start of the existing dataset
        if any(append_times):
            if append_times[0] < original_dataset[self.time_dim][0]:
                raise IndexError(
                    f"Attempting to append data at {insert_times[0]} "
                    f"before dataset start {original_dataset[self.time_dim][0]}. "
                    "This is not possible. If you need an earlier start date, "
                    "please reparse the dataset"
                )
        if any(insert_times):
            if insert_times[0] < original_dataset[self.time_dim][0]:
                raise IndexError(
                    f"Attempting to insert data at {insert_times[0]} "
                    f"before dataset start {original_dataset[self.time_dim][0]}. "
                    "This is not possible. If you need an earlier start date, "
                    "please reparse the dataset"
                )
        # Check that the first value of the append times and the last value of the original dataset are contiguous
        # Skip if original dataset time dim is of len 1 becasue there's no way to calculate an expected delta in situ
        if any(append_times) and len(original_dataset[self.time_dim]) > 1:
            original_append_bridge_times = [original_dataset[self.time_dim].values[-1], append_times[0]]
            expected_delta = original_dataset[self.time_dim][1] - original_dataset[self.time_dim][0]
            # Check these two values against the expected delta. All append times will be checked later in the stand
            if not self.are_times_in_expected_order(times=original_append_bridge_times, expected_delta=expected_delta):
                raise IndexError("Append would create out of order or incomplete dataset, aborting")
        # Raise an exception if there is no data to write
        if not any(insert_times) and not any(append_times):
            raise IndexError("Update started with no new records to insert or append to original zarr")

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
        # Check if times meet expected_delta or fall within the anticipated range.
        # Raise a warning and return false if so.
        # Raise a descriptive error message in the enclosing function describing the specific operation that failed.
        previous_time = times[0]
        for instant in times[1:]:
            # Warn if not using expected delta
            if self.update_cadence_bounds:
                self.warn(
                    f"Because dataset has irregular cadence {self.update_cadence_bounds} expected delta"
                    f" {expected_delta} is not being used for checking time contiguity"
                )
                if not self.update_cadence_bounds[0] <= (instant - previous_time) <= self.update_cadence_bounds[1]:
                    self.warn(
                        f"Time value {instant} and previous time {previous_time} do not fit within anticipated update "
                        f"cadence {self.update_cadence_bounds}"
                    )
                    return False
            elif instant - previous_time != expected_delta:
                self.warn(
                    f"Time value {instant} and previous time {previous_time} do not fit within expected time delta "
                    f"{expected_delta}"
                )
                return False
            previous_time = instant
        # Return True if no problems found
        return True

    def post_parse_quality_check(self, checks: int = 100, threshold: float = 10e-5):
        """
        Master function to check values written after a parse for discrepancies with the source data

        Parameters
        ----------
        checks
            The number of values to check. Defaults to 100.
        threshold
            The tolerance for diversions between original and parsed values.
            Absolute differences between them beyond this limit will raise a ValueError
        """
        if self.skip_post_parse_qc:
            self.info("Skipping post-parse quality check as directed")
        else:
            self.info("Beginning post-parse quality check of the parsed dataset")
            start_checking = time.perf_counter()
            # Instantiate needed objects
            self.set_key_dims()  # in case running w/out Transform/Parse
            prod_ds = self.get_prod_update_ds()
            # Run the data check N times, incrementing after every successfuly check
            i = 0
            while i <= checks:
                random_coords = self.get_random_coords(prod_ds)
                try:
                    orig_ds = self.get_original_ds(random_coords)
                except FileNotFoundError:
                    break
                i += self.check_written_value(random_coords, orig_ds, prod_ds, threshold)
                # Theoretically this could loop endlessly if all input files don't match the prod dataset
                # in the time dimension. While improbable, let's build an automated exit just in case
                if time.perf_counter() - start_checking > 1200:
                    self.info(
                        "Breaking from checking loop after "
                        f"{datetime.timedelta(seconds=time.perf_counter() - start_checking)} "
                        "to prevent infinite checks"
                    )
                    break

            self.info(
                "Values check successfully passed. "
                f"Checking dataset took {datetime.timedelta(seconds=time.perf_counter() - start_checking)}"
            )

        return True

    def get_prod_update_ds(self) -> xr.Dataset:
        """
        Get the prod dataset and filter it to the temporal extent of the latest update

        Returns
        -------
        prod_ds
            The production dataset filtered to only the temporal extent of the latest update
        """
        prod_ds = self.store.dataset()
        update_date_range = slice(
            datetime.datetime.strptime(prod_ds.attrs["update_date_range"][0], "%Y%m%d%H"),
            datetime.datetime.strptime(prod_ds.attrs["update_date_range"][1], "%Y%m%d%H"),
        )
        time_select = {self.time_dim: update_date_range}
        return prod_ds.sel(**time_select)

    def get_original_ds(self, random_coords: dict[Any]) -> tuple[xr.Dataset, pathlib.Path]:
        """
        Get the original dataset and format it equivalently to the production dataset

        Parameters
        ----------
        random_coords
            A randomly selected set of individual coordinate values from the filtered production dataset

        Returns
        ----------
        orig_ds
            The original dataset, unformatted
        orig_file_path
            The pathlib.Path to the randomly selected original file
        """
        # Randomly select an original dataset
        if random_coords and "step" in random_coords:
            # Forecasts create loads of files so we pre-filter to speed things up
            step_hours = random_coords["step"].astype("timedelta64[h]").astype(int)
            step_filtered_original_files = [
                fil for fil in list(self.input_files()) if f"F{step_hours:03}." in str(fil)
            ]
            try:
                raw_ds, orig_file_path = self.binary_search_for_file(
                    target_datetime=random_coords[self.time_dim], possible_files=step_filtered_original_files
                )
            except IndexError:
                # For some datasets a given day may not have all forecasts records available in prod,
                # causing failures we need to escape w/in the post_parse_quality_check while loop
                raise FileNotFoundError
        else:
            raw_ds, orig_file_path = self.binary_search_for_file(target_datetime=random_coords[self.time_dim])
        # Reformat the dataset such that it can be selected from equivalently to the prod dataset
        orig_ds = self.reformat_orig_ds(raw_ds, orig_file_path)
        return orig_ds

    def binary_search_for_file(self, target_datetime: datetime.datetime, possible_files: list[str] = None):
        """
        Implement a binary search algorithm to find the file containing a desired datetime
        within a sorted list of input files. Binary search repeatedly cuts the search space (available list indices)
        in half until the desired search target (a file with the correct datetime) is found.

        This function assumes each input file represents a single datetime -- the lowest common denominator
        of time values in the production dataset.

        Parameters
        ----------
        target_datetime
            The desired datetime
        possible_files
            A list of raw input files to select from. Defaults to list(self.input_files()).

        Raises
        ------
        TypeError
            Indicates that the requested time dimension is not present in the input file,
            either because the file was improperly prepared or the dimension name improperly specified

        FileNotFoundError
            Indicates that the requested file could not be found
        """
        if not possible_files:
            possible_files = list(self.input_files())

        low, high = 0, len(possible_files) - 1
        while low <= high:
            mid = (low + high) // 2
            current_file_path = possible_files[mid]

            with self.raw_file_to_dataset(current_file_path) as ds:
                if self.time_dim in ds:
                    # Extract time values and convert them to an array for len() and filtering, if of length 1
                    time_values = np.atleast_1d(ds[self.time_dim].values)
                    # Return the file name if the target_datetime is equal to the time value in the file,
                    # otherwise cut the search space in half based on whether the file's datetime is later (greater)
                    # or earlier (lesser) than the target datetime
                    if target_datetime == time_values:
                        return ds, current_file_path
                    # Found datetime is later than the target, look only in earlier files going foward
                    elif target_datetime < time_values[len(time_values) // 2]:
                        high = mid - 1
                    # Found datetime is earlier than the target, look only in later files going foward
                    else:
                        low = mid + 1
                else:
                    raise TypeError(
                        f"Time dimension {self.time_dim} not found in {current_file_path}!"
                        "Check that the time dimension was properly specified and the input files "
                        "correctly prepared."
                    )

        raise FileNotFoundError(
            "No file found during binary search. "
            f"Last search values low: {low}, high: {high}, possible_files: {possible_files}."
        )

    def raw_file_to_dataset(self, file_path: str):
        """
        Open a raw file as an Xarray Dataset based on the anticipated input file type

        Parameters
        ----------
        file_path
            A file path
        """
        if self.protocol == "file":
            return xr.open_dataset(file_path)
        # Presumes that use_local_zarr_jsons is enabled. This avoids repeating the DL from S#
        elif self.protocol == "s3":
            if not self.use_local_zarr_jsons:
                raise ValueError(
                    "ETL protocol is S3 but it was instantiated not to use local zarr JSONs. "
                    "This prevents running needed checks for this dataset. "
                    "Please enable `use_local_zarr_jsons` to permit post-parse QC"
                )
            # Note this will apply postprocess_zarr automtically
            return self.zarr_json_to_dataset(zarr_json_path=str(file_path))

    def reformat_orig_ds(self, orig_ds: xr.Dataset, orig_file_path: pathlib.Path) -> xr.Dataset:
        """
        Open and reformat the original dataset so it can be selected from identically to the production dataset
        Basically re-run key elements of the transform step

        Parameters
        ----------
        orig_ds
            The original dataset, unformatted
        orig_file_path
            A pathlib.Path to the randomly selected original file

        Returns
        -------
        orig_ds
            The original dataset, reformatted similarly to the production dataset
        """
        # Expand the time dimension if it's of length 1 and Xarray therefore doesn't recognize it as a dimension...
        if self.time_dim in orig_ds and self.time_dim not in orig_ds.dims:
            orig_ds = orig_ds.expand_dims(self.time_dim)
        # ... or create it from the file name if missing entirely in the raw file
        elif self.time_dim not in orig_ds:
            orig_ds = orig_ds.assign_coords(
                {self.time_dim: datetime.datetime.strptime(re.search(r"([0-9]{8})", str(orig_file_path))[0], "%Y%m%d")}
            )
            orig_ds = orig_ds.expand_dims(self.time_dim)
        # Setting metadata will clean up data variables and a few other things.
        # For Zarr JSONs this is applied by the zarr_json_to_dataset all in get_original_ds
        if self.protocol == "file":
            orig_ds = self.postprocess_zarr(orig_ds)
        # Apply standard postprocessing to get other data variables in order
        return self.rename_data_variable(orig_ds)

    def check_written_value(
        self,
        random_coords: dict[Any],
        orig_ds: xr.Dataset,
        prod_ds: xr.Dataset,
        threshold: float = 10e-5,
    ):
        """
        Check random values in the original files against the written values
        in the updated dataset at the same location

        Parameters
        ----------
        random_coords
            A randomly selected set of individual coordinate values from the filtered production dataset
        orig_ds
            The original dataset, reformatted similarly to the production dataset
        prod_ds
            The filtered production dataset
        orig_file_path
            A pathlib.Path to the randomly selected original file
        threshold
            The tolerance for diversions between original and parsed values.
            Absolute differences between them beyond this limit will raise a ValueError

        Returns
        -------
        bool
            A boolean indicating that a check was successful (True) or the selected file doesn't correspond
            to the update time range (False). Failed checks will raise a ValueError instead.
        """
        # Rework selection coordinates as needed, accounting for the absence of a time dim in some input files
        selection_coords = {key: random_coords[key] for key in orig_ds.dims}
        # Open desired data values.
        if "step" in orig_ds.dims:
            # Forecast step timedeltas are hard to select from so we have to use the nearest method.
            # We don't implement this elsewhere to minimize scope for error
            orig_val = orig_ds.sel(**selection_coords, method="nearest")[self.data_var()].values
        else:
            orig_val = orig_ds[self.data_var()].sel(**selection_coords).values
        prod_val = prod_ds[self.data_var()].sel(**selection_coords).values
        # Compare values from the original dataset to the prod dataset.
        # Raise an error if the values differ more than the permitted threshold,
        # or if only one value is either Infinite or NaN
        self.info(f"{orig_val}, {prod_val}")
        if (
            abs(orig_val - prod_val) > threshold
            or sum(np.isinf([orig_val, prod_val])) == 1
            or sum(np.isnan([orig_val, prod_val])) == 1
        ):
            raise ValueError(
                f"Mismatch: orig_val {orig_val} and prod_val {prod_val}.\nQuery parameters: {random_coords}"
            )
        return True
