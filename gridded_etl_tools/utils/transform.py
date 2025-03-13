import multiprocessing
import time
import json
import re
import fsspec
import pathlib
import glob
import os
import s3fs
import zarr

from subprocess import Popen
from typing import Union
from collections.abc import MutableMapping

import xarray as xr

from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.grib2 import scan_grib
from kerchunk.combine import MultiZarrToZarr
from tqdm import tqdm

from .convenience import Convenience
from .metadata import Metadata

TWENTY_MINUTES = 1200


class Transform(Metadata, Convenience):
    """
    Base class for transforming a collection of downloaded input files first int NetCDF4 Classic format, then
    (sequentially) kerchunk JSONs, then a MultiZarr Kerchunk JSON,
    and finally an Xarray Dataset based on that MultiZarr.
    """

    ##########################
    # RAW DATA TRANSFORMATIONS
    ##########################

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

        Raises
        ------
        ValueError
            Return a ValueError if no local_file_path is specified
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

        Raises
        ------
        ValueError
            Return a ValueError if the wrong file type or an incomplete file is passed
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

        Raises
        ------
        ValueError
            Return a ValueError if the wrong file type is passed for scanning or populated to scanned_zarr_json
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

        # NOTE this code assumes zarr_jsons is already present on the data manager, initialized upstream in
        # the __init__ for Extractor for kerchunk extracts.
        # That's not obvious from within this code block so restating here.

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
        us from passing in a local file path for each forecast step

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

    # PRE PROCESSING FILES ON DISK

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
        cls, out_zarr: zarr.storage.FsspecStore
    ) -> zarr.storage.FsspecStore:
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

        Raises
        ------
        ValueError
            Return a ValueError if no files are passed
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

        Raises
        ------
        ValueError
            Return a ValueError if no files are passed for conversion
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

    ################################
    # IN-MEMORY DATA TRANSFORMATIONS
    ################################

    # LOAD RAW DATA TO IN-MEMORY DATASET

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

        input_kwargs = {
            "filename_or_obj": "reference://",
            "engine": "zarr",
            "chunks": {},
            "backend_kwargs": {
                "storage_options": {
                    "fo": zarr_json_path,
                    "remote_protocol": self.protocol,
                    "skip_instance_cache": True,
                    "default_cache_type": "readahead",
                },
                "consolidated": False,
            },
            "decode_times": decode_times,
        }

        dataset = xr.open_dataset(**input_kwargs)

        return dataset

    # IN-MEMORY TRANSFORMS

    def preprocess_zarr(self, dataset: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        """
        Method to populate with the specific preprocessing routine of each child class (if relevant)
        Essentially replicate much of the kerchunk-based preprocessing such that `postprocess_zarr`
        can work seamlessly with single files of raw data w/out Kerchunk's interventions

        If no preprocessing is happening, return the dataset untouched

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being processed

        Returns
        -------
        dataset : xr.Dataset
            The dataset, processed
        """
        return dataset

    def postprocess_zarr(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Method to populate with the specific postprocessing routine of each child class (if relevant)

        If no postprocessing is happening, return the dataset untouched

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset being processed

        Returns
        -------
        dataset : xr.Dataset
            The dataset, processed
        """
        return dataset

    def initial_ds_transform(self) -> xr.Dataset:
        """
        In-memory transform steps relevant to an initial dataset publish

        Returns
        -------
        dataset : xr.Dataset
            The dataset from `Creation.zarr_json_to_dataset` with custom metadata, normalized axes, and rechunked
        """
        # Transform the JSON Zarr into an Xarray Dataset
        dataset = self.load_dataset_from_disk()

        # Reset standard_dims to Arbol's standard now that loading + preprocessing on the original names is done
        self.set_key_dims()
        dataset = dataset.transpose(*self.standard_dims)

        # Add metadata to dataset
        dataset = self.set_zarr_metadata(dataset)

        # Log the state of the dataset before writing
        self.info(f"Initial dataset\n{dataset}")
        return dataset

    def update_ds_transform(self) -> xr.Dataset:
        """
        In-memory transform steps relevant to an update operation

        Returns
        -------
        dataset : xr.Dataset
            The dataset from `Creation.zarr_json_to_dataset` with custom metadata, normalized axes, and rechunked
        """
        # Transform the JSON Zarr into an Xarray Dataset
        dataset = self.load_dataset_from_disk()

        # Reset standard_dims to Arbol's standard now that loading + preprocessing on the original names is done
        self.set_key_dims()

        # Log the state of the original dataset before writing
        self.info(f"Original dataset\n{self.store.dataset()}")
        return dataset

    def load_dataset_from_disk(self, zarr_json_path: str = None, decode_times: bool = True) -> xr.Dataset:
        """
        Overall method to return the fully processed and transformed dataset
        Defaults to returning zarr_json_to_datset but can be overridden to return a custom transformation instead

        Parameters
        ----------
        zarr_json_path : str
            A path to a specific Zarr JSON prepared by Kerchunk. Primarily intended for debugging.
            Defaults to None, which will trigger using the `zarr_json_path` for the dataset in question.
        decode_times : bool
            Choose whether to decode the times in inputs file using the CF conventions when opening the Zarr JSON.
            In most cases this is desirable and necessary, therefore this defaults to True.

        Returns
        -------
        postprocessed_dataset : xr.Dataset
            An Xarray dataset with dataset-specific Xarray postprocessing applied
        """
        raw_dataset = self.zarr_json_to_dataset(zarr_json_path, decode_times)
        postprocessed_dataset = self.postprocess_zarr(raw_dataset)
        return postprocessed_dataset

    def set_key_dims(self):
        """
        Set the standard and time dimension instance variables based on a dataset's type.
        Valid types are an ensemble dataset, a forecast (ensemble mean) dataset, or a "normal" observational dataset.

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

    def _standard_dims_except(self, *exclude_dims: list[str]) -> list[str]:
        return [dim for dim in self.standard_dims if dim not in exclude_dims]
