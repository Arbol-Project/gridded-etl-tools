import datetime
import itertools
import time
import re
import pprint
import dask
import pathlib
import random

from typing import Any, Generator
from statsmodels.stats.proportion import proportion_confint

import pandas as pd
import numpy as np
import xarray as xr

from dask.distributed import Client, LocalCluster

from .transform import Transform
from .errors import NanFrequencyMismatchError
from .store import S3

TWENTY_MINUTES = 1200


class ZarrOutputError(Exception):
    """Raise when an exception occurs while the Zarr is being written"""


class Publish(Transform):
    """
    Base class for publishing methods -- both initial publication and updates to existing datasets
    """

    # PARSING

    def parse(self, publish_dataset: xr.Dataset):
        """
        Write the publishable dataset prepared during `transform` to the store specified by `Attributes.store`.

        If the store is S3, an existing Zarr will be searched for to be opened and appended to by default. This
        can be overridden to force writing the entire input data to a new Zarr by setting
        `Convenience.rebuild_requested` to `True`. If existing data is found, `DatasetManager.allow_overwrite` must
        also be `True`.

        This is the core function for writing data (to disk or S3) and should be standard for
        all ETLs. Modify the child methods it calls or the dask configuration settings to resolve any performance or
        parsing issues.

        In the case of inserting or appending data, because more recent versions of Xarray + Dask choke when updating
        with pre-chunked update datasets, all chunking information (as well as chunking itself) will be aggressively
        removed.

        Parameters
        ----------
        publish_dataset : xr.Dataset
            A dataset containing all records to publish, either as an initial dataset or an update to an existing one
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
            with Client(cluster):
                self.info(f"Dask Dashboard for this parse can be found at {cluster.dashboard_link}")
                try:
                    # Attempt to find an existing Zarr, using the appropriate method for the store. If there is
                    # existing data and there is no rebuild requested, start an update. If there is no existing data,
                    # start an initial parse. If rebuild is requested and there is no existing data or allow overwrite
                    # has been set, write a new Zarr, overwriting any existing data.
                    # If rebuild is requested and there is existing data, but allow overwrite is not set, do not start
                    # parsing and issue a warning.
                    if self.store.has_existing and not self.rebuild_requested:

                        # If zarr.json is present, the format is considered 3. Otherwise, it is considered format 2.
                        if self.store.has_v3_metadata:
                            if not self.output_zarr3:
                                raise RuntimeError("Existing data is Zarr v3, but output_zarr3 is not set.")
                        elif self.output_zarr3:
                            raise RuntimeError("Existing data is not Zarr v3, but output_zarr3 is set.")

                        self.info(f"Updating existing data at {self.store}")
                        self.update_zarr(publish_dataset)
                    elif not self.store.has_existing or (self.rebuild_requested and self.allow_overwrite):
                        if not self.store.has_existing:
                            self.info(f"No existing data found. Creating new Zarr at {self.store}.")
                        else:
                            self.info(f"Data at {self.store} will be replaced.")
                        self.info(f"Now writing to {self.store}")
                        self.write_initial_zarr(publish_dataset)
                    else:
                        raise RuntimeError(
                            "There is already a zarr at the specified path and a rebuild is requested, "
                            "but overwrites are not allowed."
                        )
                    # manually closing the cluter within the Client block prevents observed serialization problems
                    # for reasons not entirely understood
                    cluster.close()
                except KeyboardInterrupt:
                    self.info("CTRL-C Keyboard Interrupt detected, exiting Dask client before script terminates")

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

        The Zarr output format will be determined by DatasetManager.output_zarr3. If output_zarr3 is true, the output
        format will be 3. Otherwise, it will be 2. Do not pass "zarr_format" to "kwargs", or a ValueError will be
        raised.

        Before the Zarr is written, "update_in_progress" will be set to True in the Zarr metadata. If a Zarr is opened,
        and "update_in_progress" is True, that indicates the Zarr is currently being written and should not be read
        from. After the Zarr is written, "update_in_progress" will be updated to False in the Zarr metadata, and the
        data can be read safely.

        ZarrOutputError will be raised if any exception occurs while the Zarr is being written. This can be used, for
        example, to check whether a Zarr needs to be rolled back to a backup version.

        Parameters
        ----------
        dataset
            Dataset to write to Zarr format
        *args
            Arguments to forward to `xr.Dataset.to_zarr`
        **kwargs
            Keyword arguments to forward to `xr.Dataset.to_zarr`

        Raises
        ------
        ZarrOutputError
            If an error occurs while the Zarr is being written
        ValueError
            If "zarr_format" is passed as a keyword argument
        """
        # First check that the data makes sense
        self.pre_parse_quality_check(dataset)

        # Exit script if dry_run specified
        if self.dry_run:
            self.info("Exiting without parsing since the dataset manager was instantiated as a dry run")
            self.info(f"Dataset final state pre-parse:\n{dataset}")
        else:
            # Determine Zarr format
            if "zarr_format" in kwargs:
                raise ValueError(
                    "zarr_format may only be controlled by setting the value of DatasetManager.output_zarr3"
                )
            else:
                zarr_format = 3 if self.output_zarr3 else 2

            # Update metadata on disk with new values for update_in_progress and update_is_append_only, so that if
            # a Zarr is opened during writing, there will be indicators that show the data is being edited.
            self.info("Writing metadata before writing data to indicate write is in progress.")
            if self.store.has_existing:
                update_attrs = {
                    "update_in_progress": True,
                    "update_is_append_only": dataset.get("update_is_append_only"),
                    "initial_parse": False,
                }
                # Use Zarr format to determine metadata format.
                if zarr_format == 3:
                    self.store.write_metadata_only(update_attrs=update_attrs)
                else:
                    self.store.write_metadata_only_v2(update_attrs=update_attrs)
                dataset.attrs.update(update_attrs)
            else:
                dataset.attrs.update({"update_in_progress": True, "initial_parse": True})

            # Time the write operation
            start_writing = time.perf_counter()

            # Write to Zarr
            try:
                # xarray uses its own S3 client, so pass it the endpoint URL from store if it was set
                if isinstance(self.store, S3) and self.store.endpoint_url is not None:
                    if "storage_options" not in kwargs:
                        kwargs["storage_options"] = {"endpoint_url": self.store.endpoint_url}
                    elif "endpoint_url" not in kwargs["storage_options"]:
                        kwargs["storage_options"]["endpoint_url"] = self.store.endpoint_url
                dataset.to_zarr(*args, zarr_format=zarr_format, **kwargs)

            # Catch any exception that occurs, and raise ZarrOutputError along with the original exception.
            except Exception as error:
                raise ZarrOutputError("Error while Zarr was being written.") from error

            # Reset the update in progress flag, whether the write was successful or not.
            finally:
                # Indicate in metadata that update is not in progress.
                self.info("Writing metadata after writing data to indicate write is finished.")
                restored_attrs = {"update_in_progress": False}

                # Use Zarr format to determine metadata format.
                if zarr_format == 3:
                    self.store.write_metadata_only(update_attrs=restored_attrs)
                else:
                    self.store.write_metadata_only_v2(update_attrs=restored_attrs)

            # Log the write duration
            self.info(f"Writing Zarr took {datetime.timedelta(seconds=time.perf_counter() - start_writing)}")

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

        # OTHER USEFUL SETTINGS, USE IF ENCOUNTERING PROBLEMS WITH PARSES
        # default distributed scheduler does not allocate memory correctly for some parses
        # dask.config.set({'scheduler' : 'threads'})
        # helps clear out unused memory
        # dask.config.set({'nanny.environ.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_' : 0})
        # dask.config.set({"distributed.worker.memory.recent-to-old-time": "300s"}) #???

        # DEBUGGING
        self.info(f"dask.config.config is {pprint.pformat(dask.config.config)}")

    # INITIAL

    def write_initial_zarr(self, publish_dataset: xr.Dataset):
        """
        Writes the first iteration of zarr for the dataset to the store specified at initialization.

        Parameters
        ----------
        publish_dataset : xr.Dataset
            A dataset containing all records to publish as an initial dataset
        """
        # Re-chunk
        self.info(f"Re-chunking dataset to {self.requested_dask_chunks}")
        # store a version of the dataset that is not re-chunked for use in the pre-parse quality check
        # this is necessary for performance reasons (rechunking for every point comparison is slow)
        self.pre_chunk_dataset = publish_dataset.copy()
        publish_dataset = publish_dataset.chunk(self.requested_dask_chunks)
        self.info(f"Chunks after rechunk are {publish_dataset.chunks}")
        # Now write
        self.to_zarr(publish_dataset, store=self.store.path, mode="w")

    # UPDATES

    def update_zarr(self, publish_dataset: xr.Dataset):
        """
        Update discrete regions of an N-D dataset saved to disk as a Zarr. Trigger insert and/or append
        operations based on the presence of valid records for either. If updates span multiple date ranges,
        push separate updates to each region.

        Parameters
        ----------
        publish_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records
        """
        original_dataset = self.store.dataset()
        self.info(f"Original dataset\n{original_dataset}")
        self.info(f"Unfiltered new data\n{publish_dataset}")
        # Create a list of any datetimes to insert and/or append
        insert_times, append_times = self.prepare_update_times(original_dataset, publish_dataset)
        # First check that the data is not obviously wrong
        self.update_quality_check(original_dataset, insert_times, append_times)
        # Now write out updates to existing data using the 'region=' command...
        if len(insert_times) > 0:
            if not self.allow_overwrite:
                self.warn(
                    "Not inserting records despite historical data detected. 'allow_overwrite'"
                    "flag has not been set."
                )
            else:
                self.insert_into_dataset(original_dataset, publish_dataset, insert_times)
        else:
            self.info("No modified records to insert into original zarr")
        # ...then write new data (appends) using the 'append_dim=' command
        if len(append_times) > 0:
            self.append_to_dataset(publish_dataset, append_times)
        else:
            self.info("No new records to append to original zarr")

    def prepare_update_times(self, original_dataset: xr.Dataset, update_dataset: xr.Dataset) -> tuple[list, list]:
        """
        Create lists of any datetimes to insert and/or append, needed inputs for the update

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

    def insert_into_dataset(
        self,
        original_dataset: xr.Dataset,
        update_dataset: xr.Dataset,
        insert_times: list,
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
        """
        insert_dataset = self.prep_update_dataset(update_dataset, insert_times)
        date_ranges, regions = self.calculate_update_time_ranges(original_dataset, insert_dataset)
        for dates, region in zip(date_ranges, regions):
            insert_slice = insert_dataset.sel(**{self.time_dim: slice(*dates)})
            insert_dataset.attrs["update_is_append_only"] = False
            self.info("Indicating the dataset is not appending data only.")

            # Align incoming time chunks with chunks in the existing Zarr. Chunks must be aligned or an exception will
            # be raised by xarray, but this is left configurable for backward compatibility with older ETLs which are
            # aligned by default from having time chunk length of 1.
            if self.align_update_chunks:
                insert_slice, region = complete_insert_slice(
                    insert_slice, original_dataset, region, self.requested_dask_chunks[self.time_dim], self.time_dim
                )

            # Write to a region of the existing Zarr
            self.to_zarr(
                insert_slice.drop_vars(self._standard_dims_except(self.time_dim)),
                store=self.store.path,
                region={self.time_dim: slice(*region)},
            )

        if not self.dry_run:
            self.info(
                f"Inserted records for {len(insert_dataset[self.time_dim].values)} times from {len(regions)} date "
                "range(s) to original zarr"
            )

    def append_to_dataset(self, update_dataset: xr.Dataset, append_times: list):
        """
        Append new records to an existing dataset along its time dimension using the `append_dim=` flag.

        Parameters
        ----------
        update_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records
        append_times : list
            Datetimes corresponding to all new records to append to the original dataset
        """
        append_dataset = self.prep_update_dataset(update_dataset, append_times)

        # Align incoming time chunks with chunks in the existing Zarr. Chunks must be aligned or an exception will
        # be raised by xarray, but this is left configurable for backward compatibility with older ETLs which are
        # aligned by default from having time chunk length of 1.
        if self.align_update_chunks:
            append_dataset = self.rechunk_append_dataset(append_dataset)
            self.info(f"Chunks after rechunking the update data are {append_dataset.chunks}")

        # Write the Zarr
        append_dataset.attrs["update_is_append_only"] = True
        self.info("Indicating the dataset is appending data only.")
        self.to_zarr(append_dataset, store=self.store.path, append_dim=self.time_dim)

        if not self.dry_run:
            self.info(f"Appended records for {len(append_dataset[self.time_dim].values)} datetimes to original zarr")

    def prep_update_dataset(self, update_dataset: xr.Dataset, time_filter_vals: list) -> xr.Dataset:
        """
        Select out and format time ranges you wish to insert or append into the original dataset based on specified
        time range(s) and chunks

        NOTE in some cases a standard dimension will be missing from a dataset
        -- for example after taking the mean/max/etc. of a given dimension.
        In these cases, we exempt it from the transpose operation.

        Parameters
        ----------
        update_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records
        time_filter_vals : list
            Datetimes corresponding to all new records to insert or append

        Returns
        -------
        update_dataset : xr.Dataset
            An xr.Dataset filtered to only the time values in `time_filter_vals`, with correct metadata
        """
        # Xarray will automatically drop dimensions of size 1. A missing time dimension causes updates to fail.
        if self.time_dim in update_dataset.dims:
            update_dataset = update_dataset.sel(**{self.time_dim: time_filter_vals})
        else:
            update_dataset = update_dataset.expand_dims(self.time_dim)

        # Transpose / order the dimensions of the dataset to the standard dimensions;
        # if a dataset is missing a standard dimension, exempt it
        transpose_dims = [dim for dim in self.standard_dims if dim in update_dataset.dims]
        update_dataset = update_dataset.transpose(*transpose_dims)

        # Add metadata to dataset
        update_dataset = self.set_zarr_metadata(update_dataset)
        # Store a non-rechunked version for pre-parse quality checks
        self.pre_chunk_dataset = update_dataset.copy()

        self.info(f"Update dataset\n{update_dataset}")
        return update_dataset

    def rechunk_append_dataset(self, append_dataset: xr.Dataset) -> xr.Dataset:
        """
        Prepare the chunks for the append dataset such that they align neatly with the initial dataset.
        The science behind this is described in detail under docs/Aligning_update_chunks.md

        Parameters
        ----------
        append_dataset : xr.Dataset
            A dataset containing new records to append to the original dataset

        Returns
        -------
        xr.Dataset
            The append dataset with correctly aligned chunks
        """
        self.info(f"Aligning update data with existing data along the {self.time_dim} dimension")

        # Most chunks have sizes independent of the size of the update and can be copied right through
        rechunk_dims = self.requested_dask_chunks.copy()

        # Calculate time dimension chunks to align with existing dataset
        time_dim_chunk_size = rechunk_dims[self.time_dim]
        existing_final_chunk_length = (
            self.store.dataset().chunks[self.time_dim][-1]
            if self.store.has_existing and not self.rebuild_requested
            else 0
        )
        append_time_length = len(append_dataset[self.time_dim].values)
        # Calculate optimal chunk distribution for time dimension
        rechunk_dims[self.time_dim] = calculate_time_dim_chunks(
            existing_final_chunk_length, time_dim_chunk_size, append_time_length
        )
        # Apply the chunks to the append dataset.
        return append_dataset.chunk(**rechunk_dims)

    def calculate_update_time_ranges(
        self, original_dataset: xr.Dataset, update_dataset: xr.Dataset
    ) -> tuple[list[datetime.datetime], list[tuple[int, int]]]:
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
        regions_indices: list[tuple[int, int]]
             A List of (int, int) tuples defining the indices of records to insert
        """
        # NOTE this won't work for months (returns 1 minute) because of how pandas handles timedeltas
        # We could define a more precise method with if/else statements if needed.
        # Get the time unit from the TimeSpan enum member
        time_unit = self.time_resolution.get_time_unit()
        # Create a pandas Timedelta string in the format "1h", "1d", etc.
        dataset_time_span = f"{time_unit.value}{time_unit.unit[0]}"

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

    # QUALITY CHECKS

    def pre_parse_quality_check(self, dataset: xr.Dataset):
        """
        Guard against corrupted source data by applying quality checks to all datasets we parse, initial or update
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
        # Only valid if the dataset's time dimension has 2 or more values with which to calculate the delta
        times = dataset[self.time_dim].values
        if len(times) >= 2:
            expected_delta = times[1] - times[0]
            if not self.are_times_in_expected_order(times=times, expected_delta=expected_delta):
                raise IndexError("Dataset does not contain contiguous time data")

        # VALUES CHECK
        # Check 100 values for unanticipated NaNs and extreme values
        self.check_random_values(dataset.copy())

        # ENCODING CHECK
        # Check that data is stored in a space efficient format
        if not dataset[self.data_var].encoding["dtype"] == self.data_var_dtype:
            raise TypeError(
                f"Dtype for data variable {self.data_var} is "
                f"{dataset[self.data_var].dtype} when it should be {self.data_var_dtype}"
            )

        # NAN CHECK
        # Check that the % of NaN values approximately matches the historical average. Not applicable on first run.
        if self.store.has_existing and not self.skip_pre_parse_nan_check and not self.rebuild_requested:
            self.check_nan_frequency()

        self.info(f"Checking dataset took {datetime.timedelta(seconds=time.perf_counter() - start_checking)}")

    def check_random_values(self, dataset: xr.Dataset, checks: int = 100):
        """
        Check N random values from the finalized dataset for any obviously wrong data points,
        either unanticipated NaNs or extreme values

        Returns
        -------
        random_values
            A dictionary of randomly selected values with their corresponding coordinates.
            Intended for later reuse checking the same coordinates after a dataset is parsed.
        """
        # insert operations will create datasets w/ only time coordinates and index values for other coords
        # this will cause comparison w/ the `pre_chunk_dataset` below to fail as index values != actual vals
        # therefore we repopulate the original values to enable comparisons
        if len(dataset.coords) == 1:
            orig_coords = {
                coord: self.pre_chunk_dataset.coords[coord].values
                for coord in self.pre_chunk_dataset.drop_vars(self.time_dim).coords
            }
            dataset = dataset.assign_coords(**orig_coords)
        for random_coords in itertools.islice(shuffled_coords(dataset), checks):
            random_val = self.pre_chunk_dataset[self.data_var].sel(**random_coords).values
            # Check for unanticipated NaNs
            if np.isnan(random_val) and not self.has_nans:
                raise ValueError(f"NaN value found for random point at coordinates {random_coords}")
            # Check extreme values if they are defined
            if not np.isnan(random_val):
                unit = dataset[self.data_var].encoding["units"]
                if unit in self.EXTREME_VALUES_BY_UNIT.keys():
                    limit_vals = self.EXTREME_VALUES_BY_UNIT[unit]
                    if not limit_vals[0] <= random_val <= limit_vals[1]:
                        raise ValueError(
                            f"Value {random_val} falls outside acceptable range "
                            f"{limit_vals} for data in units {unit}. Found at {random_coords}"
                        )

    def check_nan_frequency(self):
        """
        Use a binomial test to check whether the percentage of NaN values matches
        the anticipated percentage within the dataset, based on the observed ratio of NaNs in historical data

        Tests every time period in the update dataset

        Raises
        ------
        AttributeError
            Inform ETL operator that the expected_nan_frequency field
            used by the binomial test are missing from the production dataset.
        """
        if "expected_nan_frequency" not in self.pre_chunk_dataset.attrs:
            raise AttributeError(
                "Update dataset is missing the `expected_nan_frequency` field in its attributes. "
                "Please calculate and populate this field manually "
                "to enable NaN quality checks during updates."
            )
        for update_dt_index in range(len(self.pre_chunk_dataset[self.time_dim])):
            time_value = self.pre_chunk_dataset[self.time_dim].values[update_dt_index]
            selected_array = self.pre_chunk_dataset.sel(**{self.time_dim: time_value})[self.data_var].values
            test_nan_frequency(
                data_array=selected_array,
                expected_nan_frequency=self.pre_chunk_dataset.attrs["expected_nan_frequency"],
            )

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
        if append_times and append_times[0] < original_dataset[self.time_dim][0]:
            raise IndexError(
                f"Attempting to append data at {append_times[0]} "
                f"before dataset start {original_dataset[self.time_dim][0]}. "
                "This is not possible. If you need an earlier start date, "
                "please reparse the dataset"
            )
        if insert_times and insert_times[0] < original_dataset[self.time_dim][0]:
            raise IndexError(
                f"Attempting to insert data at {insert_times[0]} "
                f"before dataset start {original_dataset[self.time_dim][0]}. "
                "This is not possible. If you need an earlier start date, "
                "please reparse the dataset"
            )

        # Check that the first value of the append times and the last value of the original dataset are contiguous
        # Skip if original dataset time dim is of len 1 because there's no way to calculate an expected delta in situ
        if append_times and len(original_dataset[self.time_dim]) > 1:
            original_append_bridge_times = [original_dataset[self.time_dim].values[-1], append_times[0]]
            expected_delta = original_dataset[self.time_dim][1] - original_dataset[self.time_dim][0]
            # Check these two values against the expected delta. All append times will be checked later in the stand
            if not self.are_times_in_expected_order(times=original_append_bridge_times, expected_delta=expected_delta):
                raise IndexError("Append would create out of order or incomplete dataset, aborting")

        # Raise an exception if there is no data to write
        if not insert_times and not append_times:
            raise ValueError("Update started with no new records to insert or append to original zarr")

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
            possible_files = self.filter_search_space(prod_ds)

            # Run the data check N times
            i = 0
            while i < checks:
                # Open and reformat the original dataset such that it's comparable with the prod dataset
                orig_ds = self.raw_file_to_dataset(random.choice(possible_files))

                # Run the checks
                self.check_written_value(orig_ds, prod_ds, threshold)
                i += 1

                # While improbable, if it takes longer than 20 minutes to get the number of checks we're looking for,
                # go ahead and bail.
                elapsed = time.perf_counter() - start_checking
                if elapsed > TWENTY_MINUTES:
                    self.info(f"Breaking from checking loop after {datetime.timedelta(seconds=elapsed)}")
                    break

            elapsed = time.perf_counter() - start_checking
            self.info(
                "Written values check successfully passed. "
                f"Checking dataset took {datetime.timedelta(seconds=elapsed)}"
            )

    def filter_search_space(self, prod_ds: xr.Dataset) -> list[pathlib.Path]:
        """
        Filter down all input files to only files that are within the update date range on the production dataset.

        NOTE this implicitly relies on input files being sorted by the time dimension, so that the determined bounds
        encapsulate the entire range of valid data.

        This was originally written for input files containing a single time step each. It now has initial support for
        input files containing multiple time steps.

        Parameters
        ----------
        prod_ds : xr.Dataset
            The production dataset, filtered down to the most recent update

        Returns
        -------
        list[pathlib.Path]
            A list of valid input files in pathlib.Path format
        """
        possible_files = list(self.input_files())
        start_date, end_date = self.strings_to_date_range(prod_ds.attrs["update_date_range"])
        n = len(possible_files)

        # Find leftmost file >= start_date. Use the latest timestamp contained in the file, so the file is checked for
        # any time steps that exceed the start date.
        left, right = 0, n
        while left < right:
            mid = (left + right) // 2
            if self.time_range_in_file(possible_files[mid])[1] < start_date:
                left = mid + 1
            else:
                right = mid
        start_idx = left

        # Find rightmost file <= end_date. Use the earliest timestamp contained in the file, so the file is checked for
        # any time steps prior to the end date.
        left, right = start_idx, n
        while left < right:
            mid = (left + right) // 2
            if self.time_range_in_file(possible_files[mid])[0] <= end_date:
                left = mid + 1
            else:
                right = mid  # pragma NO COVER
        end_idx = right - 1

        self.debug(
            "Date range of filtered input files starts with"
            f" {self.time_range_in_file(possible_files[start_idx])[0]}"
            f" and ends with {self.time_range_in_file(possible_files[end_idx])[1]}"
        )

        return possible_files[start_idx : end_idx + 1]

    def time_range_in_file(self, file_path: pathlib.Path) -> tuple[datetime.datetime, datetime.datetime]:
        """
        Convert a file to an xarray.Dataset and return the start and end of the list of timestamps contained in the time
        dimension.

        Parameters
        ----------
        file_path : pathlib.Path
            The path to the original file, on disk or remotely

        Returns
        -------
        tuple[datetime.datetime, datetime.datetime]
            The latest date contained in the time dimension
        """
        dataset = self.raw_file_to_dataset(file_path)
        return self.get_date_range_from_dataset(dataset)

    def dataset_date_in_range(
        self, date_range: tuple[datetime.datetime, datetime.datetime], file_path: pathlib.Path
    ) -> bool:
        """
        Assess whether the date in the selected original dataset falls within a specified date range,
        implicitly derived from the most recent update to the production dateaset

        Parameters
        ----------
        date_range : tuple[datetime.datetime, datetime.datetime]
            The date range for the production dataset
        file_path : pathlib.Path
            The path to the original file, on disk or remotely

        Returns
        -------
        bool
            An indication of whether a single datetime dataset falls within a specified date range
        """
        orig_ds = self.raw_file_to_dataset(file_path)

        if date_range[0] <= self.numpydate_to_py(orig_ds[self.time_dim].values[0]) <= date_range[1]:
            return True
        else:
            return False

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

    def check_written_value(
        self,
        orig_ds: xr.Dataset,
        prod_ds: xr.Dataset,
        threshold: float = 10e-5,
    ):
        """
        Check random values in the original files against the written values
        in the updated dataset at the same location

        Parameters
        ----------
        orig_ds
            A randomly selected original dataset
        prod_ds
            The production dataset, filtered down to the time range of the latest update
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

        Raises
        ------
        ValueError
            Indicates a potentially problematic mismatch between source data values and values written to production
        """
        selection_coords = self.get_random_coords(orig_ds)

        # # Rework selection coordinates as needed, accounting for the absence of a time dim in some input files
        # selection_coords = {key: check_coords[key] for key in orig_ds.dims}

        # Open desired data values.
        orig_val = orig_ds[self.data_var].sel(**selection_coords).values
        prod_val = (
            prod_ds[self.data_var].sel(**selection_coords, method="nearest", tolerance=self.check_tolerance).values
        )

        # Compare values from the original dataset to the prod dataset.
        # Raise an error if the values differ more than the permitted threshold,
        # or if only one value is either Infinite or NaN
        self.info(f"{orig_val}, {prod_val}")
        if _is_infish(orig_val):
            if _is_infish(prod_val):
                # Both are infinity, great
                return

            # else, one is infinity, raise error

        elif _is_infish(prod_val):
            # one is infinity, raise error
            pass

        elif np.isnan(orig_val) and np.isnan(prod_val):
            # Both are nan, great
            return

        elif orig_val == self.missing_value and np.isnan(prod_val):
            # Recognized NaN values were written to prod as NaNs, muy bien
            return

        # There may be one nan, or they may both be actual numbers
        elif abs(orig_val - prod_val) <= threshold:
            # They are both actual numbers and they are close enough to the same value to match
            return

        raise ValueError(
            "Mismatch in written values: "
            f"orig_val {orig_val} and prod_val {prod_val}."
            f"\nQuery parameters: {selection_coords}"
        )

    def raw_file_to_dataset(self, file_path: pathlib.Path) -> xr.Dataset:
        """
        Open a raw file as an Xarray Dataset based on the anticipated input file type

        Parameters
        ----------
        file_path
            A file path
        """
        if self.protocol == "file":
            ds = xr.open_dataset(file_path, **self.open_dataset_kwargs)
            # Apply pre- and post-processing so that file can be selected from equivalently to
            # the production dataset
            ds = self.preprocess_zarr(ds, file_path)
            ds = self.postprocess_zarr(ds)
            return self.reformat_orig_ds(ds, file_path)

        # Presumes that use_local_zarr_jsons is enabled. This avoids repeating the DL from S#
        elif self.protocol == "s3":
            if not self.use_local_zarr_jsons:
                raise ValueError(
                    "ETL protocol is S3 but it was instantiated not to use local zarr JSONs. "
                    "This prevents running needed checks for this dataset. "
                    "Please enable `use_local_zarr_jsons` to permit post-parse QC"
                )

            # This will apply postprocess_zarr automtically
            ds = self.load_dataset_from_disk(zarr_json_path=str(file_path))
            return self.reformat_orig_ds(ds, file_path)

        else:
            raise ValueError('Expected either "file" or "s3" protocol')

    def reformat_orig_ds(self, ds: xr.Dataset, orig_file_path: pathlib.Path) -> xr.Dataset:
        """
        Open and reformat the original dataset so it can be selected from identically to the production dataset
        Basically re-run key elements of the transform step

        Parameters
        ----------
        ds
            The original dataset, unformatted
        orig_file_path
            A pathlib.Path to the randomly selected original file

        Returns
        -------
        ds
            The original dataset, reformatted similarly to the production dataset
        """
        # Apply standard postprocessing to get other data variables in order
        ds = self.rename_data_variable(ds)
        # Expand any 1D dimensions as needed. This is necessary for later `sel` operations.
        for time_dim in [
            time_dim
            for time_dim in [self.time_dim, "step", "ensemble", "forecast_reference_offset"]
            if time_dim in self.standard_dims
        ]:
            # Expand the time dimension if it's of length 1 and Xarray therefore doesn't recognize it as a dimension...
            # for some reason this is unrecognized by cover, although tested in two places
            if time_dim in ds and time_dim not in ds.dims:  # pragma NO COVER
                ds = ds.expand_dims(time_dim)

            # ... or create it from the file name if missing entirely in the raw file
            elif time_dim not in ds:
                ds = ds.assign_coords(
                    {
                        time_dim: datetime.datetime.strptime(
                            re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", str(orig_file_path))[0], "%Y-%m-%d"
                        )
                    }
                )
                ds = ds.expand_dims(time_dim)

            # Also expand it for the data var!
            if time_dim in ds.dims and time_dim not in ds[self.data_var].dims:
                ds[self.data_var] = ds[self.data_var].expand_dims(time_dim)

        return ds


def test_nan_frequency(
    data_array: np.ndarray,
    expected_nan_frequency: float,
    sample_size: int = 5000,
    alpha: float = 0.00001,
):
    """
    Test whether the frequency of NaNs in an Xarray DataArray matches the expected distribution
    using a binomial test

    This will raise an error if the binomial test fails, otherwise passes silently

    Parameters
    ----------
    data_array : np.ndarray
        The numpy array to test.
    expected_frequency : float
        The expected frequency of NaNs
    sample_size : int
        The number of sample values to randomly extract from the selected time series
        of the update dataset
    alpha : float
        The significance level for the hypothesis test (default is 0.001).

    Returns
    -------
    bool
        True if the observed frequency matches the expected frequency, False otherwise.
    float
        The p-value from the binomial test.

    Raises
    ------
    NanFrequencyMismatchError
        An error indicating the observed frequency of NaNs does not correpsond
        with the observed frequency in historical dataset, even allowing for a margin
        of error defined by the standard deviation of sample-based estimates of that
        historical frequency
    """
    # Select N random values
    flat_array = np.ravel(data_array)  # necessary for random selection
    if np.size(flat_array) < sample_size:
        sample_size = np.prod(flat_array.shape)
    random_values = np.random.default_rng().choice(flat_array, sample_size, replace=False)
    nan_count = np.isnan(random_values).sum()

    # Calculate the confidence interval
    lower_bound, upper_bound = proportion_confint(nan_count, sample_size, alpha=alpha, method="binom_test")

    # Check if the expected frequency falls within the confidence interval
    if not (lower_bound <= expected_nan_frequency <= upper_bound):
        raise NanFrequencyMismatchError(nan_count / sample_size, expected_nan_frequency, lower_bound, upper_bound)


def shuffled_coords(dataset: xr.Dataset) -> Generator[dict[str, Any], None, None]:
    """
    Iterate through all of the coordinates in a dataset in a random order.

    Parameters
    ----------
    dataset
        An Xarray dataset

    Returns
    -------
        A generator of {str: Any} dicts of coordinate values, where key is the name of the dimension, and value is
            the coordinate value for that dimension.
    """
    n_elements = 1
    shuffled = {}
    for dim in dataset.dims:
        coords = dataset[dim].values.copy()
        np.random.shuffle(coords)
        shuffled[dim] = coords
        n_elements *= len(coords)

    for i in range(n_elements):
        coords = []
        offset = 0
        x = i
        for dim in dataset.dims:
            values = shuffled[dim]
            n_values = len(values)
            index = x % n_values
            coords.append(values[(index + offset) % n_values])
            offset += index
            x //= n_values

        yield dict(zip(dataset.dims, coords))


def _is_infish(n):
    """Tell if a value is infinite-ish or not.

    An infinite-ish value may either be one of the floating point `inf` values or have an absolute value greater than
    1e100.
    """
    if n.dtype == np.float64:
        limit = 1e100
    else:
        limit = 1e38
    return np.isinf(n) or abs(n) > limit


def calculate_time_dim_chunks(
    old_dataset_final_chunk_length: int, time_dim_chunk_size: int, append_time_length: int
) -> tuple[int, ...]:
    """
    Create a bespoke chunk tuple to ensure the first chunk in the append can be added to the
    last chunk of the initial dataset smoothly, i.e. it does not bridge two chunks.

    For example, for an 8 record update to a dataset with desired chunks of 5 time
    and a final time dim chunk of length 3, the desired chunks would be (2,5,1).

    Parameters
    ----------
    old_dataset_final_chunk_length : int
        The length of the final chunk in the original dataset along the time dimension.
    time_dim_chunk_size : int
        The desired size of each time chunk.
    append_time_length : int
        The length of the time dimension in the append.

    Returns
    -------
    tuple of int
        Chunk tuple whose entries sum to `append_time_length`.
    """
    # Calculate first chunk size to align with existing dataset
    first_chunk_size = min(time_dim_chunk_size - old_dataset_final_chunk_length, append_time_length)

    # Calculate remaining length and full chunks
    remaining_length = append_time_length - first_chunk_size
    full_chunks_count = remaining_length // time_dim_chunk_size
    final_chunk_size = remaining_length % time_dim_chunk_size

    desired_chunks = []
    # first chunk must align with end of initial dataset
    if first_chunk_size > 0:
        desired_chunks.append(first_chunk_size)
    # now write out full chunks until we run out of room for full chunks
    desired_chunks.extend([time_dim_chunk_size] * full_chunks_count)
    # tack on the remaining partial chunk (if it exists)
    if final_chunk_size > 0:
        desired_chunks.append(final_chunk_size)

    return tuple(desired_chunks)


def complete_insert_slice(
    insert_slice: xr.Dataset, original_dataset: xr.Dataset, region: tuple[int, int], chunk_size: int, time_dim: str
) -> tuple[xr.Dataset, tuple[int, int]]:
    """
    With new zarr versions, inserts will fail when the insert dataset crosses chunk boundaries or
    occupies only a part of a chunk in the original dataset. To fix this, we fill in the insert
    with datapoints from the original dataset, so that it exactly fills an integer number of chunks,
    and then rechunk it so it aligns with the original dataset

    Parameters
    ----------
    insert_slice : xr.Dataset
        insert slice to complete
    original_dataset : xr.Dataset
        full original dataset to use to complete insert_slice
    region : tuple[int, int]
        start and end indices of insert_slice within original_dataset
    chunk_size : int
        desired chunk size for time dim
    time_dim : str
        name of time dimension

    Returns
    -------
    tuple[xr.Dataset, tuple[int, int]]:
        Completed, rechunked slice along with the indices of that slice within larger dataset
    """

    # to find the start index of the completed slice, move back from the start index
    # of the update until we get to the nearest chunk boundary
    chunk_start = (region[0] // chunk_size) * chunk_size

    # to find the end index, move forward from the slice end index until we hit a boundary
    chunk_end = (((region[1] - 1) // chunk_size) + 1) * chunk_size

    # take only the relevant region of the original dataset
    original_dataset_slice = original_dataset.isel({time_dim: slice(chunk_start, chunk_end)})

    # Fill in the missing regions of input slice with the relevant region from the original
    full_index_slice = insert_slice.combine_first(original_dataset_slice)

    # now that we know that that the start and end are aligned with chunk boundaries,
    # we can safely rechunk to the original time chunks schema
    full_index_slice_rechunked = full_index_slice.chunk({time_dim: chunk_size})
    return full_index_slice_rechunked, (chunk_start, chunk_end)
