import datetime
import itertools
import time
import re
import pprint
import dask
import pathlib

from contextlib import nullcontext
from typing import Any, Generator

import pandas as pd
import numpy as np
import xarray as xr

from dask.distributed import Client, LocalCluster

from .store import IPLD
from .transform import Transform

TWENTY_MINUTES = 1200


class Publish(Transform):
    """
    Base class for publishing methods -- both initial publication and updates to existing datasets
    """

    # PARSING

    def parse(self, publish_dataset: xr.Dataset):
        """
        Write the publishable dataset prepared during `transform` to the store specified by `Attributes.store`.

        If the store is IPLD or S3, an existing Zarr will be searched for to be opened and appended to by default. This
        can be overridden to force writing the entire input data to a new Zarr by setting
        `Convenience.rebuild_requested` to `True`. If existing data is found, `DatasetManager.allow_overwrite` must
        also be `True`.

        This is the core function for writing data (to disk, S3, or IPLD) and should be standard for
        all ETLs. Modify the child methods it calls or the dask configuration settings to resolve any performance or
        parsing issues.

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
                    update_attrs = {
                        "update_in_progress": True,
                        "update_is_append_only": dataset.get("update_is_append_only"),
                        "initial_parse": False,
                    }
                    self.store.write_metadata_only(update_attrs=update_attrs)
                    dataset.attrs.update(update_attrs)
                else:
                    dataset.attrs.update({"update_in_progress": True, "initial_parse": True})
                # Remove update attributes from the dataset putting them in a dictionary to be written post-parse
                post_parse_attrs = self.move_post_parse_attrs_to_dict(dataset=dataset)

            # Write data to Zarr and log duration.
            start_writing = time.perf_counter()
            dataset.to_zarr(*args, **kwargs)
            self.info(f"Writing Zarr took {datetime.timedelta(seconds=time.perf_counter() - start_writing)}")

            # Don't use update-in-progress metadata flag on IPLD
            if not isinstance(self.store, IPLD):
                # Indicate in metadata that update is complete.
                self.info("Writing metadata after writing data to indicate write is finished.")
                self.store.write_metadata_only(update_attrs=post_parse_attrs)

    def move_post_parse_attrs_to_dict(self, dataset: xr.Dataset) -> dict[str, Any]:
        """
        Build a dictionary of attributes that should only be populated to a Zarr after parsing finishes.
        Parameters
        ----------
        dataset
            The xr.Dataset about to be written

        Returns
        -------
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
                update_attrs[attr] = dataset.attrs[attr]

        return update_attrs

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

    # INITIAL

    def write_initial_zarr(self, publish_dataset: xr.Dataset):
        """
        Writes the first iteration of zarr for the dataset to the store specified at initialization. If the store is
        `IPLD`, does some additional metadata processing

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
        mapper = self.store.mapper(set_root=False)
        self.to_zarr(publish_dataset, mapper, consolidated=True, mode="w")
        if isinstance(self.store, IPLD):
            self.dataset_hash = str(mapper.freeze())

    # UPDATES

    def update_zarr(self, publish_dataset: xr.Dataset):
        """
        Update discrete regions of an N-D dataset saved to disk as a Zarr. Trigger insert and/or append
        operations based on the presence of valid records for either. If updates span multiple date ranges,
        push separate updates to each region.

        If the IPLD store is in use, after updating the dataset, this function updates
        the corresponding STAC Item and summaries in the parent STAC Collection.

        Parameters
        ----------
        publish_dataset : xr.Dataset
            A dataset containing all updated (insert) and new (append) records
        """
        original_dataset = self.store.dataset()
        self.info(f"Original dataset\n{original_dataset}")
        # Create a list of any datetimes to insert and/or append
        insert_times, append_times = self.prepare_update_times(original_dataset, publish_dataset)
        # First check that the data is not obviously wrong
        self.update_quality_check(original_dataset, insert_times, append_times)
        # Now write out updates to existing data using the 'region=' command...
        if len(insert_times) > 0:
            if not self.allow_overwrite:
                self.warn(
                    "Not inserting records despite historical data detected. 'allow_overwrite'"
                    "flag has not been set and store is not IPLD"
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
        mapper = self.store.mapper()

        insert_dataset = self.prep_update_dataset(update_dataset, insert_times)
        date_ranges, regions = self.calculate_update_time_ranges(original_dataset, insert_dataset)
        for dates, region in zip(date_ranges, regions):
            insert_slice = insert_dataset.sel(**{self.time_dim: slice(*dates)})
            insert_dataset.attrs["update_is_append_only"] = False
            self.info("Indicating the dataset is not appending data only.")
            self.to_zarr(
                insert_slice.drop_vars(self._standard_dims_except(self.time_dim)),
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

    def prep_update_dataset(self, update_dataset: xr.Dataset, time_filter_vals: list) -> xr.Dataset:
        """
        Select out and format time ranges you wish to insert or append into the original dataset based on specified
        time range(s) and chunks

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
        update_dataset = update_dataset.chunk(self.requested_dask_chunks)

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

    # QUALITY CHECKS

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
        self.check_random_values(dataset.copy())

        # ENCODING CHECK
        # Check that data is stored in a space efficient format
        if not dataset[self.data_var()].encoding["dtype"] == self.data_var_dtype:
            raise TypeError(
                f"Dtype for data variable {self.data_var()} is "
                f"{dataset[self.data_var()].dtype} when it should be {self.data_var_dtype}"
            )
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
            random_val = self.pre_chunk_dataset[self.data_var()].sel(**random_coords).values
            # Check for unanticipated NaNs
            if np.isnan(random_val) and not self.has_nans:
                raise ValueError(f"NaN value found for random point at coordinates {random_coords}")
            # Check extreme values if they are defined
            if not np.isnan(random_val):
                unit = dataset[self.data_var()].encoding["units"]
                if unit in self.EXTREME_VALUES_BY_UNIT.keys():
                    limit_vals = self.EXTREME_VALUES_BY_UNIT[unit]
                    if not limit_vals[0] <= random_val <= limit_vals[1]:
                        raise ValueError(
                            f"Value {random_val} falls outside acceptable range "
                            f"{limit_vals} for data in units {unit}. Found at {random_coords}"
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

            # Run the data check N times, incrementing after every successfuly check
            for random_coords in itertools.islice(shuffled_coords(prod_ds), checks):
                self.check_written_value(random_coords, prod_ds, threshold)

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
        check_coords: dict[Any],
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

        Raises
        ------
        ValueError
            Indicates a potentially problematic mismatch between source data values and values written to production
        """
        orig_ds = self.get_original_ds(check_coords)

        # Rework selection coordinates as needed, accounting for the absence of a time dim in some input files
        selection_coords = {key: check_coords[key] for key in orig_ds.dims}

        # Open desired data values.
        orig_val = orig_ds[self.data_var()].sel(**selection_coords, method="nearest", tolerance=0.0001).values
        prod_val = prod_ds[self.data_var()].sel(**selection_coords).values

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

        # There may be one nan, or they may both be actual numbers
        elif abs(orig_val - prod_val) <= threshold:
            # They are both actual numbers and they are close enough to the same value to match
            return

        raise ValueError(
            "Mismatch in written values: "
            f"orig_val {orig_val} and prod_val {prod_val}."
            f"\nQuery parameters: {check_coords}"
        )

    def get_original_ds(self, coords: dict[Any]) -> xr.Dataset:
        """
        Retrieve the original dataset that corresponds to the given coordinates by conducting a consecutive series of
        binary searches for files matching each coordinate.

        Parameters
        ----------
        coords
            Randomly selected coordinates from the production dataset
            used to locate the corresponding original dataset.

        Returns
        ----------
        orig_ds
            The original dataset, formatted minimally to allow comparison with the production dataset
        """
        # Apply any custom filters to the input files. Returns all input files if no custom filter defined.
        possible_files = self.custom_file_filter(list(self.input_files()))

        # Select an original dataset containing the same time dimension value as the production dataset
        # Orig_file_path is unneeded but useful for debugging
        orig_ds, orig_file_path = self.binary_search_for_file(
            coords[self.time_dim], time_dim=self.time_dim, possible_files=possible_files
        )

        # If a forecast dataset then search again, this time for the correct step
        if "step" in coords:
            time_filtered_original_files = self.filter_files_by_time(original_files=possible_files, raw_ds=orig_ds)
            orig_ds, orig_file_path = self.binary_search_for_file(
                coords["step"], time_dim="step", possible_files=time_filtered_original_files
            )

        # If an ensemble dataset then search again, this time for the correct ensemble number
        if "ensemble" in coords:
            step_filtered_original_files = self.filter_files_by_step(
                original_files=time_filtered_original_files, raw_ds=orig_ds
            )
            orig_ds, orig_file_path = self.binary_search_for_file(
                coords["ensemble"], time_dim="ensemble", possible_files=step_filtered_original_files
            )

        # If a hindcast dataset then search again, this time for the correct forecast_reference_offset
        if "forecast_reference_offset" in coords:
            ensemble_filtered_original_files = self.filter_files_by_ensemble(
                original_files=step_filtered_original_files, raw_ds=orig_ds
            )
            orig_ds, orig_file_path = self.binary_search_for_file(
                coords["forecast_reference_offset"],
                time_dim="forecast_reference_offset",
                possible_files=ensemble_filtered_original_files,
            )

        return orig_ds

    def custom_file_filter(self, original_files: tuple[str]) -> tuple[str]:
        """
        Filter down a list of local files to include only files that meet a custom criteria

        Meant to be modified as a child method.

        Parameters
        ----------
        original_files: tuple[str]
            A list of raw files

        Returns
        -------
        tuple[str]
            A list of raw files, filtered to only contain files that meet the criteria.
            Defaults to returning the same list as a pass through.
        """
        return original_files

    def filter_files_by_step(self, original_files: tuple[str], raw_ds: xr.Dataset) -> tuple[str]:
        """
        Find the time in a selected raw dataset and filter down a list of local files to include only
        files that contain that time

        Parameters
        ----------
        original_files: tuple[str]
            A list of raw files

        raw_ds : xr.Dataset
            The raw dataset used for filtering

        Returns
        -------
        time_filtered_original_files : tuple[str]
            A list of raw files, filtered to only contain files that contain the specified time
        """
        time_string = self.numpydate_to_py(np.atleast_1d(raw_ds[self.time_dim])[0]).date().isoformat()
        time_filtered_original_files = [fil for fil in original_files if time_string in str(fil)]
        return time_filtered_original_files

    def binary_search_for_file(
        self, target_datetime: np.datetime64, time_dim: str, possible_files: list[str]
    ) -> tuple[xr.Dataset, pathlib.Path]:
        """
        Implement a binary search algorithm to find the file containing a desired datetime
        within a sorted list of input files. Binary search repeatedly cuts the search space (available list indices)
        in half until the desired search target (a file with the correct datetime) is found.

        This function assumes each input file represents a single datetime -- the lowest common denominator
        of time values in the production dataset.

        Parameters
        ----------
        coords
            The coordinates to use to search for the file. Only the time dimension is used.
        time_dim
            The name of the time dimension to check against
        possible_files
            A list of raw input files to select from. Defaults to list(self.input_files()).

        Returns
        ----------
        ds
            The original dataset, unformatted
        current_file_path
            The pathlib.Path to the randomly selected original file

        Raises
        ------
        ValueError
            Indicates that the requested time dimension is not present in the input file,
            either because the file was improperly prepared or the dimension name improperly specified

        FileNotFoundError
            Indicates that the requested file could not be found
        """
        low, high = 0, len(possible_files) - 1
        while low <= high:
            mid = (low + high) // 2
            current_file_path = possible_files[mid]

            # Reformat the dataset such that it can be selected from equivalently to the prod dataset
            with self.raw_file_to_dataset(current_file_path) as ds:

                if time_dim in ds:
                    # Extract time values and convert them to an array for len() and filtering, if of length 1
                    # These should already be formatted in a np.datetime64 format via `raw_file_to_dataset`
                    time_values = np.atleast_1d(ds[time_dim].values)
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
                    raise ValueError(
                        f"Time dimension {time_dim} not found in {current_file_path}!"
                        "Check that the time dimension was properly specified and the input files "
                        "correctly prepared."
                    )

        raise FileNotFoundError(
            "No file found during binary search. "
            f"Last search values target datetime: {target_datetime} "
            f" low: {low}, high: {high}, {len(possible_files)} total possible_files."
        )

    def raw_file_to_dataset(self, file_path: str) -> xr.Dataset:
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
            return self.reformat_orig_ds(ds, file_path)

        # Presumes that use_local_zarr_jsons is enabled. This avoids repeating the DL from S#
        elif self.protocol == "s3":
            if not self.use_local_zarr_jsons:
                raise ValueError(
                    "ETL protocol is S3 but it was instantiated not to use local zarr JSONs. "
                    "This prevents running needed checks for this dataset. "
                    "Please enable `use_local_zarr_jsons` to permit post-parse QC"
                )

            # Note this will apply postprocess_zarr automtically
            return self.load_dataset_from_disk(zarr_json_path=str(file_path))

        else:
            raise ValueError('Expected either "file" or "s3" protocol')

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
        # Setting metadata will clean up data variables and a few other things.
        # For Zarr JSONs this is applied by the zarr_json_to_dataset all in get_original_ds
        if self.protocol == "file":
            orig_ds = self.preprocess_zarr(orig_ds, orig_file_path)
            orig_ds = self.postprocess_zarr(orig_ds)

        # Apply standard postprocessing to get other data variables in order
        processed_orig_ds = self.rename_data_variable(orig_ds)

        # Expand any 1D dimensions as needed. This is necessary for later `sel` operations.
        for time_dim in [
            time_dim
            for time_dim in [self.time_dim, "step", "ensemble", "forecast_reference_offset"]
            if time_dim in self.standard_dims
        ]:
            # Expand the time dimension if it's of length 1 and Xarray therefore doesn't recognize it as a dimension...
            if time_dim in processed_orig_ds and time_dim not in processed_orig_ds.dims:
                processed_orig_ds = processed_orig_ds.expand_dims(time_dim)

            # ... or create it from the file name if missing entirely in the raw file
            elif time_dim not in processed_orig_ds:
                processed_orig_ds = processed_orig_ds.assign_coords(
                    {
                        time_dim: datetime.datetime.strptime(
                            re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", str(orig_file_path))[0], "%Y-%m-%d"
                        )
                    }
                )
                processed_orig_ds = processed_orig_ds.expand_dims(time_dim)

            # Also expand it for the data var!
            if time_dim in processed_orig_ds.dims and time_dim not in processed_orig_ds[self.data_var()].dims:
                processed_orig_ds[self.data_var()] = processed_orig_ds[self.data_var()].expand_dims(time_dim)

        return processed_orig_ds


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
    return np.isinf(n) or abs(n) > 1e100
