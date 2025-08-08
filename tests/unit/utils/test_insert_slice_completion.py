import tempfile
import shutil
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gridded_etl_tools.utils.publish import complete_insert_slice


class TestInsertSliceCompletion:
    """Tests for the rechunk_append_dataset functionality"""

    DESIRED_CHUNK_SIZE = 10
    ORIGINAL_DATASET_START = datetime.datetime(2024, 1, 1)
    ORIGINAL_DATASET_END = datetime.datetime(2024, 2, 6)
    DATA_VAR = "temperature"

    def setup_method(self):
        """Set up test fixtures for each test"""
        # Create temporary directory for test zarr files
        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)

        # Copy the original test zarr to a temporary location
        original_zarr_path = Path(__file__).parents[1] / "inputs" / "timeseries_data.zarr"
        self.test_zarr_path = temp_path / "test_timeseries.zarr"
        shutil.copytree(original_zarr_path, self.test_zarr_path)

    def teardown_method(self):
        """Clean up test fixtures after each test"""
        shutil.rmtree(self.temp_dir)

    def create_insert_dataset(self, start_date: datetime.datetime, length: int) -> xr.Dataset:
        """Create a dataset to insert with specified length and chunk size"""

        # Create dates fully contained in the original dataset ends (2024-02-07)
        assert start_date >= self.ORIGINAL_DATASET_START
        assert start_date + datetime.timedelta(days=length) <= self.ORIGINAL_DATASET_END
        dates = pd.date_range(start=start_date, periods=length, freq="D")

        # Create sample data of zeros so easy to detect new data
        data_values = np.zeros(length)

        # Create dataset
        ds = xr.Dataset(
            {self.DATA_VAR: (["time"], data_values, {"units": "degrees_C", "long_name": "Temperature"})},
            coords={"time": dates},
        )

        # Chunk the dataset
        return ds.chunk({"time": self.DESIRED_CHUNK_SIZE})

    def test_insert_without_completion_raises_error(self):
        insert_start_time = datetime.datetime(2024, 1, 4)
        insert_length = 8
        insert_dataset = self.create_insert_dataset(insert_start_time, insert_length)

        region_start = (insert_start_time - self.ORIGINAL_DATASET_START).days
        region_end = region_start + insert_length
        region = (region_start, region_end)
        with pytest.raises(ValueError, match="chunk"):
            insert_dataset.to_zarr(self.test_zarr_path, region={"time": slice(*region)})

    def test_insert_with_completion_succeeds(self):
        insert_start_time = datetime.datetime(2024, 1, 4)
        insert_length = 8
        insert_dataset = self.create_insert_dataset(insert_start_time, insert_length)

        region_start = (insert_start_time - datetime.datetime(2024, 1, 1)).days
        region_end = region_start + insert_length
        region = (region_start, region_end)

        original_dataset = xr.open_zarr(self.test_zarr_path)

        complete_slice, chunks_region = complete_insert_slice(insert_dataset, original_dataset, region, 10, "time")
        complete_slice.to_zarr(self.test_zarr_path, region={"time": slice(*chunks_region)})

        full_dataset_after_insert = xr.open_zarr(self.test_zarr_path)

        # dimensions should not have changed
        assert (original_dataset.time.values == full_dataset_after_insert.time.values).all()

        slice_before_insert = slice(0, region_start)
        slice_after_insert = slice(region_end, -1)

        # insert values should be updated
        assert (full_dataset_after_insert.isel(time=slice(*region))[self.DATA_VAR].values == 0).all()

        # values located before and after insert should be the same
        xr.testing.assert_equal(
            full_dataset_after_insert.isel(time=slice_before_insert),
            original_dataset.isel(time=slice_before_insert),
        )
        xr.testing.assert_equal(
            full_dataset_after_insert.isel(time=slice_after_insert),
            original_dataset.isel(time=slice_after_insert),
        )
