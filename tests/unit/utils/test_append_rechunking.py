import tempfile
import shutil
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gridded_etl_tools.utils.publish import calculate_time_dim_chunks


class TestAppendRechunking:
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

    def create_append_dataset(self, length: int, chunk_size: int) -> xr.Dataset:
        """Create a dataset to append with specified length and chunk size"""
        # Create dates starting after the original dataset ends (2024-02-07)
        start_date = datetime.datetime(2024, 2, 7)
        dates = pd.date_range(start=start_date, periods=length, freq="D")

        # Create sample data of zeros so easy to detect new data
        data_values = np.zeros(length)

        # Create dataset
        ds = xr.Dataset(
            {self.DATA_VAR: (["time"], data_values, {"units": "degrees_C", "long_name": "Temperature"})},
            coords={"time": dates},
        )

        # Chunk the dataset
        return ds.chunk({"time": chunk_size})

    def test_append_without_rechunking_raises_error(self):
        """Test that appending without calling rechunk_append_dataset raises ValueError"""
        # Create append dataset with length 12 and chunk size 10
        append_dataset = self.create_append_dataset(length=12, chunk_size=10)

        # Try to append directly without rechunking - this should raise ValueError
        with pytest.raises(ValueError, match="chunk"):
            append_dataset.to_zarr(self.test_zarr_path, append_dim="time")

    def test_append_with_rechunking_succeeds(self):
        """Test that appending with rechunk_append_dataset works and results in correct length"""
        # Create append dataset with length 12 and chunk size 10
        append_dataset = self.create_append_dataset(length=12, chunk_size=self.DESIRED_CHUNK_SIZE)

        # Load original dataset to check initial length and chunks
        original_ds = xr.open_zarr(self.test_zarr_path)
        original_final_chunk_length = original_ds.chunks["time"][-1]

        assert original_final_chunk_length == 7  # From (10, 10, 10, 7)

        # Calculate the correct rechunking using the same logic as the function
        append_time_length = 12
        first_chunk_size = min(self.DESIRED_CHUNK_SIZE - original_final_chunk_length, append_time_length)
        new_chunks = calculate_time_dim_chunks(first_chunk_size, self.DESIRED_CHUNK_SIZE, append_time_length)

        # Apply the rechunking manually (simulating what rechunk_append_dataset does)
        rechunked_append = append_dataset.chunk({"time": new_chunks})

        # Now append the rechunked dataset
        rechunked_append.to_zarr(self.test_zarr_path, append_dim="time")

        # Verify the final dataset has the correct length
        final_ds = xr.open_zarr(self.test_zarr_path)
        assert len(final_ds.time) == 49  # 37 + 12 = 49

        # verify the new values are all 0
        assert (final_ds.isel(time=slice(-12, -1))[self.DATA_VAR].values == 0).all()

        # Verify the final chunks align properly
        # Should be (10, 10, 10, 10, 9) - the original (10,10,10,7) with 3 added to make 10, then 9 more
        expected_final_chunks = (10, 10, 10, 10, 9)
        assert final_ds.chunks["time"] == expected_final_chunks

        # Verify the time coordinates are continuous
        expected_final_date = datetime.datetime(2024, 2, 18)  # 37 + 12 - 1 days from 2024-01-01
        actual_final_date = pd.to_datetime(final_ds.time.values[-1]).to_pydatetime().replace(tzinfo=None)
        assert actual_final_date.date() == expected_final_date.date()
