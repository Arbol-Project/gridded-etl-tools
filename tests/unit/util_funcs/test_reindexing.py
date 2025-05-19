import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gridded_etl_tools.util_funcs.reindexing import fill_in_missing_time_steps

START_TIME, END_TIME = datetime.datetime(2020, 1, 1), datetime.datetime(2025, 12, 31)
MISSING_TIME = np.datetime64(datetime.datetime(2023, 5, 19))
OFF_FREQUENCY_TIME = np.datetime64(datetime.datetime(2022, 4, 13, 13))


@pytest.fixture
def holey_dataset():
    times = pd.date_range(START_TIME, END_TIME).values
    times = times[times != MISSING_TIME]
    times = np.concatenate([times, np.array([OFF_FREQUENCY_TIME])])
    times.sort()

    ds = xr.Dataset(data_vars=dict(data=(["time"], np.zeros(len(times)))), coords={"time": times})

    # check we set up fixture correctly
    assert MISSING_TIME not in ds["time"].values
    assert OFF_FREQUENCY_TIME in ds["time"].values

    return ds


def test_fill_in_missing_time_steps(holey_dataset):

    filled_in_dataset = fill_in_missing_time_steps(holey_dataset, "D")

    assert MISSING_TIME in filled_in_dataset["time"].values
    assert OFF_FREQUENCY_TIME not in filled_in_dataset["time"].values

    assert len(filled_in_dataset["time"]) == len(pd.date_range(START_TIME, END_TIME))
    assert np.isnan(filled_in_dataset.sel(time=MISSING_TIME)["data"].values).all()
