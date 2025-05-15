import pandas as pd
import xarray as xr


def fill_in_missing_time_steps(dataset: xr.Dataset, freq: str, time_dim: str = "time") -> xr.Dataset:
    """Fills in a dataset that is missing coords in the time dimension with NaNs

    Args:
        dataset (xr.Dataset): Input dataset
        freq (str): Desired frequency of output, formatted according to
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        time_dim (str, optional): time dimension. Defaults to "time".

    Returns:
        xr.Dataset: Dataset without missing timesteps, with ones that were missing now filled with NaNs
    """
    start_time = dataset[time_dim].min().values
    end_time = dataset[time_dim].max().values

    full_time_index = pd.date_range(start=start_time, end=end_time, freq=freq)

    return dataset.reindex(time=full_time_index)
