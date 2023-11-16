import pytest
import copy

import pandas as pd
import numpy as np
import xarray as xr

from gridded_etl_tools.dataset_manager import DatasetManager
from ..common import get_manager, patched_update_cadence_bounds, initial


def test_pre_parse_quality_check(mocker, manager_class: DatasetManager, fake_original_dataset: xr.Dataset):
    """
    Test that the pre-parse quality check method waves through good data
    and fails as anticipated with bad data of specific types
    """
    # prepare a dataset manager
    dm = get_manager(manager_class)
    fake_original_dataset.data.encoding["units"] = "cubits"
    # Test that a dataset with out-of-order times fails
    out_of_order_ds = fake_original_dataset.copy()
    out_of_order_ds = out_of_order_ds.assign_coords({"time": np.roll(out_of_order_ds.time.values, 1)})
    with pytest.raises(IndexError):
        dm.pre_parse_quality_check(out_of_order_ds)
    # Test that a dataset with extreme values fails
    mocker.patch(
        "gridded_etl_tools.utils.convenience.Convenience.extreme_values_by_unit",
        return_value={"cubits": (-500, 500)},
        new_callable=mocker.PropertyMock,
    )
    extreme_vals_ds = copy.deepcopy(fake_original_dataset)
    extreme_vals_ds.data.values[:] = 1_000_000
    with pytest.raises(ValueError):
        dm.pre_parse_quality_check(extreme_vals_ds)
    # Test that a dataset with NaN values fails
    nan_vals_ds = copy.deepcopy(fake_original_dataset)
    nan_vals_ds.data.values[:] = np.nan
    with pytest.raises(ValueError):
        dm.pre_parse_quality_check(nan_vals_ds)
    # Test that a parse fails on mismatched data var encoding
    mocker.patch("gridded_etl_tools.utils.attributes.Attributes.data_var_dtype", return_value="<f4")
    # fake_original_dataset["data"] = fake_original_dataset["data"].astype('<f8')
    fake_original_dataset.data.encoding["dtype"] = "<f8"
    # fake_original_dataset.data.encoding["units"] = 'cubits'
    with pytest.raises(TypeError):
        dm.pre_parse_quality_check(fake_original_dataset)


def test_update_quality_check(mocker, manager_class: DatasetManager, fake_original_dataset: xr.Dataset):
    """
    Test that the pre-parse quality check method waves through good data
    and fails as anticipated with bad data of specific types
    """
    # prepare a dataset manager
    dm = get_manager(manager_class)
    # Test that a parse succeeds when append data is contiguous with existing data
    insert_times = []
    append_times = pd.date_range(start="2022-02-01", end="2023-02-15", freq="1D")
    assert not dm.update_quality_check(fake_original_dataset, insert_times=insert_times, append_times=append_times)
    # Test that a parse succeeds when insert or append data exists
    insert_times = pd.date_range(start="2021-10-01", end="2021-10-15", freq="1D")
    append_times = []
    assert not dm.update_quality_check(fake_original_dataset, insert_times=insert_times, append_times=append_times)
    # Test that a parse fails when append data is not contiguous with existing data
    insert_times = []
    append_times = pd.date_range(start="2022-02-02", end="2023-02-15", freq="1D")
    with pytest.raises(IndexError):
        dm.update_quality_check(fake_original_dataset, insert_times=insert_times, append_times=append_times)
    # Test that a parse fails when there's no data to insert or append
    insert_times = []
    append_times = []
    with pytest.raises(IndexError):
        dm.update_quality_check(fake_original_dataset, insert_times=insert_times, append_times=append_times)
    # Test that a parse fails when the new data is before the existing data
    insert_times = []
    append_times = pd.date_range(start="2021-08-01", end="2021-08-15", freq="1D")
    with pytest.raises(IndexError):
        dm.update_quality_check(fake_original_dataset, insert_times=insert_times, append_times=append_times)


def test_are_times_in_expected_order(mocker, manager_class: DatasetManager):
    """
    Test that the check for non-contiguous times successfully catches bad times
    while letting anticipated irregular times pass
    """
    # prepare a dataset manager
    dm = get_manager(manager_class)
    # Check a set of contiguous times
    contig = pd.date_range(start="2023-03-01", end="2023-03-15", freq="1D")
    expected_delta = contig[1] - contig[0]
    assert dm.are_times_in_expected_order(contig, expected_delta=expected_delta)
    # Check a single time -- one good, one not
    check1 = [contig[0], contig[1]]
    check2 = [contig[0], contig[2]]
    assert dm.are_times_in_expected_order(check1, expected_delta=expected_delta)
    assert not dm.are_times_in_expected_order(check2, expected_delta=expected_delta)
    # Check a set of times that skips a day
    week_ahead_dt = contig[-1] + pd.Timedelta(days=7)
    week_gap = contig.union([week_ahead_dt])
    assert not dm.are_times_in_expected_order(week_gap, expected_delta=expected_delta)
    # Check a set of times that's out of order
    week_behind_dt = contig[0] - pd.Timedelta(days=7)
    week_gap = contig.union([week_behind_dt])
    assert not dm.are_times_in_expected_order(week_gap, expected_delta=expected_delta)
    # Check a set of times that's badly out of order
    out_of_order = [contig[1], contig[2], contig[0], contig[12], contig[3]]
    assert not dm.are_times_in_expected_order(out_of_order, expected_delta=expected_delta)
    # Check that irregular cadences pass
    mocker.patch("gridded_etl_tools.utils.attributes.Attributes.update_cadence_bounds", patched_update_cadence_bounds)
    three_and_four_day_updates = [contig[0], contig[3], contig[6], contig[10]]
    assert dm.are_times_in_expected_order(three_and_four_day_updates, expected_delta=expected_delta)
    # Check that ranges outside the irregular cadence still fail
    five_day_updates = [contig[0], contig[3], contig[6], contig[11], contig[14]]
    assert not dm.are_times_in_expected_order(five_day_updates, expected_delta=expected_delta)

# def test_post_parse_quality_check(mocker, capfd, initial_input_path, manager_class: DatasetManager):
#     """
#     Test that the post-parse quality check method waves through good data
#     and fails as anticipated with bad data
#     """
#     # Prepare a dataset manager
#     dm = initial(manager_class, initial_input_path)
#     # Local data
#     ### Approves aligned values
#     ### Rejects misaligned values
#     ### Skips dataset without time
#     dm.protocol = 'file'
#     dm.post_parse_quality_check(checks=5)
#     # Remote data
#     ### Approves aligned values
#     ### Rejects misaligned values
#     ### Skips dataset without time
#     dm.protocol = 'remote'
#     dm.post_parse_quality_check(checks=5)
#     # Skipping the QC
#     dm.skip_post_parse_qc = True
#     dm.post_parse_quality_check(checks=5)
#     out, _ = capfd.readouterr()
#     assert "Skipping post-parse quality check" in out
