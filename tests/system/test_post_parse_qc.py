import logging
import os
import pytest
import shutil
import random
import numpy as np

from unittest.mock import Mock
from ..common import (
    run_etl,
    clean_up_input_paths,
    empty_ipns_publish,
    patched_output_root,
    patched_key,
    patched_root_stac_catalog,
    patched_zarr_json_path,
    remove_mock_output,
    remove_dask_worker_dir,
    remove_performance_report,
    remove_zarr_json,
    original_ds_normal,
    original_ds_single_time,
    original_ds_bad_data,
    original_ds_no_time_dim,
    original_ds_no_time_at_all,
    original_ds_no_time_dim_in_data_var,
    original_ds_random,
    original_ds_null,
    nc4_input_files,
    json_input_files,
)


@pytest.fixture
def create_input_directories(initial_input_path, qc_input_path, appended_input_path):
    """
    The testing directories for initial, append and insert will get created before each run
    """
    for path in (initial_input_path, qc_input_path, appended_input_path):
        if not path.exists():
            os.makedirs(path, 0o755, True)
            print(f"Created {path} for testing")
        else:
            print(f"Found existing {path}")


@pytest.fixture
def simulate_file_download(root, initial_input_path, appended_input_path, qc_input_path):
    """
    Copies the default input NCs into the default input paths, simulating a download of original data. Later, the input
    directories will be deleted during clean up.
    """
    # for chirps_init_fil in root.glob("*initial*"):
    #     shutil.copy(chirps_init_fil, initial_input_path)
    shutil.copy(root / "chirps_initial_dataset.nc", initial_input_path)
    shutil.copy(root / "chirps_append_subset_0.nc", appended_input_path)
    shutil.copy(root / "chirps_append_subset_1.nc", appended_input_path)
    shutil.copy(root / "chirps_qc_test_2003041100.nc", qc_input_path)
    print("Simulated downloading input files")


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_per_test(
    mocker,
    request,
    initial_input_path,
    appended_input_path,
    qc_input_path,
    create_heads_file_for_testing,
    create_input_directories,
    simulate_file_download,
):
    """
    Call the setup functions first, in a chain ending with `simulate_file_download`.
    Next run the test in question. Finally, remove generated inputs afterwards, even if the test fails.
    """
    # Force ipns_publish to use offline mode to make tests run faster
    mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.key", patched_key)
    mocker.patch("gridded_etl_tools.utils.publish.Publish.pre_parse_quality_check", Mock())  # speeds things up
    mocker.patch("examples.managers.chirps.CHIRPS.collection", return_value="CHIRPS_test")
    mocker.patch(
        "gridded_etl_tools.dataset_manager.DatasetManager.zarr_json_path",
        patched_zarr_json_path,
    )
    mocker.patch(
        "gridded_etl_tools.dataset_manager.DatasetManager.default_root_stac_catalog",
        patched_root_stac_catalog,
    )
    mocker.patch(
        "gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish",
        empty_ipns_publish,
    )
    # Mock the root output directory name, so no existing data will be overwritten and it can be easily cleaned up
    mocker.patch(
        "gridded_etl_tools.utils.convenience.Convenience.output_root",
        patched_output_root,
    )

    yield  # run the tests first
    # delete temp files
    remove_mock_output()
    remove_zarr_json()
    remove_dask_worker_dir()
    remove_performance_report()
    # now clean up the various files created for each test
    clean_up_input_paths(initial_input_path, appended_input_path, qc_input_path)


@pytest.fixture(scope="module", autouse=True)
def teardown_module(request, heads_path):
    """
    Remove the heads file at the end of all tests
    """

    def test_clean():
        os.remove(heads_path)
        print(f"Cleaned up {heads_path}")

    request.addfinalizer(test_clean)


def test_post_parse_quality_check(mocker, manager_class, caplog, initial_input_path):
    """
    Test that the post-parse quality check method waves through good data
    and fails as anticipated with bad data
    """
    caplog.set_level(logging.INFO)
    # Prepare a dataset manager
    dm = run_etl(manager_class, input_path=initial_input_path)
    # Approves aligned values
    dm.post_parse_quality_check(checks=5)
    dm.post_parse_quality_check(checks=5)
    # Rejects misaligned values
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_bad_data)
    with pytest.raises(ValueError):
        dm.post_parse_quality_check(checks=5)
    # Skipping the QC
    dm.skip_post_parse_qc = True
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_normal)
    dm.post_parse_quality_check(checks=5)
    assert "Skipping post-parse quality check" in caplog.text


def test_post_parse_quality_check_single_datetime(mocker, manager_class, caplog, initial_input_path):
    """
    Test that the post-parse quality check method waves through good data
    and fails as anticipated with bad data
    """
    # Prepare a dataset manager
    dm = run_etl(manager_class, input_path=initial_input_path)
    # Runs without issue for original datasets of length 1 in the time dimension
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_single_time)
    dm.post_parse_quality_check(checks=5)


def test_raw_file_to_dataset_local(mocker, manager_class, initial_input_path, appended_input_path):
    """
    Test that the raw_file_to_dataset function correctly loads in datasets as anticipated for
    local and remote files alike
    """
    # Parse a dataset manager initially, and then for an update
    dm = run_etl(manager_class, input_path=initial_input_path, use_local_zarr_jsons=False)
    dm = run_etl(manager_class, input_path=appended_input_path, use_local_zarr_jsons=False)
    # Local data
    dm.protocol = "file"
    dm.input_files = Mock(return_value=nc4_input_files(dm))
    input_files = nc4_input_files(dm)
    assert dm.raw_file_to_dataset(random.choice(input_files))


def test_raw_file_to_dataset_s3(mocker, manager_class, initial_input_path, appended_input_path):
    """
    Test that the raw_file_to_dataset function correctly loads in datasets as anticipated for
    local and remote files alike
    """
    # Parse a dataset manager initially, and then for an update
    dm = run_etl(manager_class, input_path=initial_input_path, use_local_zarr_jsons=False)
    dm = run_etl(manager_class, input_path=appended_input_path, use_local_zarr_jsons=True)
    # Remote data
    dm.protocol = "s3"
    input_files = json_input_files(dm)
    assert dm.raw_file_to_dataset(random.choice(input_files))


def test_reformat_orig_ds_no_time_dim(mocker, manager_class, initial_input_path, qc_input_path):
    """
    Test that the original dataset is correctly reformatted when fed data with time in the coordinates
    but in the dimensions
    """
    # Prepare a dataset manager
    dm = run_etl(manager_class, input_path=initial_input_path, use_local_zarr_jsons=False)
    dm = run_etl(manager_class, input_path=qc_input_path, use_local_zarr_jsons=False)
    original_files = list(dm.input_files())
    # Populates time dimension from filename if missing dataset
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_no_time_dim)
    raw_ds = dm.raw_file_to_dataset(random.choice(original_files))
    orig_ds = dm.reformat_orig_ds(raw_ds, "irrelevant/path")
    assert "time" in orig_ds.dims


def test_reformat_orig_ds_no_time_at_all(mocker, manager_class, initial_input_path, qc_input_path):
    """
    Test that the original dataset is correctly reformatted when fed data without a time dimension
    or coordinate
    """
    # Prepare a dataset manager
    dm = run_etl(manager_class, input_path=initial_input_path, use_local_zarr_jsons=False)
    dm = run_etl(manager_class, input_path=qc_input_path, use_local_zarr_jsons=False)
    original_files = list(dm.input_files())
    # Populates time dimension from filename if missing dataset
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_no_time_at_all)
    raw_ds = dm.raw_file_to_dataset(random.choice(original_files))
    orig_ds = dm.reformat_orig_ds(raw_ds, "precip_1700-01-01.nc4")
    assert "time" in orig_ds
    assert orig_ds["time"].values[0] == np.datetime64("1700-01-01")


def test_reformat_orig_ds_time_dim_not_in_data_var(mocker, manager_class, initial_input_path, qc_input_path):
    """
    Test that the original dataset is correctly reformatted when fed data with time removed from the data var
    dimensions
    """
    # Prepare a dataset manager
    dm = run_etl(manager_class, input_path=initial_input_path, use_local_zarr_jsons=False)
    dm = run_etl(manager_class, input_path=qc_input_path, use_local_zarr_jsons=False)
    original_files = list(dm.input_files())
    # Populates time dimension from filename if missing dataset
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_no_time_dim_in_data_var)
    raw_ds = dm.raw_file_to_dataset(random.choice(original_files))
    orig_ds = dm.reformat_orig_ds(raw_ds, "irrelevant/path")
    assert "time" in orig_ds[dm.data_var].dims


def test_check_values(mocker, manager_class, initial_input_path, appended_input_path):
    """
    Test that the values check exits as anticipated when given an original dataset whose
    time dimension doesn't correspond to the production dataset
    """
    # Prepare a dataset manager
    dm = run_etl(manager_class, input_path=initial_input_path, use_local_zarr_jsons=False)
    dm = run_etl(manager_class, input_path=appended_input_path, use_local_zarr_jsons=False)
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_normal)
    prod_ds = dm.get_prod_update_ds()
    original_files = list(dm.filter_search_space(prod_ds))
    orig_ds = dm.raw_file_to_dataset(random.choice(original_files))

    # pass if values match
    dm.check_written_value(orig_ds, prod_ds)

    # raise ValueError if one dataset doesn't match the other
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_random)
    orig_ds = dm.raw_file_to_dataset(random.choice(original_files))
    with pytest.raises(ValueError):
        dm.check_written_value(orig_ds, prod_ds)

    # raise ValueError if one dataset is all NaNs
    mocker.patch("gridded_etl_tools.utils.publish.Publish.raw_file_to_dataset", original_ds_null)
    orig_ds = dm.raw_file_to_dataset(random.choice(original_files))
    with pytest.raises(ValueError):
        dm.check_written_value(orig_ds, prod_ds)
