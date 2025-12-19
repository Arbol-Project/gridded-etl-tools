import os
import datetime
import pathlib
import pytest
import xarray
import shutil
import glob

from unittest.mock import Mock, patch
from ..common import (
    run_etl,
    mocked_ftp_extract_request,
    get_manager,
    clean_up_input_paths,
    patched_key,
    patched_root_stac_catalog,
    patched_zarr_json_path,
    remove_mock_output,
    remove_dask_worker_dir,
    remove_performance_report,
    remove_zarr_json,
    remove_metadata,
)

from gridded_etl_tools.utils.publish import ZarrOutputError
from gridded_etl_tools.utils.store import StoreInterface


@pytest.fixture
def create_input_directories(
    extracted_input_path,
    initial_input_path,
    initial_smaller_input_path,
    appended_input_path,
    appended_input_path_with_hole,
):
    """
    The testing directories for initial, append and insert will get created before each run
    """
    for path in (
        extracted_input_path,
        initial_input_path,
        initial_smaller_input_path,
        appended_input_path,
        appended_input_path_with_hole,
    ):
        os.makedirs(path, 0o755, exist_ok=True)


@pytest.fixture
def simulate_file_download(root, initial_input_path, initial_smaller_input_path, appended_input_path):
    """
    Copies the default input NCs into the default input paths, simulating a download of original data. Later, the input
    directories will be deleted during clean up.
    """
    shutil.copy(root / "CHIRPS25_2024-12_Nevada.nc", initial_input_path)
    shutil.copy(root / "CHIRPS25_2024-12_Nevada_subset.nc", initial_smaller_input_path)
    shutil.copy(root / "CHIRPS25_2025-01_Nevada.nc", appended_input_path)
    print("Simulated downloading input files")


@pytest.fixture
def simulate_file_download_hole(root, initial_input_path, appended_input_path_with_hole):
    """
    Copies the default input NCs into the default input paths, simulating a download of original data. Later, the input
    directories will be deleted during clean up.
    """
    shutil.copy(root / "CHIRPS25_2024-12_Nevada.nc", initial_input_path)
    shutil.copy(root / "CHIRPS25_2025-01_Nevada_with_hole.nc", appended_input_path_with_hole)
    print("Simulated downloading input files hole")


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_per_test(
    mocker,
    request,
    extracted_input_path,
    initial_input_path,
    initial_smaller_input_path,
    appended_input_path,
    appended_input_path_with_hole,
    create_input_directories,
    simulate_file_download,
    simulate_file_download_hole,
):
    """
    Call the setup functions first, in a chain ending with `simulate_file_download`.
    Next run the test in question. Finally, remove generated inputs afterwards, even if the test fails.
    """
    mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.key", patched_key)
    mocker.patch("examples.managers.chirps.CHIRPS.collection", return_value="CHIRPS_test")
    mocker.patch(
        "gridded_etl_tools.dataset_manager.DatasetManager.zarr_json_path",
        patched_zarr_json_path,
    )
    mocker.patch(
        "gridded_etl_tools.dataset_manager.DatasetManager.default_root_stac_catalog",
        patched_root_stac_catalog,
    )
    yield  # run the tests first
    # delete temp files
    remove_mock_output()
    remove_zarr_json()
    remove_dask_worker_dir()
    remove_performance_report()
    remove_metadata()
    # now clean up the various files created for each test
    clean_up_input_paths(
        extracted_input_path,
        initial_input_path,
        initial_smaller_input_path,
        appended_input_path,
        appended_input_path_with_hole,
    )


def test_extract(mocker, manager_class, test_chunks, extracted_input_path):
    """
    Test an extract of CHIRPS data.
    """
    mocker.patch("gridded_etl_tools.utils.extractor.FTPExtractor.request", mocked_ftp_extract_request)
    # Get the CHIRPS manager with rebuild set
    dm = manager_class(custom_input_path=extracted_input_path, rebuild_requested=True, store="local")
    dm.check_if_new_data = Mock(return_value=True)
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    dm.requested_dask_chunks = test_chunks
    dm.requested_zarr_chunks = test_chunks
    # run and check extract
    date_range = [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 12, 31)]
    dm.extract(date_range=date_range)
    input_files = list(dm.input_files())

    assert len(input_files) == 1
    dm.check_if_new_data.assert_called_once_with(date_range[1])


def test_initial_dry_run(mocker, manager_class, test_chunks, initial_input_path):
    """
    Test that a dry run parse of CHIRPS data does not, in fact, parse data.
    """
    # Get the CHIRPS manager with rebuild set
    dm = manager_class(custom_input_path=initial_input_path, rebuild_requested=True, dry_run=True, store="local")
    dm.store.folder = "tests"
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    dm.requested_dask_chunks = test_chunks
    dm.requested_zarr_chunks = test_chunks
    # run ETL
    dm.transform_data_on_disk()
    publish_dataset = dm.transform_dataset_in_memory()
    dm.parse(publish_dataset)
    dm.zarr_json_path().unlink(missing_ok=True)
    # Check that a path wasn't created because the dataset wasn't parsed
    assert not dm.store.has_existing


def test_initial_write_failure(manager_class, initial_input_path):
    """
    Test with a failed call to xarray.Dataset.to_zarr
    """
    dm = manager_class(custom_input_path=initial_input_path, store="local")
    dm.transform_data_on_disk()
    publish_dataset = dm.transform_dataset_in_memory()
    dm.store = Mock(spec=StoreInterface)
    dm.store.has_existing = False
    with patch("xarray.Dataset.to_zarr", side_effect=RuntimeError("Nuclear meltdown")):
        with pytest.raises(ZarrOutputError):
            dm.parse(publish_dataset)
        dm.store.write_metadata_only_v2.assert_called_once_with(update_attrs={"update_in_progress": False})


def test_initial(manager_class, initial_input_path, root):
    """
    Test a parse of CHIRPS data.
    """
    # Get the CHIRPS manager with rebuild set
    dm = run_etl(manager_class, input_path=initial_input_path, use_local_zarr_jsons=False, store="local")
    dm.zarr_json_path().unlink(missing_ok=True)
    generated_dataset = dm.store.dataset()

    # Las Vegas
    lat, lon = 36.125, -115.625

    # Validate one row of data
    output_value = (
        generated_dataset[dm.data_var]
        .sel(
            latitude=lat,
            longitude=lon,
            time=datetime.datetime(2024, 12, 12),
            method="nearest",
        )
        .values
    )

    # Pull original data for checking. The "h5netcdf" engine is specified because re-opening the netCDF file fails on
    # the GitHub actions server instance without it.
    original_dataset = xarray.open_dataset(root / "CHIRPS25_2024-12_Nevada.nc", engine="h5netcdf")
    orig_data_var = [key for key in original_dataset.data_vars][0]
    original_value = (
        original_dataset[orig_data_var].sel(latitude=lat, longitude=lon, time=datetime.datetime(2024, 12, 12)).values
    )
    assert output_value == original_value


def test_prepare_input_files(manager_class, mocker, appended_input_path):
    """
    Test that the constituent steps of prepare_input_files work, as expressed through the example CHIRPS manager
    """
    mocker.patch(
        "examples.managers.chirps.CHIRPSFinal25.local_input_path",
        return_value=appended_input_path,
    )
    dm = get_manager(manager_class, appended_input_path)
    # Test that prepare_input_files successfully expands to 31 files
    assert len(list(dm.input_files())) == 1
    dm.convert_to_lowest_common_time_denom(list(dm.input_files()), keep_originals=False)
    assert len(list(dm.input_files())) == 31
    # assert all 31 new files are NC4 files
    input_ncs = [pathlib.Path(file) for file in glob.glob(str(dm.local_input_path() / "*.nc"))]
    input_nc4s = dm.input_files()
    assert len(input_ncs) == 0
    assert len(list(input_nc4s)) == 31


def test_append_only(mocker, manager_class, test_chunks, appended_input_path, root):
    """
    Test an update of chirps data by adding new data to the end of existing data.
    """
    # Get a non-rebuild manager for testing append
    dm = run_etl(manager_class, input_path=appended_input_path, use_local_zarr_jsons=False, store="local")
    generated_dataset = dm.store.dataset()

    # Las Vegas
    lat, lon = 36.125, -115.625

    # Validate one row of data
    output_value = (
        generated_dataset[dm.data_var].sel(latitude=lat, longitude=lon, time=datetime.datetime(2025, 1, 25)).values
    )

    # Pull original data for checking. The "h5netcdf" engine is specified because re-opening the netCDF file fails on
    # the GitHub actions server instance without it.
    original_dataset = xarray.open_dataset(root / "CHIRPS25_2025-01_Nevada.nc", engine="h5netcdf")
    orig_data_var = [key for key in original_dataset.data_vars][0]
    original_value = (
        original_dataset[orig_data_var].sel(latitude=lat, longitude=lon, time=datetime.datetime(2025, 1, 25)).values
    )
    assert output_value == original_value


def test_misaligned_zarr_dask_chunks_regression(
    mocker, manager_class, initial_smaller_input_path, appended_input_path
):
    """
    Regression test that will fail if Zarr chunks span multiple Dask chunks. This can happen if the initial
    dataset is smaller than both the append dataset and the requested Dask chunks. It will trigger
    an Xarray exception.

    If the append dataset is rechunked to the maximum chunk size in `prep_update_dataset` then this problem
    is addressed. If the test fails then the most likely suspect is a problem with rechunking here.
    """
    # run initial with a dataset whose time dimension is smaller (10) than the specified dask chunks (50)
    dm = get_manager(manager_class, input_path=initial_smaller_input_path, store="local")
    # Override nan frequency defaults since the test data doesn't cover oceans, which are NaNs in CHIRPS
    dm.expected_nan_frequency = 0.02
    dm.transform_data_on_disk()
    publish_dataset = dm.transform_dataset_in_memory()
    dm.parse(publish_dataset)
    dm.publish_metadata()

    # run initial with a dataset whose time dimension is smaller (10) than the specified dask chunks (50)
    dm = get_manager(manager_class, input_path=appended_input_path, store="local")
    # Override nan frequency defaults since the test data doesn't cover oceans, which are NaNs in CHIRPS
    dm.expected_nan_frequency = 0.02
    # run the ETL
    dm.transform_data_on_disk()
    publish_dataset = dm.transform_dataset_in_memory()
    dm.parse(publish_dataset)
    dm.publish_metadata()


def test_bad_append(
    mocker,
    manager_class,
    test_chunks,
    initial_input_path,
    appended_input_path_with_hole,
):
    """
    Test an update of chirps data by adding new data to the end of existing data.
    """
    # FIRST parse the initial dataset
    run_etl(manager_class, input_path=initial_input_path, use_local_zarr_jsons=False, store="local")

    # NOW try to parse a bad append
    # Get a non-rebuild manager for testing appended_input_path_with_hole
    dm = get_manager(manager_class, input_path=appended_input_path_with_hole, store="local")
    dm.zarr_chunks = {}
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    dm.requested_dask_chunks = test_chunks
    dm.requested_zarr_chunks = test_chunks
    # run ETL
    dm.transform_data_on_disk()
    publish_dataset = dm.transform_dataset_in_memory()
    with pytest.raises(IndexError):
        dm.parse(publish_dataset)
