import os
import datetime
import pathlib
import pytest
import xarray
import shutil
import glob

from unittest.mock import Mock

from ..common import (
    clean_up_input_paths,
    empty_ipns_publish,
    get_manager,
    offline_ipns_publish,
    patched_key,
    patched_root_stac_catalog,
    patched_zarr_json_path,
    remove_dask_worker_dir,
    remove_performance_report,
    remove_zarr_json,
)


@pytest.fixture
def create_input_directories(
    extracted_input_path, initial_input_path, appended_input_path, appended_input_path_with_hole, qc_input_path
):
    """
    The testing directories for initial, append and insert will get created before each run
    """
    for path in (
        extracted_input_path,
        initial_input_path,
        appended_input_path,
        appended_input_path_with_hole,
        qc_input_path,
    ):
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


@pytest.fixture
def simulate_file_download_hole(root, initial_input_path, appended_input_path_with_hole):
    """
    Copies the default input NCs into the default input paths, simulating a download of original data. Later, the input
    directories will be deleted during clean up.
    """
    # for chirps_init_fil in root.glob("*initial*"):
    #     shutil.copy(chirps_init_fil, initial_input_path)
    shutil.copy(root / "chirps_initial_dataset.nc", initial_input_path)
    shutil.copy(root / "chirps_append_subset_with_hole.nc", appended_input_path_with_hole)
    print("Simulated downloading input files hole")


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown_per_test(
    mocker,
    request,
    extracted_input_path,
    initial_input_path,
    appended_input_path,
    appended_input_path_with_hole,
    create_heads_file_for_testing,
    create_input_directories,
    simulate_file_download,
    simulate_file_download_hole,
):
    """
    Call the setup functions first, in a chain ending with `simulate_file_download`.
    Next run the test in question. Finally, remove generated inputs afterwards, even if the test fails.
    """
    # Force ipns_publish to use offline mode to make tests run faster
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
    mocker.patch(
        "gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish",
        empty_ipns_publish,
    )
    yield  # run the tests first
    # delete temp files
    remove_zarr_json()
    remove_dask_worker_dir()
    remove_performance_report()
    # now clean up the various files created for each test
    clean_up_input_paths(extracted_input_path, initial_input_path, appended_input_path, appended_input_path_with_hole)


@pytest.fixture(scope="module", autouse=True)
def teardown_module(request, heads_path):
    """
    Remove the heads file at the end of all tests
    """

    def test_clean():
        os.remove(heads_path)
        print(f"Cleaned up {heads_path}")

    request.addfinalizer(test_clean)


def test_extract(mocker, manager_class, heads_path, test_chunks, extracted_input_path):
    """
    Test a parse of CHIRPS data. This function is run automatically by pytest because the function name starts with
    "test_".
    """
    # Get the CHIRPS manager with rebuild set
    manager = manager_class(custom_input_path=extracted_input_path, rebuild_requested=True, store="ipld")
    manager.HASH_HEADS_PATH = heads_path
    # Remove IPNS publish mocker on the first run of the dataset, so it lives as "dataset_test" in your IPNS registry
    if manager.key() not in manager.ipns_key_list():  # pragma NO COVER
        mocker.patch(
            "gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish",
            offline_ipns_publish,
        )
    manager.check_if_new_data = Mock(return_value=True)
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    manager.requested_dask_chunks = test_chunks
    manager.requested_zarr_chunks = test_chunks
    # run and check extract
    date_range = [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 12, 31)]
    manager.extract(date_range=date_range)
    input_files = list(manager.input_files())

    assert len(input_files) == 1
    manager.check_if_new_data.assert_called_once_with(date_range[1])


def test_initial_dry_run(request, mocker, manager_class, heads_path, test_chunks, initial_input_path, root):
    """
    Test that a dry run parse of CHIRPS data does not, in fact, parse data.
    """
    # Get the CHIRPS manager with rebuild set
    manager = manager_class(custom_input_path=initial_input_path, rebuild_requested=True, dry_run=True, store="ipld")
    manager.HASH_HEADS_PATH = heads_path
    # Remove IPNS publish mocker on the first run of the dataset, so it lives as "dataset_test" in your IPNS registry
    if manager.key() not in manager.ipns_key_list():  # pragma NO COVER
        mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish", offline_ipns_publish)
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    manager.requested_dask_chunks = test_chunks
    manager.requested_zarr_chunks = test_chunks
    # run ETL
    manager.transform()
    manager.parse()
    manager.zarr_json_path().unlink(missing_ok=True)
    # Check that a hash wasn't created because the dataset wasn't parsed
    with pytest.raises(FileNotFoundError):
        manager.zarr_hash_to_dataset(manager.latest_hash())


def test_initial(request, mocker, manager_class, heads_path, test_chunks, initial_input_path, root):
    """
    Test a parse of CHIRPS data. This function is run automatically by pytest because the function name starts with
    "test_".
    """
    # Get the CHIRPS manager with rebuild set
    manager = manager_class(custom_input_path=initial_input_path, rebuild_requested=True, store="ipld")
    manager.HASH_HEADS_PATH = heads_path
    # Remove IPNS publish mocker on the first run of the dataset, so it lives as "dataset_test" in your IPNS registry
    if manager.key() not in manager.ipns_key_list():  # pragma NO COVER
        mocker.patch(
            "gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish",
            offline_ipns_publish,
        )
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    manager.requested_dask_chunks = test_chunks
    manager.requested_zarr_chunks = test_chunks
    # run ETL
    manager.transform()
    manager.parse()
    manager.publish_metadata()
    manager.zarr_json_path().unlink(missing_ok=True)
    # Open the head with ipldstore + xarray.open_zarr and compare two data points with the same data points in a local
    # GRIB file
    generated_dataset = manager.zarr_hash_to_dataset(manager.latest_hash())
    lat, lon = 14.625, -91.375
    # Validate one row of data
    output_value = (
        generated_dataset[manager.data_var()]
        .sel(
            latitude=lat,
            longitude=lon,
            time=datetime.datetime(2003, 5, 12),
            method="nearest",
        )
        .values
    )
    original_dataset = xarray.open_dataset(root / "chirps_initial_dataset.nc", engine="netcdf4")
    orig_data_var = [key for key in original_dataset.data_vars][0]
    original_value = (
        original_dataset[orig_data_var].sel(latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 12)).values
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
    # Test that prepare_input_files successfully expands 2 files to 32
    assert len(list(dm.input_files())) == 2
    dm.convert_to_lowest_common_time_denom(list(dm.input_files()), keep_originals=False)
    assert len(list(dm.input_files())) == 32
    # assert all 32 new files are NC4 files
    input_ncs = [pathlib.Path(file) for file in glob.glob(str(dm.local_input_path() / "*.nc"))]
    input_nc4s = dm.input_files()
    assert len(input_ncs) == 0
    assert len(list(input_nc4s)) == 32


def test_append_only(mocker, request, manager_class, heads_path, test_chunks, appended_input_path, root):
    """
    Test an update of chirps data by adding new data to the end of existing data.
    """
    # Get a non-rebuild manager for testing append
    manager = manager_class(custom_input_path=appended_input_path, store="ipld")
    manager.HASH_HEADS_PATH = heads_path
    manager.zarr_chunks = {}
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    manager.requested_dask_chunks = test_chunks
    manager.requested_zarr_chunks = test_chunks
    # run ETL
    manager.transform()
    manager.parse()
    manager.publish_metadata()
    # Open the head with ipldstore + xarray.open_zarr and compare two data points with the same data points in a local
    # GRIB file
    generated_dataset = manager.zarr_hash_to_dataset(manager.dataset_hash)
    lat, lon = 14.625, -91.375
    # Validate one row of data
    output_value = (
        generated_dataset[manager.data_var()]
        .sel(latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 25))
        .values
    )
    original_dataset = xarray.open_dataset(root / "chirps_append_subset_0.nc", engine="netcdf4")
    orig_data_var = [key for key in original_dataset.data_vars][0]
    original_value = (
        original_dataset[orig_data_var].sel(latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 25)).values
    )
    assert output_value == original_value


def test_bad_append(
    mocker,
    request,
    manager_class,
    heads_path,
    test_chunks,
    appended_input_path_with_hole,
    root,
):
    """
    Test an update of chirps data by adding new data to the end of existing data.
    """
    # Get a non-rebuild manager for testing append
    manager = manager_class(custom_input_path=appended_input_path_with_hole, store="ipld")
    manager.HASH_HEADS_PATH = heads_path
    manager.zarr_chunks = {}
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    manager.requested_dask_chunks = test_chunks
    manager.requested_zarr_chunks = test_chunks
    # run ETL
    manager.transform()
    with pytest.raises(IndexError):
        manager.parse()


def test_metadata(manager_class, heads_path):
    """
    Test an update of CHIRPS metadata.

    This function will only work after the test dataset's metadata has been populated into IPFS and the IPNS key list.
    """
    # Get a non-rebuild manager for testing metadata creation
    manager = manager_class(store="ipld")
    manager.HASH_HEADS_PATH = heads_path
    manager.publish_metadata()
    assert manager.load_stac_metadata() != {}
