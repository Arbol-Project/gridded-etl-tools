### [test_chirps.py]
###
### included automatically in the run. The test functions check if CHIRPS data is being generated
### correctly by the current state of the repository using test data saved in data/ that is
### checked into the repository.
###
### With a running IPFS daemon and Python virtual environment set up as described in
### doc/set_up_python_virtual_environment.md (including all CHIRPS specific dependencies), the tests
### should complete successfully from the tests/ directory.
###
###     $ pytest CHIRPS/
###
### Or just
###
###     $ pytest
###
### When `pytest -s` is run from the root tests/ directory, the test functions in this file will be
### The test will generate a heads.json and input directories in the root defined in ./conftest.py. The input
### NCs are already checked into that root. These NCs are copied into the generated input directories to
### simulate downloading them from a source. The input directories are passed into the CHIRPS instance. During
### the course of running the manager, hourly files are generated in the input directories and originals are
### copied to the "_originals" directory. After the test completes, the "teardown_module" function runs automatically
### to erase everything that was generated (everything in the default root except for the input NCs).


import os
import datetime
import pytest
import xarray
import shutil
import psutil
import multiprocessing

from ..common import *  # import local functions common to all pytests


@pytest.fixture
def create_input_directories(initial_input_path, appended_input_path):
    """
    The testing directories for initial, append and insert will get created before each run
    """
    for path in (initial_input_path, appended_input_path):
        if not path.exists():
            os.makedirs(path, 0o755, True)
            print(f"Created {path} for testing")
        else:
            print(f"Found existing {path}")


@pytest.fixture
def simulate_file_download(root, initial_input_path, appended_input_path):
    """
    Copies the default input NCs into the default input paths, simulating a download of original data. Later, the input directories will be
    deleted during clean up.
    """
    # for chirps_init_fil in root.glob("*initial*"):
    #     shutil.copy(chirps_init_fil, initial_input_path)
    shutil.copy(root / "chirps_initial_dataset.nc", initial_input_path)
    shutil.copy(root / "chirps_append_subset_0.nc", appended_input_path)
    shutil.copy(root / "chirps_append_subset_1.nc", appended_input_path)
    print("Simulated downloading input files")


@pytest.fixture(scope='function', autouse=True)
def setup_and_teardown_per_test(mocker, request, initial_input_path, appended_input_path,
                                create_heads_file_for_testing, create_input_directories, simulate_file_download):
    """
    Call the setup functions first, in a chain ending with `simulate_file_download`.
    Next run the test in question. Finally, remove generated inputs afterwards, even if the test fails.
    """
    # Force ipns_publish to use offline mode to make tests run faster
    mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.json_key", patched_json_key)
    mocker.patch("examples.managers.chirps.CHIRPS.collection", return_value="CHIRPS_test")
    mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.zarr_json_path", patched_zarr_json_path)
    mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.default_root_stac_catalog", patched_root_stac_catalog)
    mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish", empty_ipns_publish)
    yield  # run the tests first
    # delete temp files
    remove_zarr_json()
    remove_dask_worker_dir()
    remove_performance_report()
    # now clean up the various files created for each test
    clean_up_input_paths(initial_input_path, appended_input_path)


@pytest.fixture(scope='module', autouse=True)
def teardown_module(request, heads_path):
    """
    Remove the heads file at the end of all tests
    """
    def test_clean():
        if heads_path.exists():
            os.remove(heads_path)
            print(f"Cleaned up {heads_path}")
    request.addfinalizer(test_clean)


@pytest.mark.order(1)
def test_initial(request, mocker, manager_class, heads_path, test_chunks, initial_input_path, root):
    """
    Test a parse of CHIRPS data. This function is run automatically by pytest because the function name starts with "test_".
    """
    # Get the CHIRPS manager with rebuild set
    manager = manager_class(
        custom_input_path=initial_input_path,
        rebuild=True,
        store='ipld')
    manager.HASH_HEADS_PATH = heads_path
    # Remove IPNS publish mocker on the first run of the dataset, so it lives as "dataset_test" in your IPNS registry
    if manager.json_key() not in manager.ipns_key_list():
        mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish", offline_ipns_publish)
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    manager.requested_dask_chunks = test_chunks
    manager.requested_zarr_chunks = test_chunks
    # run ETL
    manager.transform()
    manager.parse()
    manager.publish_metadata()
    manager.zarr_json_path().unlink(missing_ok=True)
    # Open the head with ipldstore + xarray.open_zarr and compare two data points with the same data points in a local GRIB file
    generated_dataset = manager.zarr_hash_to_dataset(manager.latest_hash())
    lat, lon = 14.625, -91.375
    # Validate one row of data
    output_value = generated_dataset[manager.data_var()].sel(
        latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 12), method='nearest').values
    original_dataset = xarray.open_dataset(root / "chirps_initial_dataset.nc", engine="netcdf4")
    orig_data_var = [key for key in original_dataset.data_vars][0]
    original_value = original_dataset[orig_data_var].sel(
        latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 12)).values
    assert output_value == original_value


def test_append_only(mocker, request, manager_class, heads_path, test_chunks, appended_input_path, root):
    """
    Test an update of chirps data by adding new data to the end of existing data.
    """
    # Get a non-rebuild manager for testing append
    manager = manager_class(custom_input_path=appended_input_path, store='ipld')
    manager.HASH_HEADS_PATH = heads_path
    manager.zarr_chunks = {}
    # Overriding the default time chunk to enable testing chunking with a smaller set of times
    manager.requested_dask_chunks = test_chunks
    manager.requested_zarr_chunks = test_chunks
    # run ETL
    manager.transform()
    manager.parse()
    manager.publish_metadata()
    # Open the head with ipldstore + xarray.open_zarr and compare two data points with the same data points in a local GRIB file
    generated_dataset = manager.zarr_hash_to_dataset(manager.dataset_hash)
    lat, lon = 14.625, -91.375
    # Validate one row of data
    output_value = generated_dataset[manager.data_var()].sel(
        latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 25)).values
    original_dataset = xarray.open_dataset(root / "chirps_append_subset_0.nc", engine="netcdf4")
    orig_data_var = [key for key in original_dataset.data_vars][0]
    original_value = original_dataset[orig_data_var].sel(
        latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 25)).values
    assert output_value == original_value


def test_metadata(manager_class, heads_path):
    """
    Test an update of CHIRPS metadata.

    This function will only work after the test dataset's metadata has been populated into IPFS and the IPNS key list.
    """
    # Get a non-rebuild manager for testing metadata creation
    manager = manager_class(store='ipld')
    manager.HASH_HEADS_PATH = heads_path
    try:
        manager.publish_metadata()
    except Exception:
        manager.fail("Metadata update failed")

def test_thread_count(mocker, manager_class):
    """
    Test if the thread count is the correct ratio to RAM size.
    """
    # Create a fake return object for `psutil.virtual_memory` that will only contain the "total" field.
    class MockMemoryStats:
        pass
    mock_vm_stats = MockMemoryStats()

    # Test if 256GB RAM + 32 CPU == 24 threads
    mock_vm_stats.total = 256 * 1000000000
    mocker.patch("multiprocessing.cpu_count", return_value=32)
    mocker.patch("psutil.virtual_memory", return_value=mock_vm_stats)
    manager = manager_class()
    assert(manager.dask_num_threads == 24)

    # Test if 128GB RAM + 32 CPU == 12 threads
    mock_vm_stats.total = 128 * 1000000000
    mocker.patch("multiprocessing.cpu_count", return_value=32)
    mocker.patch("psutil.virtual_memory", return_value=mock_vm_stats)
    manager = manager_class()
    assert(manager.dask_num_threads == 12)

    # Test if 256GB RAM + 16 CPU == 16 threads
    mock_vm_stats.total = 256 * 1000000000
    mocker.patch("multiprocessing.cpu_count", return_value=16)
    mocker.patch("psutil.virtual_memory", return_value=mock_vm_stats)
    manager = manager_class()
    assert(manager.dask_num_threads == 16)

    # Test the edge case of targeting less than one CPU == 1 thread
    mock_vm_stats.total = 8 * 1000000000
    mocker.patch("multiprocessing.cpu_count", return_value=8)
    mocker.patch("psutil.virtual_memory", return_value=mock_vm_stats)
    manager = manager_class()
    assert(manager.dask_num_threads == 1)

    # Test the edge case of one CPU == 1 thread
    mock_vm_stats.total = 32 * 1000000000
    mocker.patch("multiprocessing.cpu_count", return_value=1)
    mocker.patch("psutil.virtual_memory", return_value=mock_vm_stats)
    manager = manager_class()
    assert(manager.dask_num_threads == 1)
