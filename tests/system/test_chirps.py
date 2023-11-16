import os
import datetime
import pathlib
import pytest
import xarray
import shutil
import glob

from ..common import (
    clean_up_input_paths,
    empty_ipns_publish,
    get_manager,
    offline_ipns_publish,
    patched_json_key,
    patched_root_stac_catalog,
    patched_zarr_json_path,
    remove_dask_worker_dir,
    remove_performance_report,
    remove_zarr_json,
    initial,
    original_ds_normal,
    original_ds_bad_data,
    original_ds_no_time,
    original_ds_bad_time,
    nc4_input_files,
    json_input_files
)


@pytest.fixture
def create_input_directories(initial_input_path, appended_input_path, appended_input_path_with_hole):
    """
    The testing directories for initial, append and insert will get created before each run
    """
    for path in (
        initial_input_path,
        appended_input_path,
        appended_input_path_with_hole,
        qc_input_path
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
    mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.json_key", patched_json_key)
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
    clean_up_input_paths(initial_input_path, appended_input_path, appended_input_path_with_hole)


@pytest.fixture(scope="module", autouse=True)
def teardown_module(request, heads_path):
    """
    Remove the heads file at the end of all tests
    """

    def test_clean():
        if heads_path.exists():
            os.remove(heads_path)
            print(f"Cleaned up {heads_path}")

    request.addfinalizer(test_clean)


# def test_initial_dry_run(request, mocker, manager_class, heads_path, test_chunks, initial_input_path, root):
#     """
#     Test that a dry run parse of CHIRPS data does not, in fact, parse data.
#     """
#     # Get the CHIRPS manager with rebuild set
#     manager = manager_class(custom_input_path=initial_input_path, rebuild_requested=True, dry_run=True, store="ipld")
#     manager.HASH_HEADS_PATH = heads_path
#     # Remove IPNS publish mocker on the first run of the dataset, so it lives as "dataset_test" in your IPNS registry
#     if manager.json_key() not in manager.ipns_key_list():
#         mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish", offline_ipns_publish)
#     # Overriding the default time chunk to enable testing chunking with a smaller set of times
#     manager.requested_dask_chunks = test_chunks
#     manager.requested_zarr_chunks = test_chunks
#     # run ETL
#     manager.transform()
#     manager.parse()
#     manager.zarr_json_path().unlink(missing_ok=True)
#     # Check that a hash wasn't created because the dataset wasn't parsed
#     with pytest.raises(FileNotFoundError):
#         manager.zarr_hash_to_dataset(manager.latest_hash())


# def test_initial(request, mocker, manager_class, heads_path, test_chunks, initial_input_path, root):
#     """
#     Test a parse of CHIRPS data. This function is run automatically by pytest because the function name starts with
#     "test_".
#     """
#     # Get the CHIRPS manager with rebuild set
#     manager = manager_class(custom_input_path=initial_input_path, rebuild_requested=True, store="ipld")
#     manager.HASH_HEADS_PATH = heads_path
#     # Remove IPNS publish mocker on the first run of the dataset, so it lives as "dataset_test" in your IPNS registry
#     if manager.json_key() not in manager.ipns_key_list():
#         mocker.patch(
#             "gridded_etl_tools.dataset_manager.DatasetManager.ipns_publish",
#             offline_ipns_publish,
#         )
#     # Overriding the default time chunk to enable testing chunking with a smaller set of times
#     manager.requested_dask_chunks = test_chunks
#     manager.requested_zarr_chunks = test_chunks
#     # run ETL
#     manager.transform()
#     manager.parse()
#     manager.publish_metadata()
#     manager.zarr_json_path().unlink(missing_ok=True)
#     # Open the head with ipldstore + xarray.open_zarr and compare two data points with the same data points in a local
#     # GRIB file
#     generated_dataset = manager.zarr_hash_to_dataset(manager.latest_hash())
#     lat, lon = 14.625, -91.375
#     # Validate one row of data
#     output_value = (
#         generated_dataset[manager.data_var()]
#         .sel(
#             latitude=lat,
#             longitude=lon,
#             time=datetime.datetime(2003, 5, 12),
#             method="nearest",
#         )
#         .values
#     )
#     original_dataset = xarray.open_dataset(root / "chirps_initial_dataset.nc", engine="netcdf4")
#     orig_data_var = [key for key in original_dataset.data_vars][0]
#     original_value = (
#         original_dataset[orig_data_var].sel(latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 12)).values
#     )
#     assert output_value == original_value


# def test_prepare_input_files(manager_class, mocker, appended_input_path):
#     """
#     Test that the constituent steps of prepare_input_files work, as expressed through the example CHIRPS manager
#     """
#     mocker.patch(
#         "examples.managers.chirps.CHIRPSFinal25.local_input_path",
#         return_value=appended_input_path,
#     )
#     dm = get_manager(manager_class)
#     # Test that prepare_input_files successfully expands 2 files to 32 files
#     assert len(list(dm.input_files())) == 2
#     dm.convert_to_lowest_common_time_denom(list(dm.input_files()), keep_originals=False)
#     assert len(list(dm.input_files())) == 32
#     # Test that ncs_to_nc4s converts all NC files to NC4s, removing the original NCs in the process
#     dm.ncs_to_nc4s(keep_originals=False)
#     input_ncs = [pathlib.Path(file) for file in glob.glob(str(dm.local_input_path() / "*.nc"))]
#     input_nc4s = dm.input_files()
#     assert len(input_ncs) == 0
#     assert len(list(input_nc4s)) == 32


# def test_append_only(mocker, request, manager_class, heads_path, test_chunks, appended_input_path, root):
#     """
#     Test an update of chirps data by adding new data to the end of existing data.
#     """
#     # Get a non-rebuild manager for testing append
#     manager = manager_class(custom_input_path=appended_input_path, store="ipld")
#     manager.HASH_HEADS_PATH = heads_path
#     manager.zarr_chunks = {}
#     # Overriding the default time chunk to enable testing chunking with a smaller set of times
#     manager.requested_dask_chunks = test_chunks
#     manager.requested_zarr_chunks = test_chunks
#     # run ETL
#     manager.transform()
#     manager.parse()
#     manager.publish_metadata()
#     # Open the head with ipldstore + xarray.open_zarr and compare two data points with the same data points in a local
#     # GRIB file
#     generated_dataset = manager.zarr_hash_to_dataset(manager.dataset_hash)
#     lat, lon = 14.625, -91.375
#     # Validate one row of data
#     output_value = (
#         generated_dataset[manager.data_var()]
#         .sel(latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 25))
#         .values
#     )
#     original_dataset = xarray.open_dataset(root / "chirps_append_subset_0.nc", engine="netcdf4")
#     orig_data_var = [key for key in original_dataset.data_vars][0]
#     original_value = (
#         original_dataset[orig_data_var].sel(latitude=lat, longitude=lon, time=datetime.datetime(2003, 5, 25)).values
#     )
#     assert output_value == original_value


# def test_bad_append(
#     mocker,
#     request,
#     manager_class,
#     heads_path,
#     test_chunks,
#     appended_input_path_with_hole,
#     root,
# ):
#     """
#     Test an update of chirps data by adding new data to the end of existing data.
#     """
#     # Get a non-rebuild manager for testing append
#     manager = manager_class(custom_input_path=appended_input_path_with_hole, store="ipld")
#     manager.HASH_HEADS_PATH = heads_path
#     manager.zarr_chunks = {}
#     # Overriding the default time chunk to enable testing chunking with a smaller set of times
#     manager.requested_dask_chunks = test_chunks
#     manager.requested_zarr_chunks = test_chunks
#     # run ETL
#     manager.transform()
#     with pytest.raises(IndexError):
#         manager.parse()


# def test_metadata(manager_class, heads_path):
#     """
#     Test an update of CHIRPS metadata.

#     This function will only work after the test dataset's metadata has been populated into IPFS and the IPNS key list.
#     """
#     # Get a non-rebuild manager for testing metadata creation
#     manager = manager_class(store="ipld")
#     manager.HASH_HEADS_PATH = heads_path
#     manager.publish_metadata()
#     assert manager.load_stac_metadata() != {}


def test_post_parse_quality_check(mocker, capfd, initial_input_path, manager_class):
    """
    Test that the post-parse quality check method waves through good data
    and fails as anticipated with bad data
    """
    # Prepare a dataset manager
    dm = initial(manager_class, input_path=initial_input_path)
    # Approves aligned values
    dm.post_parse_quality_check(checks=5)
    assert dm.post_parse_quality_check(checks=5)
    # Rejects misaligned values
    mocker.patch("gridded_etl_tools.utils.zarr_methods.Publish.get_original_ds", original_ds_bad_data)
    with pytest.raises(ValueError):
        dm.post_parse_quality_check(checks=5)
    # Skipping the QC
    dm.skip_post_parse_qc = True
    mocker.patch("gridded_etl_tools.utils.zarr_methods.Publish.get_original_ds", original_ds_normal)
    dm.post_parse_quality_check(checks=5)
    out, _ = capfd.readouterr()
    assert "Skipping post-parse quality check" in out

# def test_get_original_ds(mocker, manager_class, initial_input_path):
#     """
#     Test that the get_original_ds function correctly loads in datasets as anticipated for
#     local and remote files alike
#     """
#     # Prepare a dataset manager
#     dm = initial(manager_class, input_path=initial_input_path, use_local_zarr_jsons=True)
#     # Local data
#     dm.protocol = 'file'
#     mocker.patch("gridded_etl_tools.utils.convenience.Convenience.input_files", nc4_input_files)
#     assert dm.get_original_ds()
#     # Remote data
#     dm.protocol = 's3'
#     mocker.patch("gridded_etl_tools.utils.convenience.Convenience.input_files", json_input_files)
#     assert dm.get_original_ds()

def test_reformat_orig_ds(mocker, manager_class, qc_input_path):
    """
    Test that the original dataset is correctly reformatted when fed incorect data
    """
    # Prepare a dataset manager
    dm = initial(manager_class, qc_input_path)
    # Populates time dimension from filename if missing dataset
    mocker.patch("gridded_etl_tools.utils.zarr_methods.Publish.get_original_ds", original_ds_no_time)
    assert dm.post_parse_quality_check(checks=5)

def test_check_values(mocker, initial_input_path, manager_class):
    """
    Test that the values check exits as anticipated when given an original dataset whose
    time dimension doesn't correspond to the production dataset
    """
    # Prepare a dataset manager
    dm = initial(manager_class, initial_input_path)
    ### Exits if time in original file doesn't match time in prod dataset
    mocker.patch("gridded_etl_tools.utils.zarr_methods.Publish.get_original_ds", original_ds_bad_time)
    prod_ds = dm.store.dataset()
    orig_ds = dm.get_original_ds()
    random_coords = dm.get_random_coords(prod_ds)
    assert not dm.check_value(random_coords, orig_ds, prod_ds)
