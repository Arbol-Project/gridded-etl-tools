import pathlib

import pytest

from examples.managers.chirps import CHIRPSFinal25
from ..common import patched_key, patched_zarr_json_path, patched_root_stac_catalog, patched_output_root


@pytest.fixture(scope="module")
def root():
    """
    Directory relative to tests/ where input GRIBs are and temporary input will be generated
    """
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture
def extracted_input_path(root):
    """
    Paths where test_initial input will be generated
    """
    return root / pathlib.Path("chirps_extracted_input")


@pytest.fixture
def initial_input_path(root):
    """
    Paths where test_initial input will be generated
    """
    return root / pathlib.Path("chirps_initial_input")


@pytest.fixture
def initial_smaller_input_path(root):
    """
    Paths where test_initial input will be generated
    """
    return root / pathlib.Path("chirps_initial_smaller_input")


@pytest.fixture
def appended_input_path(root):
    return root / pathlib.Path("chirps_appended_input")


@pytest.fixture
def appended_input_path_with_hole(root):
    return root / pathlib.Path("chirps_appended_input_with_hole")


@pytest.fixture
def qc_input_path(root):
    return root / pathlib.Path("chirps_qc_input")


@pytest.fixture
def manager_class():
    """
    etls.managers.CHIRPSFinal25 child to run tests with
    """
    return CHIRPSFinal25


@pytest.fixture
def test_chunks():
    """
    Time chunk value to use for tests instead of CHIRPS default
    """
    return {"time": 50, "latitude": 40, "longitude": 40}


@pytest.fixture(autouse=True)
def default_mocking(mocker, module_mocker):
    """
    Mockers that are common to every gridded DatasetManager test
    """
    module_mocker.patch("gridded_etl_tools.dataset_manager.DatasetManager.key", patched_key)
    mocker.patch("examples.managers.chirps.CHIRPS.collection", return_value="CHIRPS_test")
    mocker.patch(
        "gridded_etl_tools.dataset_manager.DatasetManager._zarr_json_path",
        patched_zarr_json_path,
    )
    mocker.patch(
        "gridded_etl_tools.dataset_manager.DatasetManager.default_root_stac_catalog",
        patched_root_stac_catalog,
    )
    # Mock the root output directory name, so no existing data will be overwritten and it can be easily cleaned up
    mocker.patch(
        "gridded_etl_tools.utils.convenience.Convenience.output_root",
        patched_output_root,
    )
