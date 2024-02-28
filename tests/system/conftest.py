import json
import pathlib

import pytest

from examples.managers.chirps import CHIRPSFinal25


@pytest.fixture(scope="module")
def root():
    """
    Directory relative to tests/ where input GRIBs are and temporary input will be generated
    """
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def heads_path(root):
    """
    A local heads file for use during testing
    """
    return root / "heads.json"


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
def appended_input_path(root):
    return root / pathlib.Path("chirps_appended_input")


@pytest.fixture
def appended_input_path_with_hole(root):
    return root / pathlib.Path("chirps_appended_input_with_hole")


@pytest.fixture
def qc_input_path(root):
    return root / pathlib.Path("chirps_qc_input")


@pytest.fixture(scope="session")
def remote_netcdf_input_path(root):
    return "s3://arbol-testing/gridded/chirps/netcdf"


@pytest.fixture(scope="session")
def remote_json_input_path(root):
    return "s3://arbol-testing/gridded/chirps/json"


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


@pytest.fixture
def create_heads_file_for_testing(heads_path):
    """
    Create the heads file only if it doesn't exist
    """
    if not heads_path.exists():
        with open(heads_path, "w") as heads:
            json.dump({}, heads)
        print(f"Created empty heads JSON at {heads_path}")
    else:
        print(f"Found existing heads JSON at {heads_path}")
