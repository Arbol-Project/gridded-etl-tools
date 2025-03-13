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
