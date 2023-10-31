import pytest

from examples.managers.chirps import CHIRPSFinal25


@pytest.fixture
def manager_class():
    """
    etls.managers.CHIRPSFinal25 child to run tests with
    """
    return CHIRPSFinal25
