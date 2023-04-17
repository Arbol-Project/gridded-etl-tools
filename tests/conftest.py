import pytest
import json

#
# Pytest fixtures, automatic fixtures, and plugins that will load automatically for the entire suite of tests.
#
# See the pytest_addoption notes for how to avoid option name conflicts.
#


def pytest_addoption(parser):
    """
    Automatically run by pytest at invocation. These options can be passed on the command line and will be forwarded
    to all functions that include `requests` as a parameter and then can be accessed as `requests.config.option.[NAME]`.

    Options are global whether they are defined in the root tests/ directory or in a subdirectory like tests/era5.
    Therefore, option names in era5/conftest.py and prism_zarr/conftest.py cannot be the same.

    For example, if you wanted an option like "time_chunk", it would either have to be defined here once and apply to
    both ERA5 and PRISM or would have to have to be defined with a different name in each, like "era5_time_chunk" and
    "prism_time_chunk".

    Per subdirectory addoptions are not supported:
    https://github.com/pytest-dev/pytest/issues/7974

    Neither are per subdirectory INI files:
    https://github.com/pytest-dev/pytest/discussions/7732
    """
    # Remove pass statement and add global command line flags here
    pass


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
