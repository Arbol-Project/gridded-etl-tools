import logging

import pytest  # noqa: F401


@pytest.fixture(scope="session")
def cap_log_info(caplog):
    caplog.set_level(logging.INFO)
