import logging

import pytest
import easyidp as idp


@pytest.fixture(scope="session")
def test_data():
    data = idp.data.TestData(notify_missing=False)
    if not data.is_ready():
        pytest.skip(
            "EasyIDP test data is not downloaded. "
            "Run `idp.data.TestData().download()` before data-dependent tests."
        )
    return data


@pytest.fixture
def report_logging_to_caplog(caplog):
    target_logger = logging.getLogger("easyidp")
    original_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="easyidp")
    target_logger.addHandler(caplog.handler)

    yield caplog

    target_logger.removeHandler(caplog.handler)
    target_logger.setLevel(original_level)
