from pathlib import Path

import pytest

import easyidp as idp


@pytest.fixture(scope="module")
def test_data():
    data = idp.data.TestData(notify_missing=False)
    if not data.is_ready():
        pytest.skip(
            "EasyIDP test data is not downloaded. "
            "Run `idp.data.TestData().download()` before data-dependent tests."
        )

    return data


def test_read_geojson_loads_data(test_data):
    """read_geojson() must actually load data."""
    roi = idp.ROI()
    roi.read_geojson(test_data.json.geojson_soy, name_field="FID")

    assert len(roi) > 0
    assert roi.crs is not None
    assert Path(roi.source) == Path(test_data.json.geojson_soy)
