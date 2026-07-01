import numpy as np
import pyproj

import easyidp as idp


def test_roi_to_crs_returns_new_roi():
    """to_crs() returns a new ROI without modifying source."""
    roi = idp.ROI()
    roi["a"] = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        dtype=float,
    )
    roi.crs = pyproj.CRS.from_epsg(4326)

    target = pyproj.CRS.from_epsg(32654)
    transformed = roi.to_crs(target)

    assert transformed is not roi
    assert transformed.crs.equals(target)
    assert roi.crs.equals(pyproj.CRS.from_epsg(4326))

    orig_idx = roi.item_label["a"]
    trans_idx = transformed.item_label["a"]
    np.testing.assert_array_almost_equal(
        transformed.id_item[trans_idx],
        idp.geotools.convert_proj(roi.id_item, roi.crs, target)[orig_idx],
    )
