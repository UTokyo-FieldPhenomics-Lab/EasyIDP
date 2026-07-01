import numpy as np
import pyproj
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


def test_roi_crop_pcd_aligns_crs_before_crop():
    """ROI.crop(pcd) with differing CRS must align before cropping."""
    utm54n = pyproj.CRS.from_epsg(32654)
    wgs84 = pyproj.CRS.from_epsg(4326)

    transformer = pyproj.Transformer.from_crs(wgs84, utm54n, always_xy=True)
    x_utm, y_utm = transformer.transform(139.5405, 35.7347)

    roi = idp.ROI()
    roi["lot"] = np.array(
        [
            [139.5405, 35.7347],
            [139.5406, 35.7347],
            [139.5406, 35.7346],
            [139.5405, 35.7346],
            [139.5405, 35.7347],
        ],
        dtype=float,
    )
    roi.crs = wgs84

    pcd = idp.PointCloud()
    pcd.points = np.array(
        [[x_utm, y_utm, 50.0], [x_utm + 1.0, y_utm + 1.0, 60.0]],
        dtype=float,
    )
    pcd.crs = utm54n

    result = roi.crop(pcd)

    assert len(result) == 1
    item = list(result.values())[0]
    assert item.has_points(), (
        "ROI.crop(pcd) with CRS mismatch should align CRS before crop; "
        "got empty point cloud"
    )


def test_roi_crop_pcd_creates_save_folder(test_data, tmp_path):
    """save_folder should create output files without crashing."""
    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    roi = roi[:3]
    dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
    roi.change_crs(dom.crs)

    save_dir = tmp_path / "pcd_crops"
    pcd = idp.PointCloud(
        test_data.pix4d.lotus_pcd,
        offset=[368043, 3955495, 98],
    )
    result = roi.crop(pcd, save_folder=str(save_dir))

    assert len(result) == 3
    for key in result:
        save_path = save_dir / (key + pcd.file_ext)
        assert save_path.exists(), f"Expected {save_path} to exist"


def test_roi_get_z_from_pcd_face_mode():
    """get_z_from_pcd() should fill polygon z values from PCD points."""
    roi = idp.ROI()
    roi["test_square"] = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        dtype=float,
    )
    roi.crs = pyproj.CRS.from_epsg(32654)

    pcd = idp.PointCloud()
    pcd.points = np.array([[5, 5, 100], [15, 15, 50], [1, 1, 10]], dtype=float)
    pcd.crs = pyproj.CRS.from_epsg(32654)

    roi_face = roi.copy()
    roi_face.get_z_from_pcd(pcd, mode="face", kernel="mean")
    assert len(roi_face["test_square"][0]) == 3
    np.testing.assert_almost_equal(roi_face["test_square"][:, -1], 55.0)

    roi_face_max = roi.copy()
    roi_face_max.get_z_from_pcd(pcd, mode="face", kernel="max")
    np.testing.assert_almost_equal(roi_face_max["test_square"][:, -1], 100.0)

    roi_buffer = roi.copy()
    roi_buffer.get_z_from_pcd(pcd, mode="face", buffer=10.0)
    np.testing.assert_almost_equal(roi_buffer["test_square"][:, -1], 53.3333333)

    pcd_wgs84 = idp.PointCloud()
    pcd_wgs84.points = pcd.points
    pcd_wgs84.crs = pyproj.CRS.from_epsg(4326)

    roi_warn = roi.copy()
    roi_warn.get_z_from_pcd(pcd_wgs84)


def test_roi_get_z_from_pcd_point_mode():
    """get_z_from_pcd() should support nearest-neighbor point mode."""
    roi = idp.ROI()
    roi["test_pt"] = np.array([[5, 5], [5.15, 5.15]])
    roi.crs = pyproj.CRS.from_epsg(32654)

    pcd = idp.PointCloud()
    pcd.points = np.array([[5, 5, 100], [5.2, 5.2, 200]])

    roi.get_z_from_pcd(pcd, mode="point", buffer=0)

    expected = np.array([100, 200])
    np.testing.assert_almost_equal(roi["test_pt"][:, 2], expected)
