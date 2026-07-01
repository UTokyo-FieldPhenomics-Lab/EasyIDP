"""ROI <-> PointCloud integration tests (Phase 5).

Tests that ROI.crop(pcd) and pcd.crop(roi) return equivalent results
without emitting FutureWarning from deprecated crop wrappers.
"""

import warnings

import numpy as np
import pytest
import pyproj

import easyidp as idp


@pytest.fixture
def lot_synthetic():
    return np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ], dtype=float)


@pytest.fixture
def utm54n():
    return pyproj.CRS.from_epsg(32654)


@pytest.fixture
def synthetic_roi(lot_synthetic, utm54n):
    roi = idp.ROI()
    roi["lot"] = np.array(lot_synthetic, dtype=float)
    roi.crs = utm54n
    return roi


@pytest.fixture
def synthetic_pcd(utm54n):
    np.random.seed(42)
    xy = np.random.uniform(low=1.0, high=9.0, size=(200, 2))
    z = np.random.uniform(low=50.0, high=100.0, size=(200,))
    pts = np.column_stack([xy, z])
    colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    pcd = idp.PointCloud()
    pcd.points = pts.astype(float)
    pcd.colors = colors
    pcd.crs = utm54n
    return pcd


class TestROICropVsPCDCrop:
    """ROI.crop(pcd) and pcd.crop(roi) should return equivalent results."""

    def test_roi_crop_no_futurewarning(self, synthetic_roi, synthetic_pcd):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = synthetic_roi.crop(synthetic_pcd)
            for warning in w:
                if issubclass(warning.category, FutureWarning):
                    pytest.fail(
                        f"ROI.crop(pcd) emitted FutureWarning: {warning.message}"
                    )

        assert result is not None
        assert "lot" in result

    def test_roi_crop_returns_pointcloud_dict(self, synthetic_roi, synthetic_pcd):
        result = synthetic_roi.crop(synthetic_pcd)
        assert isinstance(result, dict)
        assert len(result) == 1
        assert "lot" in result
        assert isinstance(result["lot"], idp.PointCloud)

    def test_roi_crop_and_pcd_crop_return_same_keys(self, synthetic_roi, synthetic_pcd):
        result_roi = synthetic_roi.crop(synthetic_pcd)
        result_pcd = synthetic_pcd.crop(synthetic_roi)
        assert isinstance(result_roi, dict)
        assert isinstance(result_pcd, dict)
        assert set(result_roi.keys()) == set(result_pcd.keys())

    def test_roi_crop_and_pcd_crop_return_same_point_count(
        self, synthetic_roi, synthetic_pcd
    ):
        result_roi = synthetic_roi.crop(synthetic_pcd)
        result_pcd = synthetic_pcd.crop(synthetic_roi)
        for key in result_roi:
            assert result_roi[key].shape[0] == result_pcd[key].shape[0]

    def test_roi_crop_and_pcd_crop_return_same_points(
        self, synthetic_roi, synthetic_pcd
    ):
        result_roi = synthetic_roi.crop(synthetic_pcd)
        result_pcd = synthetic_pcd.crop(synthetic_roi)
        for key in result_roi:
            np.testing.assert_array_almost_equal(
                result_roi[key].points, result_pcd[key].points
            )

    def test_roi_crop_preserves_colors(self, synthetic_roi, synthetic_pcd):
        result = synthetic_roi.crop(synthetic_pcd)
        for val in result.values():
            assert val.colors is not None

    def test_roi_crop_returns_empty_pointcloud_for_outside_roi(
        self, synthetic_pcd, utm54n
    ):
        roi = idp.ROI()
        roi["far_away"] = np.array([
            [100.0, 100.0],
            [110.0, 100.0],
            [110.0, 110.0],
            [100.0, 110.0],
            [100.0, 100.0],
        ], dtype=float)
        roi.crs = utm54n

        result = roi.crop(synthetic_pcd)
        assert "far_away" in result
        assert result["far_away"].has_points() is False

    def test_roi_crop_returns_empty_pointcloud_for_empty_pcd(
        self, synthetic_roi, utm54n
    ):
        empty_pcd = idp.PointCloud()
        empty_pcd.crs = utm54n

        result = synthetic_roi.crop(empty_pcd)
        assert "lot" in result
        assert result["lot"].has_points() is False

    def test_roi_crop_save_folder_writes_crs_sidecar(
        self, synthetic_roi, synthetic_pcd, tmp_path
    ):
        synthetic_roi.crop(synthetic_pcd, save_folder=tmp_path)

        out_path = tmp_path / "lot.ply"
        assert out_path.exists()
        assert out_path.with_suffix(".crs").exists()
        loaded = idp.PointCloud(out_path)
        assert loaded.crs is not None
        assert loaded.crs.equals(synthetic_pcd.crs)


class TestROIGetZFromPCDNoFutureWarning:
    """ROI.get_z_from_pcd() must not emit FutureWarning from crop_polygon()."""

    def test_get_z_from_pcd_no_futurewarning_face_mode(
        self, synthetic_roi, synthetic_pcd
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            synthetic_roi.get_z_from_pcd(synthetic_pcd, mode="face", kernel="mean")
            for warning in w:
                if issubclass(warning.category, FutureWarning):
                    pytest.fail(
                        f"get_z_from_pcd emitted FutureWarning: {warning.message}"
                    )

    def test_get_z_from_pcd_no_futurewarning_point_mode(
        self, synthetic_roi, synthetic_pcd
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            synthetic_roi.get_z_from_pcd(synthetic_pcd, mode="point", kernel="min")
            for warning in w:
                if issubclass(warning.category, FutureWarning):
                    pytest.fail(
                        f"get_z_from_pcd emitted FutureWarning: {warning.message}"
                    )

    def test_get_z_from_pcd_no_futurewarning_with_buffer(
        self, synthetic_roi, synthetic_pcd
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            synthetic_roi.get_z_from_pcd(synthetic_pcd, mode="face", buffer=1.0)
            for warning in w:
                if issubclass(warning.category, FutureWarning):
                    pytest.fail(
                        f"get_z_from_pcd emitted FutureWarning: {warning.message}"
                    )
