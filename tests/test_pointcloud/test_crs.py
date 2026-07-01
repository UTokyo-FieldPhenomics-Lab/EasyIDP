"""Tests for PointCloud CRS conversion: to_crs and change_crs."""

import numpy as np
import pytest
import pyproj

import easyidp as idp


class _SynthHelper:
    """Build synthetic PointCloud for CRS tests without disk I/O."""

    @staticmethod
    def projected_pcd(crs="EPSG:32654", with_colors=True, with_normals=True):
        """Build a PointCloud with UTM-projected coordinates."""
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [368000.0, 3955500.0, 50.0],
            [368050.0, 3955500.0, 52.0],
            [368100.0, 3955550.0, 55.0],
        ], dtype=np.float64)
        pcd.crs = crs
        if with_colors:
            pcd.colors = np.array([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ], dtype=np.uint8)
        if with_normals:
            pcd.normals = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ], dtype=np.float64)
        return pcd

    @staticmethod
    def geographic_pcd(crs="EPSG:4326", with_colors=True, with_normals=True):
        """Build a PointCloud with geographic (lon/lat) coordinates."""
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [139.69, 35.69, 50.0],
            [139.70, 35.69, 52.0],
            [139.71, 35.70, 55.0],
        ], dtype=np.float64)
        pcd.crs = crs
        if with_colors:
            pcd.colors = np.array([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ], dtype=np.uint8)
        if with_normals:
            pcd.normals = np.array([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ], dtype=np.float64)
        return pcd


# -----------------------------------------------------------------------
#  to_crs
# -----------------------------------------------------------------------

class TestToCRS:
    """Tests for PointCloud.to_crs()."""

    def test_returns_new_object(self):
        src = _SynthHelper.projected_pcd()
        result = src.to_crs("EPSG:4326")
        assert result is not src
        assert isinstance(result, idp.PointCloud)

    def test_source_not_mutated(self):
        src = _SynthHelper.projected_pcd()
        orig_points = src.points.copy()
        orig_offset = src.offset.copy()
        orig_colors = src.colors.copy() if src.has_colors() else None
        orig_crs = src.crs

        src.to_crs("EPSG:4326")

        np.testing.assert_array_almost_equal(src.points, orig_points)
        np.testing.assert_array_almost_equal(src.offset, orig_offset)
        assert src.crs.equals(orig_crs)
        if orig_colors is not None:
            np.testing.assert_array_equal(src.colors, orig_colors)

    def test_projected_to_geographic(self):
        """EPSG:32654 → EPSG:4326 should produce lon/lat values."""
        src = _SynthHelper.projected_pcd()
        result = src.to_crs("EPSG:4326")
        assert result.crs.to_epsg() == 4326
        lon, lat = result.points[0, 0], result.points[0, 1]
        assert 139.0 < lon < 140.0
        assert 35.0 < lat < 36.0

    def test_geographic_to_projected(self):
        """EPSG:4326 → EPSG:32654 should produce large meter values."""
        src = _SynthHelper.geographic_pcd()
        result = src.to_crs("EPSG:32654")
        assert result.crs.to_epsg() == 32654
        assert result.points[0, 0] > 100000  # meters, not degrees

    def test_same_crs_returns_copy(self):
        src = _SynthHelper.projected_pcd(with_colors=True, with_normals=True)
        result = src.to_crs("EPSG:32654")
        assert result is not src
        assert result.crs.equals(src.crs)
        np.testing.assert_array_almost_equal(result.points, src.points)
        np.testing.assert_array_equal(result.colors, src.colors)
        np.testing.assert_array_equal(result.normals, src.normals)

    def test_crs_none_raises_error(self):
        pcd = idp.PointCloud()
        pcd.points = np.array([[1, 2, 3]], dtype=np.float64)
        with pytest.raises(TypeError, match="no CRS"):
            pcd.to_crs("EPSG:4326")

    def test_preserves_colors(self):
        src = _SynthHelper.projected_pcd(with_colors=True)
        result = src.to_crs("EPSG:4326")
        assert result.has_colors()
        np.testing.assert_array_equal(result.colors, src.colors)

    def test_preserves_normals(self):
        src = _SynthHelper.projected_pcd(with_normals=True)
        result = src.to_crs("EPSG:4326")
        assert result.has_normals()
        np.testing.assert_array_equal(result.normals, src.normals)

    def test_colors_none_preserved(self):
        src = _SynthHelper.projected_pcd(with_colors=False, with_normals=False)
        result = src.to_crs("EPSG:4326")
        assert not result.has_colors()
        assert not result.has_normals()

    def test_offset_recomputed_for_large_coords(self):
        """When target CRS produces large coords, offset is recomputed."""
        src = _SynthHelper.geographic_pcd()  # small lon/lat values
        result = src.to_crs("EPSG:32654")    # large UTM values
        abs_max = np.max(np.abs(result.points))
        assert abs_max > 65536
        assert np.any(result.offset != 0.0)
        reconstructed = result._points + result.offset
        np.testing.assert_array_almost_equal(reconstructed, result.points)

    def test_offset_recomputed_for_small_coords(self):
        """When target CRS produces small coords, offset stays zero."""
        src = _SynthHelper.projected_pcd()   # large UTM values
        result = src.to_crs("EPSG:4326")     # small lon/lat values
        abs_max = np.max(np.abs(result.points))
        assert abs_max < 65536
        np.testing.assert_array_almost_equal(result.offset, [0.0, 0.0, 0.0])

    def test_tree_invalidated_on_result(self):
        src = _SynthHelper.projected_pcd()
        src.tree  # materialize
        assert src._tree is not None
        result = src.to_crs("EPSG:4326")
        assert result._tree is None

    def test_source_tree_preserved(self):
        src = _SynthHelper.projected_pcd()
        src.tree  # materialize
        src.to_crs("EPSG:4326")
        assert src._tree is not None

    def test_empty_points_with_crs(self):
        """to_crs() on a CRS-assigned but empty PointCloud should not crash."""
        pcd = idp.PointCloud()
        pcd.crs = "EPSG:32654"
        result = pcd.to_crs("EPSG:4326")
        assert result.crs.to_epsg() == 4326
        assert not result.has_points()


# -----------------------------------------------------------------------
#  change_crs
# -----------------------------------------------------------------------

class TestChangeCRS:
    """Tests for PointCloud.change_crs()."""

    def test_mutates_in_place(self):
        pcd = _SynthHelper.projected_pcd()
        pcd_id = id(pcd)
        result = pcd.change_crs("EPSG:4326")
        assert result is None
        assert id(pcd) == pcd_id

    def test_transforms_coordinates(self):
        pcd = _SynthHelper.projected_pcd()
        pcd.change_crs("EPSG:4326")
        assert pcd.crs.to_epsg() == 4326
        lon, lat = pcd.points[0, 0], pcd.points[0, 1]
        assert 139.0 < lon < 140.0
        assert 35.0 < lat < 36.0

    def test_same_crs_noop(self, report_logging_to_caplog):
        """Same CRS should log a warning and not change coordinates."""
        pcd = _SynthHelper.projected_pcd(with_colors=True, with_normals=True)
        orig_points = pcd.points.copy()
        orig_colors = pcd.colors.copy()
        orig_normals = pcd.normals.copy()
        pcd.change_crs("EPSG:32654")
        assert "same" in report_logging_to_caplog.text.lower()
        np.testing.assert_array_almost_equal(pcd.points, orig_points)
        np.testing.assert_array_equal(pcd.colors, orig_colors)
        np.testing.assert_array_equal(pcd.normals, orig_normals)
        assert pcd.crs.to_epsg() == 32654

    def test_crs_none_raises_error(self):
        pcd = idp.PointCloud()
        pcd.points = np.array([[1, 2, 3]], dtype=np.float64)
        with pytest.raises(TypeError, match="no CRS"):
            pcd.change_crs("EPSG:4326")

    def test_preserves_colors(self):
        pcd = _SynthHelper.projected_pcd(with_colors=True)
        colors_before = pcd.colors.copy()
        pcd.change_crs("EPSG:4326")
        assert pcd.has_colors()
        np.testing.assert_array_equal(pcd.colors, colors_before)

    def test_preserves_normals(self):
        pcd = _SynthHelper.projected_pcd(with_normals=True)
        normals_before = pcd.normals.copy()
        pcd.change_crs("EPSG:4326")
        assert pcd.has_normals()
        np.testing.assert_array_equal(pcd.normals, normals_before)

    def test_offset_recomputed_for_large_coords(self):
        """change_crs to projected CRS should recompute offset."""
        pcd = _SynthHelper.geographic_pcd()
        pcd.change_crs("EPSG:32654")
        abs_max = np.max(np.abs(pcd.points))
        assert abs_max > 65536
        assert np.any(pcd.offset != 0.0)
        reconstructed = pcd._points + pcd.offset
        np.testing.assert_array_almost_equal(reconstructed, pcd.points)

    def test_offset_recomputed_for_small_coords(self):
        """change_crs to small coords should zero offset."""
        pcd = _SynthHelper.projected_pcd()
        pcd.change_crs("EPSG:4326")
        np.testing.assert_array_almost_equal(pcd.offset, [0.0, 0.0, 0.0])

    def test_tree_invalidated(self):
        pcd = _SynthHelper.projected_pcd()
        pcd.tree  # materialize
        assert pcd._tree is not None
        pcd.change_crs("EPSG:4326")
        assert pcd._tree is None

    def test_points_property_consistent(self):
        """After change_crs, points should equal _points + offset."""
        pcd = _SynthHelper.projected_pcd()
        pcd.change_crs("EPSG:4326")
        reconstructed = pcd._points + pcd.offset
        np.testing.assert_array_almost_equal(reconstructed, pcd.points)

    def test_accepts_crs_string(self):
        pcd = _SynthHelper.projected_pcd()
        pcd.change_crs("EPSG:4326")
        assert pcd.crs.to_epsg() == 4326

    def test_accepts_pyproj_crs(self):
        pcd = _SynthHelper.projected_pcd()
        target = pyproj.CRS.from_epsg(4326)
        pcd.change_crs(target)
        assert pcd.crs.to_epsg() == 4326

    def test_empty_points_with_crs(self):
        """change_crs() on a CRS-assigned but empty PointCloud should not crash."""
        pcd = idp.PointCloud()
        pcd.crs = "EPSG:32654"
        pcd.change_crs("EPSG:4326")
        assert pcd.crs.to_epsg() == 4326
        assert not pcd.has_points()


# -----------------------------------------------------------------------
#  offset setter tree invalidation
# -----------------------------------------------------------------------

class TestOffsetTreeInvalidation:
    """Offset setter must clear the spatial KDTree."""

    def test_setting_offset_clears_tree(self):
        pcd = _SynthHelper.projected_pcd()
        pcd.tree  # materialize cached tree
        assert pcd._tree is not None
        pcd.offset = [1.0, 2.0, 3.0]
        assert pcd._tree is None
