import numpy as np
import pytest

import easyidp as idp


class TestSelectByIndex:
    """Tests for PointCloud.select_by_index()."""

    @pytest.fixture
    def pcd_with_data(self, test_data):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        return pcd

    @pytest.fixture
    def empty_pcd(self):
        return idp.PointCloud()

    def test_returns_new_pointcloud(self, pcd_with_data):
        result = pcd_with_data.select_by_index(np.array([0, 1]))
        assert isinstance(result, idp.PointCloud)
        assert result is not pcd_with_data

    def test_preserves_points(self, pcd_with_data):
        indices = np.array([0, 5, 10])
        result = pcd_with_data.select_by_index(indices)
        np.testing.assert_array_almost_equal(
            result.points,
            pcd_with_data.points[indices],
        )

    def test_preserves_colors(self, pcd_with_data):
        indices = np.array([0, 5, 10])
        result = pcd_with_data.select_by_index(indices)
        np.testing.assert_array_almost_equal(
            result.colors,
            pcd_with_data.colors[indices],
        )

    def test_preserves_normals_when_present(self, test_data):
        pcd = idp.PointCloud(test_data.pcd.maize_las)
        indices = np.array([0, 3, 7])
        result = pcd.select_by_index(indices)
        assert result.normals is not None
        np.testing.assert_array_almost_equal(
            result.normals,
            pcd.normals[indices],
        )

    def test_preserves_normals_none(self, pcd_with_data):
        assert pcd_with_data.normals is None
        indices = np.array([0, 1])
        result = pcd_with_data.select_by_index(indices)
        assert result.normals is None

    def test_preserves_offset(self, pcd_with_data):
        indices = np.array([0, 1, 2])
        result = pcd_with_data.select_by_index(indices)
        np.testing.assert_array_almost_equal(
            result.offset,
            pcd_with_data.offset,
        )

    def test_preserves_crs(self, pcd_with_data):
        pcd_with_data.crs = "EPSG:4326"
        indices = np.array([0, 1])
        result = pcd_with_data.select_by_index(indices)
        assert result.crs is not None
        assert result.crs.equals(pcd_with_data.crs)

    def test_empty_selection(self, pcd_with_data):
        indices = np.array([], dtype=int)
        result = pcd_with_data.select_by_index(indices)
        assert isinstance(result, idp.PointCloud)
        assert result.shape[0] == 0
        np.testing.assert_array_almost_equal(
            result.offset,
            pcd_with_data.offset,
        )

    def test_empty_selection_preserves_crs(self, pcd_with_data):
        pcd_with_data.crs = "EPSG:32654"
        result = pcd_with_data.select_by_index(np.array([], dtype=int))
        assert result.crs is not None
        assert result.crs.equals(pcd_with_data.crs)

    def test_invert(self, pcd_with_data):
        all_indices = np.arange(pcd_with_data.shape[0])
        exclude = np.array([0, 1, 2])
        result = pcd_with_data.select_by_index(exclude, invert=True)
        expected_indices = np.setdiff1d(all_indices, exclude)
        np.testing.assert_array_almost_equal(
            result.points,
            pcd_with_data.points[expected_indices],
        )

    def test_invert_empty_selection(self, pcd_with_data):
        result = pcd_with_data.select_by_index(
            np.array([], dtype=int),
            invert=True,
        )
        assert result.shape[0] == pcd_with_data.shape[0]

    def test_invert_all_selection(self, pcd_with_data):
        all_idx = np.arange(pcd_with_data.shape[0])
        result = pcd_with_data.select_by_index(all_idx, invert=True)
        assert result.shape[0] == 0

    def test_shape_matches_selection(self, pcd_with_data):
        indices = np.array([3, 7, 11])
        result = pcd_with_data.select_by_index(indices)
        assert result.shape == (3, 3)

    def test_internal_points_use_offset(self, pcd_with_data):
        indices = np.array([0])
        result = pcd_with_data.select_by_index(indices)
        np.testing.assert_array_almost_equal(
            result._points,
            pcd_with_data._points[indices],
        )
