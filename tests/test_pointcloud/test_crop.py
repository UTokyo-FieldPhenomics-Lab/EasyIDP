import re
import warnings

import numpy as np
import pytest

import easyidp as idp
from shapely.geometry import Polygon, MultiPolygon


class TestCropPolygon:
    """crop() with Shapely Polygon."""

    @pytest.fixture
    def pcd(self, test_data):
        return idp.PointCloud(test_data.pcd.lotus_ply_bin)

    def test_crop_polygon_returns_pointcloud(self, pcd):
        poly = Polygon([
            [-18.42576599, -16.10819054],
            [-18.00066757, -18.05295944],
            [-16.05021095, -17.63488388],
            [-16.46848488, -15.66774559],
            [-18.42576599, -16.10819054],
        ])
        result = pcd.crop(poly)
        assert isinstance(result, idp.PointCloud)
        assert result.shape[0] > 0

    def test_crop_polygon_results_preserve_offset(self, pcd):
        poly = Polygon([
            [-18.4, -16.1],
            [-18.0, -18.0],
            [-16.0, -17.6],
            [-16.4, -15.6],
            [-18.4, -16.1],
        ])
        result = pcd.crop(poly)
        assert isinstance(result, idp.PointCloud)
        np.testing.assert_array_almost_equal(result.offset, pcd.offset)

    def test_crop_polygon_results_preserve_colors(self, pcd):
        poly = Polygon([
            [-18.0, -16.0],
            [-17.5, -17.0],
            [-16.5, -17.0],
            [-17.0, -16.0],
        ])
        result = pcd.crop(poly)
        assert result.colors is not None
        assert result.colors.shape[0] == result.shape[0]

    def test_crop_polygon_empty_intersection(self, pcd):
        poly = Polygon([[100, 100], [110, 100], [110, 110], [100, 110]])
        result = pcd.crop(poly)
        assert isinstance(result, idp.PointCloud)
        assert result.has_points() is False

    def test_crop_matches_legacy_crop_point_cloud_output(self, pcd):
        """crop(Polygon) should produce same point cloud as crop_point_cloud()."""
        polygon = np.array([
            [-18.42576599, -16.10819054],
            [-18.00066757, -18.05295944],
            [-16.05021095, -17.63488388],
            [-16.46848488, -15.66774559],
            [-18.42576599, -16.10819054],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            legacy = pcd.crop_point_cloud(polygon)
        result = pcd.crop(Polygon([
            (-18.42576599, -16.10819054),
            (-18.00066757, -18.05295944),
            (-16.05021095, -17.63488388),
            (-16.46848488, -15.66774559),
            (-18.42576599, -16.10819054),
        ]))
        assert result.shape[0] == legacy.shape[0]
        np.testing.assert_array_almost_equal(result.points, legacy.points)


class TestCropMultiPolygon:
    """crop() with Shapely MultiPolygon."""

    @pytest.fixture
    def pcd(self, test_data):
        return idp.PointCloud(test_data.pcd.lotus_ply_bin)

    def test_crop_multipolygon_returns_pointcloud(self, pcd):
        poly1 = Polygon([
            (-18.4, -16.1),
            (-18.0, -18.0),
            (-16.0, -17.6),
            (-16.4, -15.6),
            (-18.4, -16.1),
        ])
        poly2 = Polygon([
            (-16.4, -17.6),
            (-16.0, -17.6),
            (-16.0, -15.6),
            (-16.4, -15.6),
        ])
        mp = MultiPolygon([poly1, poly2])
        result = pcd.crop(mp)
        assert isinstance(result, idp.PointCloud)
        assert result.shape[0] > 0


class TestCropErrors:
    """crop() with unsupported inputs."""

    @pytest.fixture
    def pcd(self, test_data):
        return idp.PointCloud(test_data.pcd.lotus_ply_bin)

    def test_crop_ndarray_raises_typeerror(self, pcd):
        arr = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        with pytest.raises(TypeError, match="crop_polygon"):
            pcd.crop(arr)

    def test_crop_list_raises_typeerror(self, pcd):
        lst = [[0, 0], [1, 0], [1, 1], [0, 1]]
        with pytest.raises(TypeError, match="crop_polygon"):
            pcd.crop(lst)

    def test_crop_str_raises_typeerror(self, pcd):
        with pytest.raises(TypeError):
            pcd.crop("not valid")


class TestLegacyFutureWarning:
    """Legacy crop methods emit FutureWarning."""

    @pytest.fixture
    def pcd(self, test_data):
        return idp.PointCloud(test_data.pcd.lotus_ply_bin)

    def test_crop_point_cloud_warns(self, pcd):
        polygon = np.array([
            [-18.4, -16.1],
            [-18.0, -18.0],
            [-16.0, -17.6],
            [-16.4, -15.6],
            [-18.4, -16.1],
        ])
        with pytest.warns(FutureWarning, match="crop\\(\\)"):
            pcd.crop_point_cloud(polygon)

    def test_crop_point_cloud_still_works(self, pcd):
        polygon = np.array([
            [-18.42576599, -16.10819054],
            [-18.00066757, -18.05295944],
            [-16.05021095, -17.63488388],
            [-16.46848488, -15.66774559],
            [-18.42576599, -16.10819054],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = pcd.crop_point_cloud(polygon)
        assert isinstance(result, idp.PointCloud)
        assert result.shape[0] == 22422

    def test_crop_polygon_warns(self, pcd):
        polygon = np.array([
            [-18.4, -16.1],
            [-18.0, -18.0],
            [-16.0, -17.6],
            [-16.4, -15.6],
            [-18.4, -16.1],
        ])
        with pytest.warns(FutureWarning, match="crop\\(\\)"):
            pcd.crop_polygon(polygon)

    def test_crop_polygon_still_works(self, pcd):
        polygon = np.array([
            [-18.42576599, -16.10819054],
            [-18.00066757, -18.05295944],
            [-16.05021095, -17.63488388],
            [-16.46848488, -15.66774559],
            [-18.42576599, -16.10819054],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = pcd.crop_polygon(polygon)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0
        assert result.shape[1] == 3

    # ROI-specific behavior is covered in test_roi_integration.py.
    # Legacy crop_rois behavior is covered in test_legacy_crop.py.


class TestCropPolygonLegacy:
    """Legacy crop_polygon returns ndarray."""

    @pytest.fixture
    def pcd(self, test_data):
        return idp.PointCloud(test_data.pcd.lotus_ply_bin)

    def test_crop_polygon_returns_ndarray(self, pcd):
        polygon = np.array([
            [-18.42576599, -16.10819054],
            [-18.00066757, -18.05295944],
            [-16.05021095, -17.63488388],
            [-16.46848488, -15.66774559],
            [-18.42576599, -16.10819054],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = pcd.crop_polygon(polygon)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3


class TestLegacyExisting:
    """Preserve test_class_crop behavior from original test suite."""

    def test_class_point_cloud_crop(self, test_data, report_logging_to_caplog):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)

        polygon = np.array([
            [-18.42576599, -16.10819054],
            [-18.00066757, -18.05295944],
            [-16.05021095, -17.63488388],
            [-16.46848488, -15.66774559],
            [-18.42576599, -16.10819054]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cropped = pcd.crop_point_cloud(polygon)

        assert isinstance(cropped, idp.PointCloud)
        assert cropped.shape[0] == 22422

        p1 = [[1,2,3], [4,5,6]]
        p2 = np.array(p1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(TypeError, match=re.escape(
                "Only numpy ndarray are supported as `polygon_xy` inputs, not <class 'list'>")):
                cropped = pcd.crop_point_cloud(p1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(IndexError, match=re.escape(
                "Please only spcify shape like (N, 2), not (2, 3)")):
                cropped = pcd.crop_point_cloud(p2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cropped = pcd.crop_point_cloud(polygon + 10)
        assert cropped is None
        assert "Cropped 0 point in given polygon." in report_logging_to_caplog.text
