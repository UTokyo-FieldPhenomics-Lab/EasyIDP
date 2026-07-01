"""Legacy PointCloud class IO: init, print, points setter, offset, read, clear.

Merged from tests/test_pointcloud_legacy.py.
CRS property and change_crs legacy baselines are already covered in test_crs.py.
"""

import re

import numpy as np
import pytest

import easyidp as idp


class TestPointCloudLegacyInit:
    """Legacy PointCloud init and construction."""

    def test_init_empty(self):
        pcd = idp.PointCloud()

        assert pcd.points is None
        assert pcd.colors is None
        assert pcd.normals is None
        np.testing.assert_array_almost_equal(
            pcd._offset, np.array([0.0, 0.0, 0.0]),
        )

        assert pcd.has_points() is False
        assert pcd.has_colors() is False
        assert pcd.has_normals() is False

        assert pcd.shape == (0, 3)

    def test_init_wrong_path(self, report_logging_to_caplog):
        idp.PointCloud("a/wrong/path.ply")
        assert "Can not find file" in report_logging_to_caplog.text


class TestPointCloudLegacyPrint:
    """Legacy _btf_print display format."""

    def test_print_short_table(self):
        expected = (
            '      x    y    z  r       g       b           '
            'nx      ny      nz\n 0    1    2    3  nodata  nodata  '
            'nodata  nodata  nodata  nodata\n 1    4    5    6  nodata  '
            'nodata  nodata  nodata  nodata  nodata\n 2    7    8    9  '
            'nodata  nodata  nodata  nodata  nodata  nodata'
        )
        pcd = idp.PointCloud()
        pcd.points = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert pcd._btf_print.replace(' ', '') == expected.replace(' ', '')

    def test_print_long_table(self, test_data):
        expected = (
            '             x        y        z  r    g    b        '
            'nx      ny      nz\n    0  -18.908  -15.778   -0.779  '
            '123  103  79   nodata  nodata  nodata\n    1  -18.908  '
            '-15.777   -0.78   124  104  81   nodata  nodata  nodata'
            '\n    2  -18.907  -15.775   -0.802  123  103  80   '
            'nodata  nodata  nodata\n  ...  ...      ...      ...   '
            '   ...  ...  ...     ...     ...     ...\n42451  -15.789  '
            '-17.961   -0.847  116  98   80   nodata  nodata  nodata'
            '\n42452  -15.789  -17.939   -0.84   113  95   76   '
            'nodata  nodata  nodata\n42453  -15.786  -17.937   -0.833  '
            '115  97   78   nodata  nodata  nodata'
        )
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        assert pcd._btf_print.replace(' ', '') == expected.replace(' ', '')


class TestPointCloudLegacyPoints:
    """Legacy points property validation."""

    def test_points_setter_type_error(self):
        pcd = idp.PointCloud()
        with pytest.raises(TypeError, match=re.escape(
            "Only numpy ndarray object are acceptable for setting values",
        )):
            pcd.points = [[1, 2, 3], [4, 5, 6]]

    def test_points_setter_ndarray(self):
        pts1 = np.asarray([[1, 2, 3], [4, 5, 6]])
        pcd = idp.PointCloud()
        pcd.points = pts1
        assert pcd.shape == (2, 3)

    def test_points_setter_shape_mismatch(self):
        pts1 = np.asarray([[1, 2, 3], [4, 5, 6]])
        pts2 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        pcd = idp.PointCloud()
        pcd.points = pts1

        with pytest.raises(IndexError, match=re.escape(
            "The given shape [(3, 3)] does not match current point cloud "
            "shape [(2, 3)]",
        )):
            pcd.points = pts2

    def test_points_setter_after_shape_change(self):
        pts1 = np.asarray([[1, 2, 3], [4, 5, 6]])
        pts2 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        pcd = idp.PointCloud()
        pcd.points = pts1
        pcd.shape = pts2.shape
        pcd.points = pts2
        np.testing.assert_almost_equal(pcd.points, pts2)


class TestPointCloudLegacyOffset:
    """Legacy offset construction and mutation."""

    def test_offset_wrong_type_list(self):
        with pytest.raises(ValueError, match=re.escape(
            "Please give correct 3D coordinate [x, y, z], only 2 was given",
        )):
            idp.PointCloud(offset=[367900, 3955800])

    def test_offset_wrong_type_array(self):
        with pytest.raises(ValueError, match=re.escape(
            "Please give correct 3D coordinate [x, y, z], only 2 was given",
        )):
            idp.PointCloud(offset=np.array([367900, 3955800]))

    def test_offset_wrong_type_dict(self):
        with pytest.raises(ValueError, match=re.escape(
            "Only [x, y, z] list or np.array([x, y, z]) are acceptable",
        )):
            idp.PointCloud(offset={"x": 367900, "y": 3955800, "z": 0})

    def test_offset_set_value_default(self):
        pts = np.asarray([[1, 2, 3], [4, 5, 6]])
        pcd = idp.PointCloud()
        pcd.points = pts

        np.testing.assert_almost_equal(pcd.points, pts)
        np.testing.assert_almost_equal(pcd._points, pts)

        pcd.update_offset_value(np.array([1, 1, 1]))

        np.testing.assert_almost_equal(pcd.points, pts)
        np.testing.assert_almost_equal(pcd._points,
                                       np.array([[0, 1, 2], [3, 4, 5]]))

    def test_offset_set_value_with_offset(self):
        pts = np.asarray([[1, 2, 3], [4, 5, 6]])
        pcd = idp.PointCloud(offset=[10, 10, 10])
        pcd.points = pts

        np.testing.assert_almost_equal(pcd.points, pts)
        np.testing.assert_almost_equal(pcd._points,
                                       np.array([[-9, -8, -7], [-6, -5, -4]]))

        pcd.update_offset_value([0, 0, 0])

        np.testing.assert_almost_equal(pcd.points, pts)
        np.testing.assert_almost_equal(pcd._points, pts)


class TestPointCloudLegacyRead:
    """Legacy PointCloud read via constructor and read_point_cloud()."""

    def test_read_no_offset(self, test_data):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)

        assert pcd.points is not None
        np.testing.assert_array_almost_equal(
            pcd.points[0, :],
            np.array([-18.908312, -15.777558, -0.77878], dtype=np.float32),
        )
        np.testing.assert_array_almost_equal(
            pcd.points[-1, :],
            np.array([-15.786219, -17.936579, -0.8327141], dtype=np.float32),
        )

        assert pcd.colors is not None
        np.testing.assert_array_almost_equal(
            pcd.colors[0, :],
            np.array([123, 103, 79], dtype=np.uint8),
        )
        np.testing.assert_array_almost_equal(
            pcd.colors[-1, :],
            np.array([115, 97, 78], dtype=np.uint8),
        )

        assert pcd.normals is None
        np.testing.assert_array_almost_equal(
            pcd._offset, np.array([0.0, 0.0, 0.0]),
        )

        assert pcd.has_points()
        assert pcd.has_colors()
        assert pcd.has_normals() is False

    def test_read_with_offsets(self, test_data):
        pcd = idp.PointCloud(test_data.pcd.maize_las)

        np.testing.assert_almost_equal(
            pcd._offset, np.array([367900.0, 3955800.0, 0.0]),
        )
        np.testing.assert_almost_equal(
            pcd._points[0, :], np.array([93.0206, 65.095, 57.9707]),
        )

    def test_specify_origin_offset_list(self):
        pcd = idp.PointCloud(offset=[367900, 3955800, 0])
        np.testing.assert_almost_equal(
            pcd._offset, np.array([367900.0, 3955800.0, 0.0]),
        )

    def test_specify_origin_offset_tuple(self):
        pcd = idp.PointCloud(offset=(367900, 3955800, 0))
        np.testing.assert_almost_equal(
            pcd._offset, np.array([367900.0, 3955800.0, 0.0]),
        )

    def test_specify_origin_offset_array(self):
        pcd = idp.PointCloud(offset=np.array([367900, 3955800, 0]))
        np.testing.assert_almost_equal(
            pcd._offset, np.array([367900.0, 3955800.0, 0.0]),
        )


class TestPointCloudLegacyClear:
    """Legacy PointCloud.clear() behavior."""

    def test_clear(self, test_data):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        pcd.clear()

        assert pcd.points is None
        assert pcd.colors is None
        assert pcd.normals is None
        np.testing.assert_array_almost_equal(
            pcd._offset, np.array([0.0, 0.0, 0.0]),
        )

        assert pcd.has_points() is False
        assert pcd.has_colors() is False
        assert pcd.has_normals() is False

        assert pcd.shape == (0, 3)
