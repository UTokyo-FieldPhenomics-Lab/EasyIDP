"""Tests for read_point_cloud / write_point_cloud dispatchers and IO module."""
import warnings

import numpy as np
import pytest

import easyidp as idp
import easyidp.pointcloud as pcmod


# -----------------------------------------------------------------------
#  read_point_cloud — suffix dispatch
# -----------------------------------------------------------------------

class TestReadPointCloudSuffixDispatch:
    """Format inferred from file suffix."""

    def test_read_ply_by_suffix(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_ply_bin)
        assert isinstance(pcd, idp.PointCloud)
        assert pcd.has_points()
        assert pcd.shape[0] > 0

    def test_read_las_by_suffix(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_las)
        assert isinstance(pcd, idp.PointCloud)
        assert pcd.has_points()
        assert pcd.shape[0] > 0

    def test_read_laz_by_suffix(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_laz)
        assert isinstance(pcd, idp.PointCloud)
        assert pcd.has_points()
        assert pcd.shape[0] > 0


# -----------------------------------------------------------------------
#  read_point_cloud — explicit format
# -----------------------------------------------------------------------

class TestReadPointCloudExplicitFormat:
    """Format given explicitly, overriding suffix."""

    def test_explicit_ply(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_ply_bin, format="ply")
        assert isinstance(pcd, idp.PointCloud)
        assert pcd.shape[0] > 0

    def test_explicit_las(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_las, format="las")
        assert isinstance(pcd, idp.PointCloud)
        assert pcd.shape[0] > 0

    def test_explicit_laz(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_laz, format="laz")
        assert isinstance(pcd, idp.PointCloud)
        assert pcd.shape[0] > 0


# -----------------------------------------------------------------------
#  read_point_cloud — case insensitivity
# -----------------------------------------------------------------------

class TestReadPointCloudCaseInsensitive:
    """Format is case-insensitive."""

    def test_format_uppercase(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_ply_bin, format="PLY")
        assert pcd.shape[0] > 0

    def test_format_mixed(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_las, format="Las")
        assert pcd.shape[0] > 0

    def test_format_laz_upper(self, test_data):
        pcd = pcmod.read_point_cloud(test_data.pcd.lotus_laz, format="LAZ")
        assert pcd.shape[0] > 0


# -----------------------------------------------------------------------
#  read_point_cloud — errors
# -----------------------------------------------------------------------

class TestReadPointCloudErrors:
    """Error conditions for read_point_cloud."""

    def test_unsupported_format_raises(self, test_data):
        with pytest.raises(ValueError, match="Unsupported format"):
            pcmod.read_point_cloud(test_data.pcd.lotus_ply_bin, format="xyz")

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            pcmod.read_point_cloud("/nonexistent/path.ply")


# -----------------------------------------------------------------------
#  write_point_cloud — suffix dispatch
# -----------------------------------------------------------------------

class TestWritePointCloudSuffixDispatch:
    """Format inferred from target suffix."""

    def test_write_ply_by_suffix(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        target = tmp_path / "test.ply"
        result = pcmod.write_point_cloud(target, pcd)
        assert result == target
        assert target.exists()

    def test_write_las_by_suffix(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_las)
        target = tmp_path / "test.las"
        result = pcmod.write_point_cloud(target, pcd)
        assert result == target
        assert target.exists()

    def test_write_laz_by_suffix(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_laz)
        target = tmp_path / "test.laz"
        result = pcmod.write_point_cloud(target, pcd)
        assert result == target
        assert target.exists()


# -----------------------------------------------------------------------
#  write_point_cloud — explicit format
# -----------------------------------------------------------------------

class TestWritePointCloudExplicitFormat:
    """Format given explicitly."""

    def test_format_adds_suffix_when_missing(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        target = tmp_path / "no_suffix"
        result = pcmod.write_point_cloud(target, pcd, format="ply")
        assert result == tmp_path / "no_suffix.ply"
        assert result.exists()

    def test_format_las_adds_suffix(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_las)
        target = tmp_path / "foo"
        result = pcmod.write_point_cloud(target, pcd, format="las")
        assert result == tmp_path / "foo.las"
        assert result.exists()

    def test_format_laz_adds_suffix(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_laz)
        target = tmp_path / "foo"
        result = pcmod.write_point_cloud(target, pcd, format="laz")
        assert result == tmp_path / "foo.laz"
        assert result.exists()


# -----------------------------------------------------------------------
#  write_point_cloud — errors
# -----------------------------------------------------------------------

class TestWritePointCloudErrors:
    """Error conditions for write_point_cloud."""

    def test_no_suffix_no_format_raises(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        target = tmp_path / "foo"
        with pytest.raises(ValueError, match="Cannot determine format"):
            pcmod.write_point_cloud(target, pcd)

    def test_unsupported_format_raises(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        target = tmp_path / "foo"
        with pytest.raises(ValueError, match="Unsupported format"):
            pcmod.write_point_cloud(target, pcd, format="xyz")

    def test_mismatch_warns_and_adjusts_path(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        target = tmp_path / "foo.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pcmod.write_point_cloud(target, pcd, format="ply")
        assert len(w) >= 1
        assert "format mismatch" in str(w[0].message).lower()
        assert result == tmp_path / "foo.las.ply"
        assert result.exists()


# -----------------------------------------------------------------------
#  Legacy standalone IO — FutureWarning
# -----------------------------------------------------------------------

class TestLegacyStandaloneIOWarnings:
    """Legacy read_ply/read_las/read_laz/write_ply/write_las/write_laz
    must emit FutureWarning and still function."""

    _write_pts = np.array([[-1.9083118, -1.7775583, -0.77878],
                            [-1.9082794, -1.7772741, -0.7802601],
                            [-1.907196, -1.7748289, -0.8017483]], dtype=np.float64)
    _write_cls = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype=np.uint8)

    def test_read_ply_warns(self, test_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pcmod.read_ply(test_data.pcd.lotus_ply_bin)
        assert any("FutureWarning" in str(wrn.category.__name__)
                   for wrn in w), "read_ply should emit FutureWarning"
        pts, cls, nms = result
        assert pts.shape[0] > 0

    def test_read_las_warns(self, test_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pcmod.read_las(test_data.pcd.lotus_las)
        assert any("FutureWarning" in str(wrn.category.__name__)
                   for wrn in w), "read_las should emit FutureWarning"
        pts, cls, nms = result
        assert pts.shape[0] > 0

    def test_read_laz_warns(self, test_data):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pcmod.read_laz(test_data.pcd.lotus_laz)
        assert any("FutureWarning" in str(wrn.category.__name__)
                   for wrn in w), "read_laz should emit FutureWarning"
        pts, cls, nms = result
        assert pts.shape[0] > 0

    def test_write_ply_warns(self, tmp_path):
        out = tmp_path / "legacy.ply"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pcmod.write_ply(out, self._write_pts, self._write_cls)
        assert any("FutureWarning" in str(wrn.category.__name__)
                   for wrn in w), "write_ply should emit FutureWarning"

    def test_write_las_warns(self, tmp_path):
        out = tmp_path / "legacy.las"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pcmod.write_las(out, self._write_pts, self._write_cls)
        assert any("FutureWarning" in str(wrn.category.__name__)
                   for wrn in w), "write_las should emit FutureWarning"

    def test_write_laz_warns(self, tmp_path):
        out = tmp_path / "legacy.laz"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pcmod.write_laz(out, self._write_pts, self._write_cls)
        assert any("FutureWarning" in str(wrn.category.__name__)
                   for wrn in w), "write_laz should emit FutureWarning"


# -----------------------------------------------------------------------
#  Synthetic no-color PLY roundtrip
# -----------------------------------------------------------------------

class TestNoColorPLYRoundtrip:
    """PLY files without color/normal fields roundtrip correctly."""

    def test_no_color_ply_roundtrip(self, tmp_path):
        data = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]], dtype=np.float64)
        pcd = idp.PointCloud()
        pcd.points = data

        target = tmp_path / "nocolor.ply"
        pcmod.write_point_cloud(target, pcd, format="ply")

        pcd2 = pcmod.read_point_cloud(target)
        np.testing.assert_array_almost_equal(pcd2.points, data)
        assert not pcd2.has_colors()
        assert not pcd2.has_normals()


# -----------------------------------------------------------------------
#  read_point_cloud / write_point_cloud — roundtrip
# -----------------------------------------------------------------------

class TestRoundtrip:
    """Full read→write→read cycle via dispatchers preserves data."""

    def test_ply_roundtrip(self, test_data, tmp_path):
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        target = tmp_path / "roundtrip.ply"
        pcmod.write_point_cloud(target, pcd)
        pcd2 = pcmod.read_point_cloud(target)
        np.testing.assert_array_almost_equal(pcd.points, pcd2.points, decimal=3)
        np.testing.assert_array_equal(pcd.colors, pcd2.colors)

    def test_read_write_cycle_preserves_points(self, test_data, tmp_path):
        pcd1 = pcmod.read_point_cloud(test_data.pcd.lotus_las)
        target = tmp_path / "cycle.las"
        pcmod.write_point_cloud(target, pcd1, format="las")
        pcd2 = pcmod.read_point_cloud(target)
        np.testing.assert_array_almost_equal(
            pcd1.points, pcd2.points, decimal=1
        )


# -----------------------------------------------------------------------
#  read_point_cloud — offset parameter
# -----------------------------------------------------------------------

class TestReadPointCloudOffset:
    """read_point_cloud accepts an offset parameter."""

    def test_offset_applied(self, test_data):
        pcd = pcmod.read_point_cloud(
            test_data.pcd.lotus_ply_bin,
            offset=[100.0, 200.0, 50.0],
        )
        np.testing.assert_array_almost_equal(
            pcd.offset, np.array([100.0, 200.0, 50.0])
        )


# -----------------------------------------------------------------------
#  io submodule structure
# -----------------------------------------------------------------------

class TestIOSubmodule:
    """The io/ subpackage is importable."""

    def test_io_package_importable(self):
        from easyidp.pointcloud.io import read_point_cloud, write_point_cloud
        assert callable(read_point_cloud)
        assert callable(write_point_cloud)

    def test_io_las_importable(self):
        from easyidp.pointcloud.io import las
        assert las is not None

    def test_io_ply_importable(self):
        from easyidp.pointcloud.io import ply
        assert ply is not None


# -----------------------------------------------------------------------
#  io backend module API — read/write
# -----------------------------------------------------------------------

class TestIOBackendAPI:
    """las.py and ply.py expose module-scoped read/write."""

    def test_las_read_is_callable(self):
        from easyidp.pointcloud.io.las import read, write
        assert callable(read)
        assert callable(write)

    def test_ply_read_is_callable(self):
        from easyidp.pointcloud.io.ply import read, write
        assert callable(read)
        assert callable(write)


# -----------------------------------------------------------------------
#  io backend module — no private impl names exposed
# -----------------------------------------------------------------------

class TestNoPrivateImplNames:
    """Old _read_*_impl / _write_*_impl names are gone."""

    def test_las_no_old_impl_names(self):
        import easyidp.pointcloud.io.las as las_mod
        for name in ("_read_las_impl", "_write_las_impl"):
            with pytest.raises(AttributeError, match=f"has no attribute '{name}'"):
                getattr(las_mod, name)

    def test_ply_no_old_impl_names(self):
        import easyidp.pointcloud.io.ply as ply_mod
        for name in ("_read_ply_impl", "_write_ply_impl"):
            with pytest.raises(AttributeError, match=f"has no attribute '{name}'"):
                getattr(ply_mod, name)
