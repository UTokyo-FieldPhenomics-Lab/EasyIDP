"""Legacy standalone read/write functions with detailed value assertions.

Merged from tests/test_pointcloud_legacy.py.
All legacy functions emit FutureWarning — use pytest.warns for clean output.
"""

import numpy as np
import pytest

import easyidp.pointcloud as pcmod

WRITE_POINTS = np.asarray([
    [-1.9083118, -1.7775583, -0.77878],
    [-1.9082794, -1.7772741, -0.7802601],
    [-1.907196, -1.7748289, -0.8017483],
    [-1.7892904, -1.9612598, -0.8468666],
    [-1.7885809, -1.9391041, -0.839632],
    [-1.7862186, -1.9365788, -0.8327141],
], dtype=np.float64)

WRITE_COLORS = np.asarray([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [192, 64, 128],
    [92, 88, 83],
    [64, 64, 64],
], dtype=np.uint8)

WRITE_NORMALS = np.asarray([
    [-0.03287353, 0.36604664, 0.9300157],
    [0.08860216, 0.07439037, 0.9932853],
    [-0.01135951, 0.2693031, 0.9629885],
    [0.4548034, -0.15576138, 0.876865],
    [0.4550802, -0.29450312, 0.8403392],
    [0.32758632, 0.27255052, 0.9046565],
], dtype=np.float64)


class TestLegacyReadPly:
    """Legacy read_ply with specific coordinate/color assertions."""

    def test_read_ply_binary(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_ply(test_data.pcd.lotus_ply_bin)

        np.testing.assert_array_almost_equal(
            points[0, :],
            np.array([-18.908312, -15.777558, -0.77878], dtype=np.float32),
        )
        np.testing.assert_array_almost_equal(
            points[-1, :],
            np.array([-15.786219, -17.936579, -0.8327141], dtype=np.float32),
        )
        np.testing.assert_array_almost_equal(
            colors[0, :],
            np.array([123, 103, 79], dtype=np.uint8),
        )
        np.testing.assert_array_almost_equal(
            colors[-1, :],
            np.array([115, 97, 78], dtype=np.uint8),
        )
        assert normals is None

    def test_read_ply_ascii(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_ply(test_data.pcd.lotus_ply_asc)

        np.testing.assert_array_almost_equal(
            points[0, :],
            np.array([-18.908312, -15.777558, -0.77878], dtype=np.float32),
        )
        np.testing.assert_array_almost_equal(
            points[-1, :],
            np.array([-15.786219, -17.936579, -0.8327141], dtype=np.float32),
        )
        np.testing.assert_array_almost_equal(
            colors[0, :],
            np.array([123, 103, 79], dtype=np.uint8),
        )
        np.testing.assert_array_almost_equal(
            colors[-1, :],
            np.array([115, 97, 78], dtype=np.uint8),
        )
        assert normals is None

    def test_read_ply_with_normals(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_ply(test_data.pcd.maize_ply)

        assert normals.shape == (49658, 3)


class TestLegacyReadLas:
    """Legacy read_las with specific assertions."""

    def test_read_las(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_las(test_data.pcd.lotus_las)

        assert points.max() == 0.8320407999999999
        assert points.min() == -18.9083118
        assert colors.max() == 250
        assert colors.min() == 15

    def test_read_las_13ver(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_las(test_data.pcd.lotus_las13)

        assert points.max() == 0.8320407
        assert points.min() == -18.9083118
        assert colors.max() == 250
        assert colors.min() == 15

    def test_read_las_with_normals(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_las(test_data.pcd.maize_las)

        assert normals.shape == (49658, 3)


class TestLegacyReadLaz:
    """Legacy read_laz with specific assertions."""

    def test_read_laz(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_laz(test_data.pcd.lotus_laz)

        assert points.max() == 0.8320407999999999
        assert points.min() == -18.9083118
        assert colors.max() == 250
        assert colors.min() == 15

    def test_read_laz_13ver(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_laz(test_data.pcd.lotus_laz13)

        assert points.max() == 0.8320407
        assert points.min() == -18.9083118
        assert colors.max() == 250
        assert colors.min() == 15

    def test_read_laz_with_normals(self, test_data):
        with pytest.warns(FutureWarning):
            points, colors, normals = pcmod.read_laz(test_data.pcd.maize_laz)

        assert normals.shape == (49658, 3)


class TestLegacyWritePly:
    """Legacy write_ply roundtrip with specific data."""

    def test_write_ply(self, test_data):
        out_dir = test_data.pcd.out
        bin_path = out_dir / "test_def_write_ply_bin.ply"
        asc_path = out_dir / "test_def_write_ply_asc.ply"
        nbin_path = out_dir / "test_def_write_nply_bin.ply"
        nasc_path = out_dir / "test_def_write_nply_asc.ply"

        for p in [bin_path, asc_path, nbin_path, nasc_path]:
            if p.exists():
                p.unlink()

        with pytest.warns(FutureWarning):
            pcmod.write_ply(bin_path, WRITE_POINTS, WRITE_COLORS, binary=True)
        with pytest.warns(FutureWarning):
            pcmod.write_ply(asc_path, WRITE_POINTS, WRITE_COLORS, binary=False)
        with pytest.warns(FutureWarning):
            pcmod.write_ply(nbin_path, WRITE_POINTS, WRITE_COLORS,
                            normals=WRITE_NORMALS, binary=True)
        with pytest.warns(FutureWarning):
            pcmod.write_ply(nasc_path, WRITE_POINTS, WRITE_COLORS,
                            normals=WRITE_NORMALS, binary=False)

        assert bin_path.exists()
        assert asc_path.exists()
        assert nbin_path.exists()
        assert nasc_path.exists()

        with pytest.warns(FutureWarning):
            p, c, n = pcmod.read_ply(bin_path)
        np.testing.assert_almost_equal(p, WRITE_POINTS)
        np.testing.assert_almost_equal(c, WRITE_COLORS)
        assert n is None

        with pytest.warns(FutureWarning):
            p, c, n = pcmod.read_ply(asc_path)
        np.testing.assert_almost_equal(p, WRITE_POINTS)
        np.testing.assert_almost_equal(c, WRITE_COLORS)
        assert n is None

        with pytest.warns(FutureWarning):
            p, c, n = pcmod.read_ply(nbin_path)
        np.testing.assert_almost_equal(p, WRITE_POINTS)
        np.testing.assert_almost_equal(c, WRITE_COLORS)
        np.testing.assert_almost_equal(n, WRITE_NORMALS)

        with pytest.warns(FutureWarning):
            p, c, n = pcmod.read_ply(nasc_path)
        np.testing.assert_almost_equal(p, WRITE_POINTS)
        np.testing.assert_almost_equal(c, WRITE_COLORS)
        np.testing.assert_almost_equal(n, WRITE_NORMALS)


class TestLegacyWriteLas:
    """Legacy write_las roundtrip."""

    def test_write_las(self, test_data):
        out = test_data.pcd.out
        path = out / "test_def_write_las.las"
        npath = out / "test_def_write_nlas.las"

        for p in [path, npath]:
            if p.exists():
                p.unlink()

        with pytest.warns(FutureWarning):
            pcmod.write_las(path, WRITE_POINTS, WRITE_COLORS)
        with pytest.warns(FutureWarning):
            pcmod.write_las(npath, WRITE_POINTS, WRITE_COLORS,
                            normals=WRITE_NORMALS)

        assert path.exists()
        assert npath.exists()

        with pytest.warns(FutureWarning):
            p, c, n = pcmod.read_las(path)
        np.testing.assert_almost_equal(p, WRITE_POINTS, decimal=2)
        np.testing.assert_almost_equal(c, WRITE_COLORS)
        assert n is None

        with pytest.warns(FutureWarning):
            p, c, n = pcmod.read_las(npath)
        np.testing.assert_almost_equal(p, WRITE_POINTS, decimal=2)
        np.testing.assert_almost_equal(c, WRITE_COLORS)
        np.testing.assert_almost_equal(n, WRITE_NORMALS)


class TestLegacyWriteLaz:
    """Legacy write_laz roundtrip."""

    def test_write_laz(self, test_data):
        out = test_data.pcd.out
        path = out / "test_def_write_laz.laz"
        npath = out / "test_def_write_nlaz.laz"

        for p in [path, npath]:
            if p.exists():
                p.unlink()

        with pytest.warns(FutureWarning):
            pcmod.write_laz(path, WRITE_POINTS, WRITE_COLORS)
        with pytest.warns(FutureWarning):
            pcmod.write_laz(npath, WRITE_POINTS, WRITE_COLORS,
                            normals=WRITE_NORMALS)

        assert path.exists()
        assert npath.exists()

        with pytest.warns(FutureWarning):
            p, c, n = pcmod.read_laz(path)
        np.testing.assert_almost_equal(p, WRITE_POINTS, decimal=5)
        np.testing.assert_almost_equal(c, WRITE_COLORS)
        assert n is None

        with pytest.warns(FutureWarning):
            p, c, n = pcmod.read_laz(npath)
        np.testing.assert_almost_equal(p, WRITE_POINTS, decimal=5)
        np.testing.assert_almost_equal(c, WRITE_COLORS)
        np.testing.assert_almost_equal(n, WRITE_NORMALS)
