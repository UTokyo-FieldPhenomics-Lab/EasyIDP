"""Regression tests for point-cloud issues resolved or deferred in v2.1.

Each class maps to one or more related GitHub issues, providing
targeted proof that the issue is fixed (or intentionally deferred).
Test are designed to pass without any production code change when
the underlying fixes from Phases 1-5 already exist.

Issue mapping
-------------
* #55  — cropped PointCloud loses CRS
* #78  — reconstruction pcd assignment still works via idp.PointCloud
* #93  — indoor / no-GPS point cloud workflows (no CRS required)
* #101 — cropped preview loses color info
* #102 — import scope: no Open3D dependency
* #121 — DEM/CHM higher-level processing (deferred, non-goal)
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from shapely.geometry import Polygon, MultiPolygon

import easyidp as idp
import easyidp.pointcloud as pcmod


# -----------------------------------------------------------
#  #55 & #101 — crop metadata: CRS, colors, normals, offset
# -----------------------------------------------------------

class TestCropCRS55:
    """#55: cropped point cloud preserves CRS."""

    def test_crop_polygon_preserves_crs(self):
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [368001.0, 3955501.0, 50.0],
            [368002.0, 3955502.0, 52.0],
            [368003.0, 3955503.0, 55.0],
        ], dtype=np.float64)
        pcd.crs = "EPSG:32654"
        poly = Polygon([
            (368000, 3955500), (368010, 3955500),
            (368010, 3955510), (368000, 3955510),
        ])
        result = pcd.crop(poly)
        assert result.crs is not None
        assert result.crs.equals(pcd.crs)

    def test_crop_multipolygon_preserves_crs(self):
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [1.0, 1.0, 0.0], [2.0, 2.0, 0.0],
            [5.0, 5.0, 0.0], [6.0, 6.0, 0.0],
        ], dtype=np.float64)
        pcd.crs = "EPSG:4326"
        mp = MultiPolygon([
            Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
            Polygon([(4, 4), (7, 4), (7, 7), (4, 7)]),
        ])
        result = pcd.crop(mp)
        assert result.crs is not None
        assert result.crs.equals(pcd.crs)

    def test_select_by_index_preserves_crs(self):
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
        ], dtype=np.float64)
        pcd.crs = "EPSG:32654"
        result = pcd.select_by_index(np.array([0]))
        assert result.crs is not None
        assert result.crs.equals(pcd.crs)


class TestCropColors101:
    """#101: cropped preview preserves colors, normals, display state."""

    def test_crop_preserves_colors_and_normals(self):
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [1.0, 2.0, 0.0], [3.0, 4.0, 0.0],
            [5.0, 6.0, 0.0], [7.0, 8.0, 0.0],
        ], dtype=np.float64)
        pcd.colors = np.array([
            [255, 0, 0], [0, 255, 0],
            [0, 0, 255], [128, 128, 128],
        ], dtype=np.uint8)
        pcd.normals = np.array([
            [0., 0., 1.], [0., 1., 0.],
            [1., 0., 0.], [0., 0., -1.],
        ], dtype=np.float64)
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = pcd.crop(poly)
        np.testing.assert_array_equal(result.colors, pcd.colors)
        np.testing.assert_array_equal(result.normals, pcd.normals)

    def test_crop_refreshes_display_cache(self):
        """After crop, _btf_print must be refreshed for the result."""
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
        ], dtype=np.float64)
        pcd.crs = "EPSG:32654"
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = pcd.crop(poly)
        assert result._btf_print.strip()
        assert {"x", "y", "z"}.issubset(set(result._btf_print.splitlines()[0].split()))
        assert result.shape[0] == 2

    def test_crop_preserves_offset_on_empty_intersection(self):
        pcd = idp.PointCloud()
        pcd.points = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        pcd.offset = [100.0, 200.0, 50.0]
        pcd.crs = "EPSG:32654"
        poly = Polygon([(999, 999), (1000, 999), (1000, 1000), (999, 1000)])
        result = pcd.crop(poly)
        np.testing.assert_array_almost_equal(result.offset, pcd.offset)
        assert result.crs.equals(pcd.crs)
        assert not result.has_points()


# -----------------------------------------------------------
#  #93 — no-CRS workflow
# -----------------------------------------------------------

class TestNoCRSWorkflow93:
    """#93: indoor / no-GPS point clouds work without CRS."""

    @pytest.fixture
    def nocrc_pcd(self):
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0], [10.0, 11.0, 12.0],
        ], dtype=np.float64)
        pcd.colors = np.array([
            [255, 0, 0], [0, 255, 0],
            [0, 0, 255], [128, 128, 128],
        ], dtype=np.uint8)
        return pcd

    def test_construct_without_crs(self, nocrc_pcd):
        assert nocrc_pcd.crs is None

    def test_crop_without_crs(self, nocrc_pcd):
        poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        result = nocrc_pcd.crop(poly)
        assert result.crs is None
        assert result.shape[0] == 4

    def test_select_by_index_without_crs(self, nocrc_pcd):
        result = nocrc_pcd.select_by_index(np.array([0, 2]))
        assert result.crs is None
        assert result.shape[0] == 2

    def test_crop_roi_dict_without_crs(self, nocrc_pcd):
        roi = idp.ROI()
        roi["box"] = np.array([
            [0, 0], [20, 0], [20, 20], [0, 20], [0, 0],
        ], dtype=float)
        result = nocrc_pcd.crop(roi)
        assert isinstance(result, dict)
        assert "box" in result
        assert result["box"].crs is None
        assert result["box"].has_points()

    def test_save_without_crs(self, nocrc_pcd, tmp_path):
        out = tmp_path / "nocrc.ply"
        written = pcmod.write_point_cloud(out, nocrc_pcd)
        assert written.exists()
        assert not os.path.exists(str(out.with_suffix(".crs")))
        pcd2 = pcmod.read_point_cloud(written)
        assert pcd2.crs is None

    def test_save_las_without_crs(self, nocrc_pcd, tmp_path):
        out = tmp_path / "nocrc.las"
        written = pcmod.write_point_cloud(out, nocrc_pcd)
        assert written.exists()
        assert not os.path.exists(str(out.with_suffix(".crs")))

    def test_crs_none_to_crs_raises(self, nocrc_pcd):
        with pytest.raises(TypeError, match="no CRS"):
            nocrc_pcd.to_crs("EPSG:4326")

    def test_crs_none_change_crs_raises(self, nocrc_pcd):
        with pytest.raises(TypeError, match="no CRS"):
            nocrc_pcd.change_crs("EPSG:4326")

    def test_assign_crs_then_crop_preserves(self, nocrc_pcd):
        nocrc_pcd.crs = "EPSG:4326"
        poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        result = nocrc_pcd.crop(poly)
        assert result.crs is not None
        assert result.crs.equals(nocrc_pcd.crs)

    def test_empty_pointcloud_len_is_zero(self):
        pcd = idp.PointCloud()
        assert len(pcd) == 0


class TestSaveCRSSidecar:
    """PointCloud.save() should preserve CRS metadata on disk."""

    def test_save_writes_crs_sidecar(self, tmp_path):
        pcd = idp.PointCloud()
        pcd.points = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        pcd.crs = "EPSG:32654"

        out = tmp_path / "with_crs.ply"
        pcd.save(out)

        assert out.exists()
        assert out.with_suffix(".crs").exists()
        loaded = idp.PointCloud(out)
        assert loaded.crs is not None
        assert loaded.crs.equals(pcd.crs)


# -----------------------------------------------------------
#  #102 — import scope: no Open3D required
# -----------------------------------------------------------

class TestNoOpen3DImport102:
    """#102: importing easyidp.pointcloud must not require Open3D."""

    def test_subprocess_import_no_open3d(self):
        """Import in a clean subprocess must not load Open3D."""
        code = (
            "import easyidp.pointcloud\n"
            "import sys\n"
            "has = any('open3d' in k.lower() for k in sys.modules)\n"
            "print('HAS_OPEN3D=' + str(has))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
        )
        assert "HAS_OPEN3D=False" in result.stdout, result.stdout

    def test_pointcloud_has_no_open3d_attribute(self):
        assert not hasattr(pcmod, "open3d")
        assert not hasattr(pcmod, "to_open3d")
        assert not hasattr(pcmod.PointCloud, "to_open3d")


# -----------------------------------------------------------
#  #78 — reconstruction compatibility
# -----------------------------------------------------------

class TestReconstructionPcd78:
    """#78: existing reconstruction pcd assignment still works."""

    def test_pix4d_load_pcd_returns_pointcloud(self, test_data):
        p4d = idp.Pix4D(
            project_path=test_data.pix4d.lotus_folder,
            param_folder=test_data.pix4d.lotus_param,
        )
        pcd_path = test_data.pix4d.lotus_pcd
        p4d.load_pcd(pcd_path)
        assert isinstance(p4d.pcd, idp.PointCloud)
        assert p4d.pcd.has_points()
        assert p4d.pcd.colors is not None

    def test_recons_pcd_setter_accepts_pointcloud(self):
        rc = idp.reconstruct.Recons()
        pcd = idp.PointCloud()
        pcd.points = np.array([[1, 2, 3]], dtype=float)
        rc.pcd = pcd
        assert rc.pcd is pcd

    def test_recons_pcd_setter_accepts_path(self, tmp_path, test_data):
        out = tmp_path / "test.ply"
        pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        pcmod.write_point_cloud(out, pcd)

        rc = idp.reconstruct.Recons()
        rc.pcd = str(out)
        assert isinstance(rc.pcd, idp.PointCloud)
        assert rc.pcd.has_points()

    def test_pix4d_pcd_is_pointcloud(self, test_data):
        p4d = idp.Pix4D(
            project_path=test_data.pix4d.maize_folder,
        )
        assert isinstance(p4d.pcd, idp.PointCloud)
        assert p4d.pcd.has_points()


# -----------------------------------------------------------
#  IO edge cases
# -----------------------------------------------------------

class TestIOEdgeCases:
    """IO edge-case regression tests."""

    def test_no_color_las_roundtrip(self, tmp_path):
        """LAS point format 2 always produces RGB; verify points survive."""
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        pcd = idp.PointCloud()
        pcd.points = data
        assert not pcd.has_colors()

        out = tmp_path / "nocolor.las"
        pcmod.write_point_cloud(out, pcd, format="las")
        pcd2 = pcmod.read_point_cloud(out)
        np.testing.assert_array_almost_equal(pcd2.points, data)
        # LAS point format 2 always has RGB; writer fills zeros if absent.
        # Verifying point data is the important regression check here.

    def test_no_color_laz_roundtrip(self, tmp_path):
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float64)
        pcd = idp.PointCloud()
        pcd.points = data

        out = tmp_path / "nocolor.laz"
        pcmod.write_point_cloud(out, pcd, format="laz")
        pcd2 = pcmod.read_point_cloud(out)
        np.testing.assert_array_almost_equal(pcd2.points, data)

    def test_las_crs_sidecar_roundtrip(self, tmp_path):
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [1.0, 2.0, 50.0],
            [3.0, 4.0, 52.0],
        ], dtype=np.float64)
        pcd.crs = "EPSG:32654"

        out = tmp_path / "with_crs.las"
        pcmod.write_point_cloud(out, pcd, format="las")

        crs_path = tmp_path / "with_crs.crs"
        assert crs_path.exists()
        assert "EPSG:32654" in crs_path.read_text()

        pcd2 = pcmod.read_point_cloud(out)
        assert pcd2.crs is not None
        assert pcd2.crs.to_epsg() == 32654

    def test_ply_crs_sidecar_roundtrip(self, tmp_path):
        pcd = idp.PointCloud()
        pcd.points = np.array([
            [368001.0, 3955501.0, 50.0],
        ], dtype=np.float64)
        pcd.crs = "EPSG:4326"

        out = tmp_path / "with_crs.ply"
        pcmod.write_point_cloud(out, pcd, format="ply")

        crs_path = tmp_path / "with_crs.crs"
        assert crs_path.exists()

        pcd2 = pcmod.read_point_cloud(out)
        assert pcd2.crs is not None
        assert pcd2.crs.to_epsg() == 4326

    def test_suffix_mismatch_warns_writer(self, tmp_path):
        pcd = idp.PointCloud()
        pcd.points = np.array([[1, 2, 3]], dtype=float)
        out = tmp_path / "mismatch.las"
        with pytest.warns(UserWarning, match="Format mismatch"):
            result = pcmod.write_point_cloud(out, pcd, format="ply")
        assert result.suffix == ".ply"

    def test_no_suffix_no_format_raises_writer(self, tmp_path):
        pcd = idp.PointCloud()
        pcd.points = np.array([[1, 2, 3]], dtype=float)
        out = tmp_path / "nosfx"
        with pytest.raises(ValueError, match="Cannot determine format"):
            pcmod.write_point_cloud(out, pcd)


# -----------------------------------------------------------
#  Crop-engine equivalence
# -----------------------------------------------------------

class TestCropEngineEquivalence:
    """crop(), crop_point_cloud(), crop_polygon(), crop_rois() consistency."""

    @pytest.fixture
    def pcd_utm(self):
        pytest.importorskip("shapely")
        pcd = idp.PointCloud()
        rng = np.random.default_rng(42)
        xy = rng.uniform(low=368000, high=368020, size=(50, 2))
        z = rng.uniform(low=80, high=120, size=(50,))
        pcd.points = np.column_stack([xy, z]).astype(np.float64)
        pcd.crs = "EPSG:32654"
        return pcd

    def test_crop_rois_vs_crop_roi_same_keys(self, pcd_utm):
        roi = idp.ROI()
        roi["a"] = np.array([
            [368000, 3955500],
            [368010, 3955500],
            [368010, 3955510],
            [368000, 3955510],
            [368000, 3955500],
        ], dtype=float)
        roi["b"] = np.array([
            [368005, 3955505],
            [368015, 3955505],
            [368015, 3955515],
            [368005, 3955515],
            [368005, 3955505],
        ], dtype=float)
        roi.crs = "EPSG:32654"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            out1 = pcd_utm.crop_rois(roi)
        out2 = pcd_utm.crop(roi)

        assert set(out1.keys()) == set(out2.keys())
        for k in out1:
            assert out1[k].shape[0] == out2[k].shape[0]

    def test_crop_point_cloud_vs_crop_same_point_count(self, pcd_utm):
        """crop_point_cloud() and crop(Polygon) select same points."""
        xmin, ymin = pcd_utm._points_xy.min(axis=0) - 1
        xmax, ymax = pcd_utm._points_xy.max(axis=0) + 1
        polygon_xy = np.array([
            [xmin, ymin], [xmax, ymin],
            [xmax, ymax], [xmin, ymax],
            [xmin, ymin],
        ])
        poly = Polygon([
            (xmin, ymin), (xmax, ymin),
            (xmax, ymax), (xmin, ymax),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            out_legacy = pcd_utm.crop_point_cloud(polygon_xy)
        out_new = pcd_utm.crop(poly)

        assert out_legacy.shape[0] == out_new.shape[0]
        assert out_legacy.shape[0] > 0

    def test_crop_polygon_vs_crop_same_count(self, pcd_utm):
        polygon_xy = np.array([
            [368000, 3955500],
            [368010, 3955500],
            [368010, 3955510],
            [368000, 3955510],
            [368000, 3955500],
        ])
        poly = Polygon([
            (368000, 3955500), (368010, 3955500),
            (368010, 3955510), (368000, 3955510),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ndarray_result = pcd_utm.crop_polygon(polygon_xy)
        pc_result = pcd_utm.crop(poly)

        assert ndarray_result.shape[0] == pc_result.shape[0]
        np.testing.assert_array_almost_equal(ndarray_result, pc_result.points)


# -----------------------------------------------------------
#  #121 — DEM/CHM deferred (non-goal documentation)
# -----------------------------------------------------------

class TestDeferredNonGoals121:
    """#121 & #102 deferred features are not implemented."""

    def test_pointcloud_has_no_dem_method(self):
        """EasyIDP PointCloud must not expose DEM/CHM processing."""
        assert not hasattr(idp.PointCloud, "compute_dem")
        assert not hasattr(idp.PointCloud, "compute_chm")
        assert not hasattr(idp.PointCloud, "rasterize")

    def test_pointcloud_has_no_open3d_method(self):
        """#102: no Open3D conversion method."""
        assert not hasattr(idp.PointCloud, "to_open3d")

    def test_pointcloud_module_has_no_processing(self):
        """pointcloud module must not expose high-level processing."""
        assert not hasattr(pcmod, "compute_dem")
        assert not hasattr(pcmod, "compute_chm")
        assert not hasattr(pcmod, "rasterize")

    def test_no_open3d_import_in_module_source(self):
        """Verify Open3D is not imported anywhere in the pointcloud package."""
        import easyidp.pointcloud as pc

        pkg_dir = os.path.dirname(pc.__file__)
        for root, _dirs, files in os.walk(pkg_dir):
            for fn in files:
                if fn.endswith(".py") and not fn.startswith("__"):
                    fpath = os.path.join(root, fn)
                    content = open(fpath).read()
                    assert "open3d" not in content.lower(), (
                        f"Open3D reference found in {fpath}"
                    )
                    assert "o3d" not in content.lower().split(), (
                        f"o3d reference found in {fpath}"
                    )
