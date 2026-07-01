"""Test compat module: legacy standalone functions and PointCloudCompatMixin."""

import warnings
import numpy as np

import easyidp as idp
import easyidp.pointcloud as pcmod
import easyidp.pointcloud.core as pccore
from easyidp.pointcloud import compat


STANDALONE_NAMES = [
    "read_ply",
    "read_las",
    "read_laz",
    "write_ply",
    "write_las",
    "write_laz",
]

LEGACY_METHODS = [
    "crop_polygon",
    "crop_rois",
    "crop_point_cloud",
    "read_point_cloud",
    "write_point_cloud",
]


def test_compat_module_importable():
    assert compat is not None


def test_compat_has_pointcloud_compat_mixin():
    assert hasattr(compat, "PointCloudCompatMixin")
    assert isinstance(compat.PointCloudCompatMixin, type)


def test_compat_has_six_standalone_functions():
    for name in STANDALONE_NAMES:
        assert hasattr(compat, name), f"compat missing {name}()"
        assert callable(getattr(compat, name)), f"compat.{name} is not callable"


def test_core_module_no_longer_has_standalone_functions():
    """After migration, core.py must not expose legacy standalone functions."""
    for name in STANDALONE_NAMES:
        assert not hasattr(pccore, name), (
            f"core module should not expose {name}() anymore; "
            f"it was moved to compat.py"
        )


def test_core_module_no_longer_has_legacy_methods_directly():
    """Legacy methods should be inherited from PointCloudCompatMixin, not
    defined directly in PointCloud.__dict__."""
    for name in LEGACY_METHODS:
        assert name not in pccore.PointCloud.__dict__, (
            f"'{name}' is still defined directly in PointCloud.__dict__; "
            f"should be inherited from PointCloudCompatMixin"
        )


def test_legacy_methods_inherited_from_mixin():
    """PointCloud must inherit from PointCloudCompatMixin."""
    assert compat.PointCloudCompatMixin in pccore.PointCloud.__mro__, (
        "PointCloud must have PointCloudCompatMixin in its MRO"
    )


def test_legacy_methods_callable_on_instance():
    pcd = idp.PointCloud()
    for name in LEGACY_METHODS:
        assert hasattr(pcd, name), f"PointCloud instance missing {name}()"
        assert callable(getattr(pcd, name)), f"pcd.{name} is not callable"


def test_standalone_functions_warn(test_data):
    """Public easyidp.pointcloud.read_las etc. must warn."""
    las_path = test_data.pcd.lotus_las
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        points, colors, normals = pcmod.read_las(las_path)
        assert len(w) >= 1
        assert issubclass(w[0].category, FutureWarning)
    assert points is not None


def test_crop_polygon_warns(test_data):
    pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
    polygon = np.array([
        [-18.42576599, -16.10819054],
        [-18.00066757, -18.05295944],
        [-16.05021095, -17.63488388],
        [-16.46848488, -15.66774559],
        [-18.42576599, -16.10819054],
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pcd.crop_polygon(polygon)
        assert len(w) >= 1
        assert issubclass(w[0].category, FutureWarning)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] > 0


def test_crop_point_cloud_warns(test_data):
    pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
    polygon = np.array([
        [-18.42576599, -16.10819054],
        [-18.00066757, -18.05295944],
        [-16.05021095, -17.63488388],
        [-16.46848488, -15.66774559],
        [-18.42576599, -16.10819054],
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cropped = pcd.crop_point_cloud(polygon)
        assert len(w) >= 1
        assert issubclass(w[0].category, FutureWarning)
    assert isinstance(cropped, idp.PointCloud)
    assert cropped.shape[0] > 0


def test_write_point_cloud_works(test_data):
    pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
    out_path = test_data.pcd.out / "test_compat_write_pcd.ply"
    if out_path.exists():
        out_path.unlink()
    pcd.write_point_cloud(out_path)
    assert out_path.exists()


def test_save_delegates_to_write_point_cloud(test_data):
    """PointCloud.save() should work, delegating to write_point_cloud."""
    pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
    out_path = test_data.pcd.out / "test_compat_save.ply"
    if out_path.exists():
        out_path.unlink()
    pcd.save(out_path)
    assert out_path.exists()


def test_read_point_cloud_legacy_works(test_data):
    pcd = idp.PointCloud()
    pcd.read_point_cloud(test_data.pcd.lotus_ply_bin)
    assert pcd.has_points()
    assert pcd.shape[0] > 0


def test_standalone_on_pcmod_still_available():
    """All legacy standalone functions must be accessible on
    easyidp.pointcloud."""
    for name in STANDALONE_NAMES:
        assert hasattr(pcmod, name), f"pcmod missing {name}()"
        assert callable(getattr(pcmod, name)), f"pcmod.{name} is not callable"
