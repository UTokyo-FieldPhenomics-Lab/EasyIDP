from pathlib import Path

import easyidp as idp
import easyidp.pointcloud as pcmod


EXPECTED_STANDALONE = [
    "read_ply",
    "read_las",
    "read_laz",
    "write_ply",
    "write_laz",
    "write_las",
]


def test_pointcloud_is_package():
    """Phase 1: easyidp.pointcloud should be a package, not a flat module."""
    assert hasattr(pcmod, "__path__"), "easyidp.pointcloud must be a package"


def test_idp_pointcloud_is_pcmod_pointcloud():
    """idp.PointCloud and pcmod.PointCloud must be the same class."""
    assert idp.PointCloud is pcmod.PointCloud


def test_standalone_functions_on_pcmod():
    """All standalone IO functions must be accessible on easyidp.pointcloud."""
    for name in EXPECTED_STANDALONE:
        assert hasattr(pcmod, name), f"pcmod missing {name}()"
        assert callable(getattr(pcmod, name)), f"pcmod.{name} is not callable"


def test_core_importable():
    """Phase 1: PointCloud must be importable from easyidp.pointcloud.core."""
    from easyidp.pointcloud.core import PointCloud

    assert PointCloud is pcmod.PointCloud


def test_legacy_standalone_file_removed():
    """Ensure test_pointcloud_legacy.py has been merged into tests/test_pointcloud/."""
    legacy_file = Path(__file__).parent.parent / "test_pointcloud_legacy.py"
    assert not legacy_file.exists(), (
        f"{legacy_file} should be removed after merging "
        f"into tests/test_pointcloud/"
    )
