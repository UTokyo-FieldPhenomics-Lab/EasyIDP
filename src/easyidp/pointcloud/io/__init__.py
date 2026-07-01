"""Public IO dispatchers for point cloud reading and writing.

Provides :func:`read_point_cloud` and :func:`write_point_cloud` that
dispatch by explicit ``format`` or by filename suffix.
"""

import warnings
from pathlib import Path

import numpy as np

from easyidp.pointcloud.core import PointCloud
from . import las as las_io
from . import ply as ply_io

__all__ = ["read_point_cloud", "write_point_cloud"]

_SUPPORTED = frozenset({"ply", "las", "laz"})


def read_point_cloud(path, format=None, offset=None):
    """Read a point cloud file and return a new :class:`PointCloud`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the point cloud file (``.ply``, ``.las``, ``.laz``).
    format : str or None, optional
        Explicit format (``"ply"``, ``"las"``, ``"laz"``, case-insensitive).
        When None, the format is inferred from the path suffix.
    offset : list or ndarray of shape (3,) or None, optional
        User-supplied offset for the returned PointCloud.

    Returns
    -------
    PointCloud

    Raises
    ------
    ValueError
        If *format* is unsupported.
    FileNotFoundError
        If *path* does not exist.

    Examples
    --------
    Read by suffix:

    >>> pcd = idp.pointcloud.read_point_cloud("cloud.ply")
    >>> isinstance(pcd, idp.PointCloud)
    True

    Read by explicit format:

    >>> pcd = idp.pointcloud.read_point_cloud("cloud.data", format="las")

    Read a Pix4D-offset local point cloud:

    >>> pcd = idp.pointcloud.read_point_cloud(
    ...     "hasu_tanashi.ply",
    ...     offset=[368043, 3955495, 98],
    ... )
    """
    path = Path(path)
    fmt = _resolve_format(path, format)

    if fmt == "ply":
        pts, cls, nms = ply_io.read(path)
    else:  # las, laz
        pts, cls, nms = las_io.read(path)

    pcd = PointCloud()
    if offset is not None:
        pcd.offset = pcd._offset_type_check(offset)

    if abs(np.max(pts)) > 65536:
        if not np.any(pcd.offset):
            pcd._offset = np.floor(pts.min(axis=0) / 100) * 100
        pcd._points = pts - pcd._offset
    else:
        pcd._points = pts

    pcd.colors = cls
    pcd.normals = nms
    pcd.shape = pts.shape
    pcd.file_path = str(path.resolve())
    pcd.file_ext = f".{fmt}"
    pcd._update_btf_print()

    # check sidecar CRS
    _load_sidecar_crs(pcd, path)

    return pcd


def write_point_cloud(target, pcd, format=None):
    """Write a :class:`PointCloud` to a file.

    Parameters
    ----------
    target : str or pathlib.Path
        Target file path. Suffix is used to infer format unless *format*
        is given.
    pcd : PointCloud
        The point cloud to write.
    format : str or None, optional
        Explicit format (``"ply"``, ``"las"``, ``"laz"``, case-insensitive).

    Returns
    -------
    pathlib.Path
        The actual file path written to (may differ from *target* when
        the suffix is adjusted).

    Raises
    ------
    ValueError
        If the format cannot be determined or is unsupported.

    Examples
    --------
    Write by suffix:

    >>> out_path = idp.pointcloud.write_point_cloud("cloud.ply", pcd)
    >>> out_path.name
    'cloud.ply'

    Write by explicit format:

    >>> idp.pointcloud.write_point_cloud("cloud_output", pcd, format="laz")

    If the suffix and format disagree, EasyIDP writes to an adjusted path:

    >>> idp.pointcloud.write_point_cloud("cloud.las", pcd, format="ply")
    PosixPath('cloud.las.ply')
    """
    target = Path(target)
    target_suffix = target.suffix.lstrip(".").lower()

    if format is None and not target_suffix:
        raise ValueError(
            "Cannot determine format from target path with no suffix. "
            "Specify format=..."
        )

    fmt = _resolve_format(target, format)

    # Resolve target filename
    if format is not None:
        if target_suffix and target_suffix != fmt:
            warnings.warn(
                f"Format mismatch: target suffix '.{target_suffix}' vs "
                f"format '{fmt}'; writing to "
                f"'{target}.{fmt}'.",
            )
            target = target.with_name(f"{target.name}.{fmt}")
        elif not target_suffix:
            target = target.with_suffix(f".{fmt}")

    abs_pts = pcd._points + pcd.offset if pcd._points is not None else np.zeros((0, 3))

    if fmt == "ply":
        cls_arr = pcd.colors
        ply_io.write(
            target, abs_pts, cls_arr, normals=pcd.normals,
        )
    else:  # las, laz
        las_io.write(
            target, abs_pts, pcd.colors,
            normals=pcd.normals, offset=pcd.offset,
        )

    # sidecar CRS
    if pcd.crs is not None:
        crs_path = target.with_suffix(".crs")
        crs_path.write_text(pcd.crs.to_string())

    return target


def _resolve_format(path, format):
    """Resolve the canonical point-cloud format string.

    Parameters
    ----------
    path : pathlib.Path
        Source or target path whose suffix may identify the format.
    format : str or None
        Explicit format name. Supported values are ``"ply"``, ``"las"``,
        and ``"laz"``; matching is case-insensitive.

    Returns
    -------
    str
        Canonical lowercase format name.

    Raises
    ------
    ValueError
        If neither the explicit format nor path suffix is supported.

    Examples
    --------
    >>> from pathlib import Path
    >>> _resolve_format(Path("cloud.PLY"), None)
    'ply'
    >>> _resolve_format(Path("cloud.bin"), "LAS")
    'las'
    """
    if format is not None:
        fmt = format.lower()
        if fmt not in _SUPPORTED:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported: {sorted(_SUPPORTED)}"
            )
        return fmt

    suffix = path.suffix.lstrip(".").lower()
    if suffix not in _SUPPORTED:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {sorted(_SUPPORTED)}"
        )
    return suffix


def _load_sidecar_crs(pcd, path):
    """Load CRS from a ``.crs`` sidecar file if it exists.

    Parameters
    ----------
    pcd : easyidp.PointCloud
        Point cloud object to mutate when a valid sidecar is found.
    path : pathlib.Path
        Point-cloud file path whose suffix will be replaced with ``.crs``.

    Returns
    -------
    None
        Mutates ``pcd.crs`` when the sidecar file exists and is valid.

    Notes
    -----
    Invalid sidecar files are ignored with a debug log so point-cloud
    reading remains non-fatal.

    Examples
    --------
    >>> pcd = idp.PointCloud()
    >>> _load_sidecar_crs(pcd, Path("cloud.ply"))
    """
    import pyproj
    from easyidp.logger import logger
    crs_path = path.with_suffix(".crs")
    if crs_path.exists():
        try:
            pcd.crs = pyproj.CRS.from_user_input(crs_path.read_text().strip())
        except Exception as e:
            logger.debug(
                f"Found CRS file [{crs_path}] but failed to load: {e}"
            )
