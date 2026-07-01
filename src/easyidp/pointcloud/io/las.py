"""Internal LAS/LAZ reading and writing helpers.

These are not part of the public API. Use :func:`read_point_cloud` and
:func:`write_point_cloud` instead.
"""

import numpy as np
from datetime import datetime

import laspy

import easyidp as idp


def read(path):
    """Read a LAS or LAZ file and return (points, colors, normals).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.las`` or ``.laz`` file.

    Returns
    -------
    ndarray, ndarray or None, ndarray or None
        ``(points (N,3), colors (N,3) uint8, normals (N,3))``.
        *colors* and *normals* are ``None`` when not present.

    Examples
    --------
    >>> from easyidp.pointcloud.io import las
    >>> points, colors, normals = las.read("cloud.las")
    """
    las = laspy.read(str(path))

    points = np.vstack([las.x, las.y, las.z]).T

    colors = (
        np.vstack([las.points["red"], las.points["green"], las.points["blue"]]).T
        / 256
    )
    colors = colors.astype(np.uint8)

    if "normal x" in las.points.array.dtype.names:
        normals = np.vstack(
            [las.points["normal x"], las.points["normal y"], las.points["normal z"]]
        ).T
    else:
        normals = None

    return points, colors, normals


def write(path, points, colors, normals=None,
                    offset=np.array([0.0, 0.0, 0.0]), decimal=5):
    """Write a LAS or LAZ file (LAS version 1.2, point format 2).

    Parameters
    ----------
    path : str or pathlib.Path
        Output path (``.las`` or ``.laz`` suffix).
    points : ndarray of shape (N, 3)
        Absolute XYZ coordinates.
    colors : ndarray of shape (N, 3) or None
        RGB values in [0, 255] as ``uint8``, or ``None`` to use zeros.
    normals : ndarray of shape (N, 3) or None, optional
        Normal vectors.
    offset : ndarray of shape (3,), optional
        LAS header offset, default ``[0, 0, 0]``.
    decimal : int, optional
        Scale precision exponent, default 5.

    Returns
    -------
    None

    Examples
    --------
    >>> from easyidp.pointcloud.io import las
    >>> las.write("cloud.las", points, colors, offset=np.array([0., 0., 0.]))
    """
    n_pts = points.shape[0]
    if colors is None:
        colors = np.zeros((n_pts, 3), dtype=np.uint8)

    header = laspy.LasHeader(point_format=2, version="1.2")
    if normals is not None:
        header.add_extra_dim(
            laspy.ExtraBytesParams(name="normal x", type=np.float64))
        header.add_extra_dim(
            laspy.ExtraBytesParams(name="normal y", type=np.float64))
        header.add_extra_dim(
            laspy.ExtraBytesParams(name="normal z", type=np.float64))
    header.offsets = offset
    header.scales = np.array([float(f"1e-{decimal}")] * 3)
    header.generating_software = (
        f"EasyIDP v{idp.__version__} "
        f"on {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}"
    )

    las = laspy.LasData(header)

    las.points["x"] = points[:, 0]
    las.points["y"] = points[:, 1]
    las.points["z"] = points[:, 2]

    cls16 = colors.astype(np.uint16)
    las.points["red"] = cls16[:, 0] * 256
    las.points["green"] = cls16[:, 1] * 256
    las.points["blue"] = cls16[:, 2] * 256

    if normals is not None:
        las.points["normal x"] = normals[:, 0]
        las.points["normal y"] = normals[:, 1]
        las.points["normal z"] = normals[:, 2]

    las.write(str(path))
