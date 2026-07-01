"""Internal PLY reading and writing helpers.

These are not part of the public API. Use :func:`read_point_cloud` and
:func:`write_point_cloud` instead.
"""

import numpy as np
import numpy.lib.recfunctions as rfn
from datetime import datetime
from plyfile import PlyData, PlyElement

import easyidp as idp
from easyidp.logger import logger


def read(path):
    """Read a PLY file and return (points, colors, normals).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.ply`` file.

    Returns
    -------
    ndarray, ndarray or None, ndarray or None
        ``(points (N,3), colors (N,3) uint8, normals (N,3))``.
        *colors* and *normals* are ``None`` when not present.

    Examples
    --------
    >>> from easyidp.pointcloud.io import ply
    >>> points, colors, normals = ply.read("cloud.ply")
    """
    cloud_data = PlyData.read(str(path)).elements[0].data
    ply_names = cloud_data.dtype.names

    points = np.vstack(
        (cloud_data["x"], cloud_data["y"], cloud_data["z"])
    ).T

    if "red" in ply_names:
        colors = np.vstack(
            (cloud_data["red"], cloud_data["green"], cloud_data["blue"])
        ).T
    elif "diffuse_red" in ply_names:
        colors = np.vstack(
            (
                cloud_data["diffuse_red"],
                cloud_data["diffuse_green"],
                cloud_data["diffuse_blue"],
            )
        ).T
    else:
        logger.warning(f"Can not find color info in {ply_names}")
        colors = None

    if colors is not None:
        colors = colors.astype(np.uint8)

    if "nx" in ply_names:
        normals = np.vstack(
            (cloud_data["nx"], cloud_data["ny"], cloud_data["nz"])
        ).T
    else:
        normals = None

    return points, colors, normals


def write(path, points, colors, normals=None, binary=True):
    """Write a PLY file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output path (``.ply`` suffix).
    points : ndarray of shape (N, 3)
        XYZ coordinates.
    colors : ndarray of shape (N, 3) or None
        RGB values in [0, 255] as ``uint8``, or ``None`` to omit
        color vertex properties.
    normals : ndarray of shape (N, 3) or None, optional
        Normal vectors.
    binary : bool, optional
        ``True`` for binary PLY, ``False`` for ASCII PLY. Default ``True``.

    Returns
    -------
    None

    Examples
    --------
    >>> from easyidp.pointcloud.io import ply
    >>> ply.write("cloud.ply", points, colors, binary=True)
    """
    struct_points = np.rec.fromarrays(points.T, names="x, y, z")
    merged_list = [struct_points]

    if colors is not None:
        struct_colors = np.rec.fromarrays(
            colors.T,
            dtype=np.dtype(
                [("red", np.uint8), ("green", np.uint8), ("blue", np.uint8)]
            ),
        )
        merged_list.append(struct_colors)

    if normals is not None:
        struct_normals = np.rec.fromarrays(
            normals.T, names="nx, ny, nz"
        )
        merged_list.append(struct_normals)

    struct_merge = rfn.merge_arrays(merged_list, flatten=True, usemask=False)

    el = PlyElement.describe(
        struct_merge,
        "vertex",
        comments=[
            f"Created by EasyIDP v{idp.__version__}",
            f"Created {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",
        ],
    )

    if binary:
        PlyData([el]).write(str(path))
    else:
        PlyData([el], text=True).write(str(path))
