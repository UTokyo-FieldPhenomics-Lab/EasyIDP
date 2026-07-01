"""Legacy compatibility wrappers for pointcloud standalone functions
and deprecated PointCloud instance methods.

The top-level functions emit ``FutureWarning`` and delegate to the
corresponding ``io`` backend.  ``PointCloudCompatMixin`` carries the
old *PointCloud* instance methods that are kept for backward
compatibility.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pyproj
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

import easyidp as idp
from easyidp.logger import logger

from .geometry import query_indices_by_polygon
from .io import las as las_io
from .io import ply as ply_io


# ---------------------------------------------------------------------------
# Standalone legacy read/write wrappers
# ---------------------------------------------------------------------------

def read_ply(ply_path):
    """Read a PLY file with the legacy standalone API.

    .. deprecated:: 2.1.0
        Use :func:`easyidp.pointcloud.read_point_cloud` instead.

    Parameters
    ----------
    ply_path : str or pathlib.Path
        Path to a PLY point-cloud file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None, numpy.ndarray or None]
        Raw ``(points, colors, normals)`` arrays.

    Warns
    -----
    FutureWarning
        Always emitted to guide migration to ``read_point_cloud()``.

    Examples
    --------
    >>> points, colors, normals = idp.pointcloud.read_ply("cloud.ply")
    >>> pcd = idp.pointcloud.read_point_cloud("cloud.ply")
    """
    warnings.warn(
        "read_ply() is deprecated, use read_point_cloud() instead.",
        FutureWarning,
        stacklevel=2,
    )
    return ply_io.read(ply_path)


def read_las(las_path):
    """Read a LAS file with the legacy standalone API.

    .. deprecated:: 2.1.0
        Use :func:`easyidp.pointcloud.read_point_cloud` instead.

    Parameters
    ----------
    las_path : str or pathlib.Path
        Path to a LAS point-cloud file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None, numpy.ndarray or None]
        Raw ``(points, colors, normals)`` arrays.

    Warns
    -----
    FutureWarning
        Always emitted to guide migration to ``read_point_cloud()``.

    Examples
    --------
    >>> points, colors, normals = idp.pointcloud.read_las("cloud.las")
    >>> pcd = idp.pointcloud.read_point_cloud("cloud.las")
    """
    warnings.warn(
        "read_las() is deprecated, use read_point_cloud() instead.",
        FutureWarning,
        stacklevel=2,
    )
    return las_io.read(las_path)


def read_laz(laz_path):
    """Read a LAZ file with the legacy standalone API.

    .. deprecated:: 2.1.0
        Use :func:`easyidp.pointcloud.read_point_cloud` instead.

    Parameters
    ----------
    laz_path : str or pathlib.Path
        Path to a LAZ point-cloud file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None, numpy.ndarray or None]
        Raw ``(points, colors, normals)`` arrays.

    Warns
    -----
    FutureWarning
        Always emitted to guide migration to ``read_point_cloud()``.

    Examples
    --------
    >>> points, colors, normals = idp.pointcloud.read_laz("cloud.laz")
    >>> pcd = idp.pointcloud.read_point_cloud("cloud.laz")
    """
    warnings.warn(
        "read_laz() is deprecated, use read_point_cloud() instead.",
        FutureWarning,
        stacklevel=2,
    )
    return las_io.read(laz_path)


def write_ply(ply_path, points, colors, normals=None, binary=True):
    """Write a PLY file with the legacy standalone API.

    .. deprecated:: 2.1.0
        Use :func:`easyidp.pointcloud.write_point_cloud` instead.

    Parameters
    ----------
    ply_path : str or pathlib.Path
        Output PLY path.
    points : numpy.ndarray of shape (N, 3)
        Absolute XYZ coordinates.
    colors : numpy.ndarray of shape (N, 3) or None
        RGB values as ``uint8``. ``None`` writes a no-color PLY.
    normals : numpy.ndarray of shape (N, 3) or None, optional
        Normal vectors.
    binary : bool, optional
        Write binary PLY when True, ASCII PLY when False.

    Returns
    -------
    None

    Warns
    -----
    FutureWarning
        Always emitted to guide migration to ``write_point_cloud()``.

    Examples
    --------
    >>> idp.pointcloud.write_ply("cloud.ply", points, colors)
    >>> pcd = idp.PointCloud()
    >>> pcd.points = points
    >>> pcd.colors = colors
    >>> idp.pointcloud.write_point_cloud("cloud.ply", pcd)
    """
    warnings.warn(
        "write_ply() is deprecated, use write_point_cloud() instead.",
        FutureWarning,
        stacklevel=2,
    )
    ply_io.write(ply_path, points, colors, normals=normals, binary=binary)


def write_laz(
    laz_path, points, colors, normals=None, offset=np.array([0.0, 0.0, 0.0]), decimal=5
):
    """Write a LAZ file with the legacy standalone API.

    .. deprecated:: 2.1.0
        Use :func:`easyidp.pointcloud.write_point_cloud` instead.

    Parameters
    ----------
    laz_path : str or pathlib.Path
        Output LAZ path.
    points : numpy.ndarray of shape (N, 3)
        Absolute XYZ coordinates.
    colors : numpy.ndarray of shape (N, 3) or None
        RGB values as ``uint8``. ``None`` writes zero-valued LAS colors.
    normals : numpy.ndarray of shape (N, 3) or None, optional
        Normal vectors.
    offset : numpy.ndarray of shape (3,), optional
        LAS header offset.
    decimal : int, optional
        Number of decimal digits used to build LAS scale values.

    Returns
    -------
    None

    Warns
    -----
    FutureWarning
        Always emitted to guide migration to ``write_point_cloud()``.

    Examples
    --------
    >>> idp.pointcloud.write_laz("cloud.laz", points, colors)
    >>> pcd = idp.PointCloud()
    >>> pcd.points = points
    >>> pcd.colors = colors
    >>> idp.pointcloud.write_point_cloud("cloud.laz", pcd)
    """
    warnings.warn(
        "write_laz() is deprecated, use write_point_cloud() instead.",
        FutureWarning,
        stacklevel=2,
    )
    las_io.write(
        laz_path, points, colors, normals=normals, offset=offset, decimal=decimal
    )


def write_las(
    las_path, points, colors, normals=None, offset=np.array([0.0, 0.0, 0.0]), decimal=5
):
    """Write a LAS file with the legacy standalone API.

    .. deprecated:: 2.1.0
        Use :func:`easyidp.pointcloud.write_point_cloud` instead.

    Parameters
    ----------
    las_path : str or pathlib.Path
        Output LAS path.
    points : numpy.ndarray of shape (N, 3)
        Absolute XYZ coordinates.
    colors : numpy.ndarray of shape (N, 3) or None
        RGB values as ``uint8``. ``None`` writes zero-valued LAS colors.
    normals : numpy.ndarray of shape (N, 3) or None, optional
        Normal vectors.
    offset : numpy.ndarray of shape (3,), optional
        LAS header offset.
    decimal : int, optional
        Number of decimal digits used to build LAS scale values.

    Returns
    -------
    None

    Warns
    -----
    FutureWarning
        Always emitted to guide migration to ``write_point_cloud()``.

    Examples
    --------
    >>> idp.pointcloud.write_las("cloud.las", points, colors)
    >>> pcd = idp.PointCloud()
    >>> pcd.points = points
    >>> pcd.colors = colors
    >>> idp.pointcloud.write_point_cloud("cloud.las", pcd)
    """
    warnings.warn(
        "write_las() is deprecated, use write_point_cloud() instead.",
        FutureWarning,
        stacklevel=2,
    )
    las_io.write(
        las_path, points, colors, normals=normals, offset=offset, decimal=decimal
    )


# ---------------------------------------------------------------------------
# PointCloudCompatMixin
# ---------------------------------------------------------------------------

class PointCloudCompatMixin:
    """Mixin carrying deprecated instance methods for backward compatibility.

    Methods defined here emit ``FutureWarning`` and delegate to the
    current ``PointCloud`` public API.
    """

    def crop_polygon(self, polygon_xy):
        """Return XYZ points inside a 2D polygon.

        .. deprecated:: 2.1.0
            Use :meth:`easyidp.PointCloud.crop` with a Shapely polygon
            and access ``result.points`` instead.

        Parameters
        ----------
        polygon_xy : array-like of shape (N, 2) or (N, >=2)
            Polygon boundary coordinates in the point-cloud XY plane.

        Returns
        -------
        numpy.ndarray of shape (M, 3)
            Absolute XYZ coordinates inside the polygon. Empty crops return
            an empty ``(0, 3)`` array.

        Warns
        -----
        FutureWarning
            Always emitted to guide migration to ``crop()``.

        Examples
        --------
        >>> xyz = pcd.crop_polygon(polygon_xy)
        >>> from shapely.geometry import Polygon
        >>> cropped = pcd.crop(Polygon(polygon_xy))
        >>> xyz = cropped.points
        """
        warnings.warn(
            "crop_polygon() is deprecated, use crop() with a "
            "Shapely Polygon instead.",
            FutureWarning,
            stacklevel=2,
        )

        if not self.has_points():
            return np.array([]).reshape(0, 3)

        poly_pts_xy = np.asarray(polygon_xy)
        if poly_pts_xy.ndim != 2 or poly_pts_xy.shape[1] < 2:
            raise ValueError(
                f"polygon_xy must have shape (n, 2), got {poly_pts_xy.shape}"
            )
        poly_pts_xy = poly_pts_xy[:, 0:2]

        indices = query_indices_by_polygon(
            self._points_xy, poly_pts_xy, tree=self.tree,
        )
        if len(indices) == 0:
            return np.array([]).reshape(0, 3)
        return self.points[indices]

    def crop_rois(self, roi, save_folder=None):
        """Crop several ROI polygons with the legacy instance API.

        .. deprecated:: 2.1.0
            Use :meth:`easyidp.PointCloud.crop` or
            :meth:`easyidp.ROI.crop` instead.

        Parameters
        ----------
        roi : easyidp.ROI or dict[str, numpy.ndarray]
            ROI collection or mapping from label to polygon coordinates.
        save_folder : str or pathlib.Path or None, optional
            Existing folder for writing each cropped point cloud. If None
            or missing, crops are returned without writing files.

        Returns
        -------
        dict[str, easyidp.PointCloud]
            Mapping from ROI label to cropped point cloud.

        Warns
        -----
        FutureWarning
            Always emitted to guide migration to ``crop()``.

        Examples
        --------
        >>> crops = pcd.crop_rois(roi)
        >>> crops = pcd.crop(roi)
        >>> crops = roi.crop(pcd)
        """
        warnings.warn(
            "crop_rois() is deprecated, use crop() with an "
            "EasyIDP ROI instead.",
            FutureWarning,
            stacklevel=2,
        )

        if not self.has_points():
            raise ValueError(
                "Could not operate when PointCloud has no points"
            )

        if not isinstance(roi, (dict, idp.ROI)):
            raise TypeError(
                f"Only <dict> and <easyidp.ROI> are accepted, "
                f"not {type(roi)}"
            )

        out_dict = {}
        pbar = tqdm(
            roi.items(),
            desc=f"Crop roi from point cloud "
                 f"[{os.path.basename(self.file_path)}]",
        )
        for k, polygon_hv in pbar:
            if save_folder is not None and Path(save_folder).exists():
                save_path = Path(save_folder) / (k + self.file_ext)
            else:
                save_path = None

            poly = ShapelyPolygon(polygon_hv[:, 0:2])
            out_dict[k] = self._crop_shapely_polygon(poly)

            if save_path is not None:
                out_dict[k].write_point_cloud(save_path)

        return out_dict

    def crop_point_cloud(self, polygon_xy):
        """Crop one polygon with the legacy ndarray API.

        .. deprecated:: 2.1.0
            Use :meth:`easyidp.PointCloud.crop` with a Shapely polygon
            instead.

        Parameters
        ----------
        polygon_xy : numpy.ndarray of shape (N, 2)
            Polygon boundary coordinates in the point-cloud XY plane.

        Returns
        -------
        easyidp.PointCloud or None
            Cropped point cloud, or None when no points are selected.

        Warns
        -----
        FutureWarning
            Always emitted to guide migration to ``crop()``.

        Examples
        --------
        >>> cropped = pcd.crop_point_cloud(polygon_xy)
        >>> from shapely.geometry import Polygon
        >>> cropped = pcd.crop(Polygon(polygon_xy))
        """
        warnings.warn(
            "crop_point_cloud() is deprecated, use crop() with a "
            "Shapely Polygon instead.",
            FutureWarning,
            stacklevel=2,
        )

        if not isinstance(polygon_xy, np.ndarray):
            raise TypeError(
                f"Only numpy ndarray are supported as `polygon_xy` "
                f"inputs, not {type(polygon_xy)}"
            )

        if len(polygon_xy.shape) != 2 or polygon_xy.shape[1] != 2:
            raise IndexError(
                f"Please only spcify shape like (N, 2), "
                f"not {polygon_xy.shape}"
            )

        indices = query_indices_by_polygon(
            self._points_xy, polygon_xy, tree=self.tree,
        )

        if len(indices) > 0:
            return self.select_by_index(indices)
        else:
            logger.warning(
                "Cropped 0 point in given polygon. Please check "
                "whether the coords is correct."
            )
            return None

    def read_point_cloud(self, pcd_path):
        """Load a point-cloud file into the current object.

        .. deprecated:: 2.1.0
            Prefer constructing a new :class:`easyidp.PointCloud` or using
            :func:`easyidp.pointcloud.read_point_cloud`.

        Parameters
        ----------
        pcd_path : str or pathlib.Path
            Path to a ``.ply``, ``.las``, or ``.laz`` point-cloud file.

        Returns
        -------
        None
            Mutates the current ``PointCloud`` in place.

        Examples
        --------
        >>> pcd = idp.PointCloud()
        >>> pcd.read_point_cloud("cloud.ply")
        >>> pcd = idp.PointCloud("cloud.ply")
        >>> pcd = idp.pointcloud.read_point_cloud("cloud.ply")
        """
        if not os.path.exists(pcd_path):
            logger.warning(
                f"Can not find file [{pcd_path}], skip loading"
            )
            return

        pcd_path = Path(pcd_path)
        suffix = pcd_path.suffix
        if suffix == ".ply":
            pts, cls, nms = ply_io.read(pcd_path)
        elif suffix in (".laz", ".las"):
            pts, cls, nms = las_io.read(pcd_path)
        else:
            raise IOError(
                "Only support point cloud file format ['*.ply', "
                "'*.laz', '*.las']"
            )

        if self.has_points():
            self.clear()

        self.file_ext = suffix
        self.file_path = str(pcd_path.resolve())

        if abs(np.max(pts)) > 65536:
            if not np.any(self._offset):
                self._offset = np.floor(pts.min(axis=0) / 100) * 100
            self._points = pts - self._offset
        else:
            self._points = pts

        self.colors = cls
        self.normals = nms
        self.shape = pts.shape

        self._update_btf_print()

        crs_path = pcd_path.with_suffix(".crs")
        if crs_path.exists():
            try:
                with open(crs_path, "r") as f:
                    crs_str = f.read().strip()
                self.crs = pyproj.CRS.from_user_input(crs_str)
                logger.info(
                    f"Loaded CRS from sidecar file [{crs_path}]"
                )
            except Exception as e:
                logger.warning(
                    f"Found CRS file [{crs_path}] but failed to "
                    f"load: {e}"
                )

    def write_point_cloud(self, pcd_path):
        """Write the current point cloud with legacy suffix behavior.

        .. deprecated:: 2.1.0
            Prefer :meth:`easyidp.PointCloud.save` or
            :func:`easyidp.pointcloud.write_point_cloud`.

        Parameters
        ----------
        pcd_path : str or pathlib.Path
            Output path. When no suffix is given, the current ``file_ext``
            is appended for legacy compatibility.

        Returns
        -------
        None

        Examples
        --------
        >>> pcd.write_point_cloud("cloud.ply")
        >>> pcd.save("cloud.ply")
        >>> idp.pointcloud.write_point_cloud("cloud.ply", pcd)
        """
        pcd_path = Path(pcd_path)
        file_ext = pcd_path.suffix

        if file_ext == "":
            logger.warning(
                f"It seems file name [{pcd_path}] has no file "
                f"suffix, using default suffix [{self.file_ext}] "
                f"instead"
            )
            out_path = pcd_path.with_name(
                f"{pcd_path.name}{self.file_ext}"
            )
        else:
            if file_ext not in [".ply", ".las", ".laz"]:
                raise IOError(
                    "Only support point cloud file format ['*.ply', "
                    "'*.laz', '*.las']"
                )
            out_path = pcd_path

        abs_pts = (
            self._points + self._offset
            if self._points is not None
            else np.zeros((0, 3))
        )

        if out_path.suffix == ".ply":
            ply_io.write(
                out_path,
                abs_pts,
                self.colors,
                normals=self.normals,
            )
        else:
            las_io.write(
                out_path,
                abs_pts,
                self.colors,
                normals=self.normals,
                offset=self._offset,
            )
