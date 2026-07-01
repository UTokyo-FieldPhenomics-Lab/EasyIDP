"""Internal KDTree-based point cloud geometry helpers.

These helpers are importable but not part of the default public API.
"""

import numpy as np
from scipy.spatial import cKDTree
from matplotlib.path import Path as mplPath


def query_indices_by_polygon(points_xy, polygon_xy, tree=None):
    """Find indices of points inside a 2D polygon using KDTree pre-filtering.

    Uses KDTree bounding-box query (Chebyshev, p=inf) to find candidate
    points, then applies exact polygon containment via matplotlib Path.

    Parameters
    ----------
    points_xy : ndarray of shape (N, 2)
        Absolute XY coordinates of all points.
    polygon_xy : ndarray of shape (M, 2)
        Polygon boundary vertices in XY plane.
    tree : cKDTree or None, optional
        Pre-built KDTree over ``points_xy``. If None, builds one.

    Returns
    -------
    ndarray of int
        Indices of points inside the polygon.

    Examples
    --------
    >>> points_xy = np.array([[0, 0], [1, 1], [3, 3]])
    >>> polygon_xy = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    >>> query_indices_by_polygon(points_xy, polygon_xy)
    array([1])
    """
    if tree is None:
        tree = cKDTree(points_xy)

    poly_pts = np.asarray(polygon_xy)
    poly_2d = poly_pts[:, 0:2]

    xmin, ymin = poly_2d.min(axis=0)
    xmax, ymax = poly_2d.max(axis=0)

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    r = max((xmax - xmin) / 2, (ymax - ymin) / 2)

    candidate_idx = tree.query_ball_point([cx, cy], r, p=np.inf)
    if len(candidate_idx) == 0:
        return np.array([], dtype=int)

    candidate_pts = points_xy[candidate_idx]
    mpl_poly = mplPath(poly_2d)
    mask = mpl_poly.contains_points(candidate_pts)

    return np.array(candidate_idx)[mask]


def query_indices_by_multipolygon(points_xy, multipolygon, tree=None):
    """Query indices for all sub-polygons in a MultiPolygon.

    Parameters
    ----------
    points_xy : ndarray of shape (N, 2)
        Absolute XY coordinates of all points.
    multipolygon : shapely.geometry.MultiPolygon
        The multipolygon to query against.
    tree : cKDTree or None, optional
        Pre-built KDTree over ``points_xy``.

    Returns
    -------
    ndarray of int
        Union of indices inside any sub-polygon, without duplicates.

    Examples
    --------
    >>> from shapely.geometry import MultiPolygon, Polygon
    >>> points_xy = np.array([[0, 0], [1, 1], [5, 5]])
    >>> geom = MultiPolygon([
    ...     Polygon([[0, 0], [2, 0], [2, 2], [0, 2]]),
    ...     Polygon([[4, 4], [6, 4], [6, 6], [4, 6]]),
    ... ])
    >>> query_indices_by_multipolygon(points_xy, geom)
    array([1, 2])
    """
    all_indices = []
    for geom in multipolygon.geoms:
        coords = np.array(geom.exterior.coords)
        if coords.ndim == 1:
            continue
        idx = query_indices_by_polygon(points_xy, coords[:, 0:2], tree=tree)
        if len(idx) > 0:
            all_indices.append(idx)

    if not all_indices:
        return np.array([], dtype=int)

    return np.unique(np.concatenate(all_indices))
