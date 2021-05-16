import pandas as pd
import pyproj
import numpy as np


def apply_transform_matrix(matrix, points_df):
    """
    Transforms a point or points in homogeneous coordinates.
    equal to Metashape.Matrix.mulp() or Metashape.Matrix.mulv()

    Parameters
    ----------
    matrix: np.ndarray
        4x4 transform numpy array
    points_df: pd.DataFrame
        # 1x3 single point
        >>> pd.DataFrame([np.asarray([1,2,3])], columns=['x', 'y', 'z'])
           x  y  z
        0  1  2  3

        # nx3 points
        >>> pd.DataFrame(data, columns=['x', 'y', 'z'])
           x  y  z
        0  1  2  3
        1  4  5  6

    Returns
    -------
    out: pd.DataFrame
        same size as input points_np
           x  y  z
        0  1  2  3
        1  4  5  6
    """
    point_ext = np.insert(points_df.values, 3, 1, axis=1)
    dot_matrix = point_ext.dot(matrix.T)
    dot_points = dot_matrix[:, 0:3] / dot_matrix[:, 3][:, np.newaxis]

    return pd.DataFrame(dot_points, columns=points_df.columns)


def apply_transform_crs(crs_origin, crs_target, points_df):
    """
    Transform a point or points from one CRS to another CRS, by pyproj.CRS.Transformer function

    Attention:
    pyproj.CRS order: (lat, lon, alt)
    points order in this package are commonly (lon, lat, alt)

    Parameters
    ----------
    crs_origin: pyproj.CRS object

    crs_target: pyproj.CRS object

    points_df: pd.DataFrame
           x  y  z
        0  1  2  3

        or

           lon  lat  alt
        0    1    2    3
        1    4    5    6

    Returns
    -------

    """
    ts = pyproj.Transformer.from_crs(crs_origin, crs_target)

    # xyz order common points
    if 'x' in points_df.columns.values:
        if crs_target.is_geocentric:
            x, y, z = ts.transform(points_df.x.values, points_df.y.values, points_df.z.values)
            return pd.DataFrame({'x':x, 'y':y, 'z':z})

        if crs_target.is_geographic:
            lat, lon, alt = ts.transform(points_df.x.values, points_df.y.values, points_df.z.values)
            return pd.DataFrame({'lon': lon, 'lat': lat, 'alt': alt})


    if 'lat' in points_df.columns.values:
        if crs_target.is_geocentric:
            x, y, z = ts.transform(points_df.lat.values, points_df.lon.values, points_df.alt.values)
            return pd.DataFrame({'x': x, 'y': y, 'z': z})

        if crs_target.is_geographic:
            lat, lon, alt = ts.transform(points_df.lat.values, points_df.lon.values, points_df.alt.values)
            return pd.DataFrame({'lon': lon, 'lat': lat, 'alt': alt})