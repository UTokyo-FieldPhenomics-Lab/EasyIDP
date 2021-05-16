import pyproj
import numpy as np
import pandas as pd
from easyidp.core import Points
from easyidp.core.math import apply_transform_matrix, apply_transform_crs


def test_apply_transform_matrix():
    matrix = np.asarray([[-0.86573098, -0.01489186, 0.08977677, 7.65034123],
                         [0.06972335, 0.44334391, 0.74589315, 1.85910928],
                         [-0.05848325, 0.74899678, -0.43972184, -0.1835615],
                         [0., 0., 0., 1.]], dtype=np.float)
    matrix2 = np.linalg.inv(matrix)

    point1 = Points([0.5, 1, 1.5], dtype=np.float)
    out1 = apply_transform_matrix(matrix2, point1)
    assert type(out1) == pd.DataFrame

    ans1 = Points([7.96006409,  1.30195288, -2.66971818])
    np.testing.assert_array_almost_equal(out1.values, ans1, decimal=6)

    points = Points([[0.5, 1, 1.5],
                     [0.5, 1, 1.5]], dtype=np.float)
    out2 = apply_transform_matrix(matrix2, points)
    ans2 = Points([[7.96006409, 1.30195288, -2.66971818],
                   [7.96006409, 1.30195288, -2.66971818]])
    np.testing.assert_array_almost_equal(out2.values, ans2, decimal=6)


def test_apply_transform_crs():
    # transform between geocentric and geodetic
    geocentric = Points([-3943658.7087006606, 3363404.124223561, 3704651.3067566575])
    geodetic = Points([139.54033578028609, 35.73756358928734, 96.87827569602781], columns=['lon', 'lat', 'alt'])

    geo_d = pyproj.CRS.from_epsg(4326)
    geo_c = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})

    out_c = apply_transform_crs(geo_d, geo_c, geodetic)
    # x: array([[-3943658.715, 3363404.132, 3704651.343]])
    # y: array([[-3943658.709, 3363404.124, 3704651.307]])
    np.testing.assert_array_almost_equal(out_c.values, geocentric.values, decimal=1)

    out_d = apply_transform_crs(geo_c, geo_d, geocentric)
    # x: array([[139.540336,  35.737563,  96.849   ]])
    # y: array([[139.540336,  35.737564,  96.878276]])
    np.testing.assert_array_almost_equal(out_d.values[:, 0:2], geodetic.values[:, 0:2], decimal=5)
    np.testing.assert_array_almost_equal(out_d.values[:, 2], geodetic.values[:, 2], decimal=1)