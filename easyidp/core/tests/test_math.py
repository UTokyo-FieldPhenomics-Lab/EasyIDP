import numpy as np
from easyidp.core.math import apply_transform


def test_apply_transform():
    matrix = np.asarray([[-0.86573098, -0.01489186, 0.08977677, 7.65034123],
                         [0.06972335, 0.44334391, 0.74589315, 1.85910928],
                         [-0.05848325, 0.74899678, -0.43972184, -0.1835615],
                         [0., 0., 0., 1.]], dtype=np.float)
    matrix2 = np.linalg.inv(matrix)

    point1 = np.asarray([0.5, 1, 1.5], dtype=np.float)
    out1 = apply_transform(matrix2, point1)
    ans1 = np.asarray([7.96006409,  1.30195288, -2.66971818])

    np.testing.assert_array_almost_equal(out1, ans1, decimal=6)

    points = np.asarray([[0.5, 1, 1.5],
                         [0.5, 1, 1.5]], dtype=np.float)
    out2 = apply_transform(matrix2, points)
    ans2 = np.asarray([[7.96006409, 1.30195288, -2.66971818],
                       [7.96006409, 1.30195288, -2.66971818]])
    np.testing.assert_array_almost_equal(out2, ans2, decimal=6)