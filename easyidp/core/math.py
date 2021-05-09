import numpy as np


def apply_transform(matrix, points):
    """
    Transforms a point in homogeneous coordinates.
    equal to Metashape.Matrix.mulp() or Metashape.Matrix.mulv()

    Parameters
    ----------
    matrix: np.ndarray
        4x4 transform numpy array
    points: np.ndarray
        1x3 single point: np.asarray([1,2,3])
        or
        nx3 points: np.asarray([[1,2,3],
                                [4,5,6],
                                ...    ])

    Returns
    -------

    """
    if len(points.shape) == 1:
        # the 1x3 single point
        point_ext = np.insert(points, 3, 1)
        out = point_ext.dot(matrix.T)
        return out[0:3] / out[3]

    elif len(points.shape) == 2:
        # the nx3 multiple points
        point_ext = np.insert(points, 3, 1, axis=1)
        out = point_ext.dot(matrix.T)

        return out[:, 0:3] / out[:, 3][:, np.newaxis]
    else:
        raise ValueError("only 1x3 single point or nx3 multiple points are accepted")