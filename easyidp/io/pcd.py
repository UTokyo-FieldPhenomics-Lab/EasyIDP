import pylas
import numpy as np
from easyidp.external import plyfile

def read_ply(ply_path):
    colors = np.zeros(3)
    points = np.zeros(3)

    return points, colors


def read_laz(laz_path):
    las = pylas.read(laz_path)

    colors = np.vstack([las.points['red'], las.points['green'], las.points['blue']]).T / 65536
    points = np.vstack([las.x, las.y, las.z]).T

    return points, colors