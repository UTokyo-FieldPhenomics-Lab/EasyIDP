import pylas
import numpy as np
from plyfile import PlyData

def read_ply(ply_path):
    cloud_data = PlyData.read(ply_path).elements[0].data
    ply_names = cloud_data.dtype.names

    points = np.vstack((cloud_data['x'], cloud_data['y'], cloud_data['z'])).T

    if 'red' in ply_names:
        colors = np.vstack((cloud_data['red'] / 255, cloud_data['green'] / 255, cloud_data['blue'] / 255)).T
    elif 'diffuse_red' in ply_names:
        colors = np.vstack((cloud_data['diffuse_red'] / 255, cloud_data['diffuse_green'] / 255,
                            cloud_data['diffuse_blue'] / 255)).T
    else:
        print(f"Can not find color info in {ply_names}")
        colors = None

    return points, colors


def read_laz(laz_path):
    las = pylas.read(laz_path)

    colors = np.vstack([las.points['red'], las.points['green'], las.points['blue']]).T / 65536
    points = np.vstack([las.x, las.y, las.z]).T

    return points, colors