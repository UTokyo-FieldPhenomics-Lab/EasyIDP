import numpy as np

import laspy
from plyfile import PlyData

points_dtype = np.dtype([('x', 'float64'), ('y', 'float64'), ('z', 'float64')])
colors_dtype = np.dtype([('r', 'uint8'), ('g', 'uint8'), ('b', 'uint8')])

class PointCloud(object):

    def __init__(self, pcd_path="", origin_offset=np.zeros(3)) -> None:

        self.file_path = pcd_path

        self.points = np.empty((0,3), dtype=points_dtype)
        self.colors = np.empty((0,3), dtype=colors_dtype)
        self.offset = origin_offset

    def read_pcd(self, pcd_path):
        if pcd_path[-4:] == ".ply":
            points, colors = read_ply(pcd_path)
        elif pcd_path[-4:] == ".laz" or pcd_path[-4:] == ".las":
            points, colors = read_laz(pcd_path)
        else:
            raise IOError("Only support point cloud file ['*.ply', '*.laz', '*.las']")

        if abs(np.max(points)) > 65536:   # need offseting
            if not np.any(self.offset):    # not given any offset (0,0,0) -> calculate offset
                self.offset = np.floor(points.min(axis=0) / 100) * 100
            self.points = points - self.offset
        else:
            self.points = points

        self.colors = colors

    def crop_pcd_xy(self, poly_boundary, save_as=""):
        # need write 1. crop by roi 2. save to ply & las files
        pass


def read_ply(ply_path):
    cloud_data = PlyData.read(ply_path).elements[0].data
    ply_names = cloud_data.dtype.names

    points = np.vstack((cloud_data['x'], cloud_data['y'], cloud_data['z'])).T

    if 'red' in ply_names:
        # range in 0-255
        colors = np.vstack((cloud_data['red'], cloud_data['green'], cloud_data['blue'])).T
    elif 'diffuse_red' in ply_names:
        colors = np.vstack((cloud_data['diffuse_red'], cloud_data['diffuse_green'], cloud_data['diffuse_blue'])).T
    else:
        print(f"Can not find color info in {ply_names}")
        colors = None

    points.dtype = points_dtype
    colors.dtype = colors_dtype

    return points, colors


def read_laz(laz_path):
    las = laspy.read(laz_path)

    # ranges 0-65536
    colors = np.vstack([las.points['red'], las.points['green'], las.points['blue']]).T / 256
    points = np.vstack([las.x, las.y, las.z]).T

    points.dtype = points_dtype
    colors.dtype = colors_dtype

    return points, colors