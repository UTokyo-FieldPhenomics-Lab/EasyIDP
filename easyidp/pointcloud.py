import os
import warnings
from datetime import datetime

import numpy as np
import numpy.lib.recfunctions as rfn

import laspy
from plyfile import PlyData, PlyElement

import easyidp as idp

class PointCloud(object):

    def __init__(self, pcd_path="", origin_offset=np.array([0.,0.,0.])) -> None:

        self.file_path = pcd_path
        self.file_ext = ".ply"

        self.points = None
        self.colors = None
        self.normals = None
        self.offset = origin_offset

        if len(pcd_path) > 0 and os.path.exists(pcd_path):
            self.read_point_cloud(pcd_path)

    def has_colors(self):
        if self.colors is None:
            return False
        else:
            return True

    def has_points(self):
        if self.points is None:
            return False
        else:
            return True

    def has_normals(self):
        if self.normals is None:
            return False
        else:
            return True

    def read_point_cloud(self, pcd_path):
        if pcd_path[-4:] == ".ply":
            points, colors = read_ply(pcd_path)
        elif pcd_path[-4:] == ".laz" or pcd_path[-4:] == ".las":
            points, colors = read_laz(pcd_path)
        else:
            raise IOError("Only support point cloud file format ['*.ply', '*.laz', '*.las']")

        self.file_ext = os.path.splitext(pcd_path)[-1]

        if abs(np.max(points)) > 65536:   # need offseting
            if not np.any(self.offset):    # not given any offset (0,0,0) -> calculate offset
                self.offset = np.floor(points.min(axis=0) / 100) * 100
            self.points = points - self.offset
        else:
            self.points = points

        self.colors = colors

    def write_point_cloud(self, pcd_path):
        # if ply -> self.points + self.offsets
        # if las -> self.points & offset = self.offsets
        split_ext = os.path.splitext(pcd_path)

        if len(split_ext) > 1:  # means has "*.ext" as suffix
            file_name = split_ext[0]
            file_ext = split_ext[-1]
            if file_ext not in ['.ply', '.las', '.laz']:
                raise IOError("Only support point cloud file format ['*.ply', '*.laz', '*.las']")
        else:
            warnings.warn(f"It seems file name [{pcd_path}] has no file suffix, using default suffix [{self.file_ext}] instead")
            file_name = pcd_path
            file_ext = self.file_ext

        if file_ext == ".ply":
            write_ply(self.points + self.offset, self.colors, file_name+file_ext)
        else:
            write_laz(self.points, self.colors, file_name+file_ext, self.offset)


    def crop_point_cloud(self, poly_boundary, save_as=""):
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

    colors.dtype = np.uint8

    return points, colors


def read_laz(laz_path):
    las = laspy.read(laz_path)

    # ranges 0-65536
    colors = np.vstack([las.points['red'], las.points['green'], las.points['blue']]).T / 256
    points = np.vstack([las.x, las.y, las.z]).T

    colors.dtype = np.uint8

    return points, colors

def write_ply(points, colors, ply_path, binary=True):
    """
    need to convert to structured arrays then save
       https://github.com/dranjan/python-plyfile#creating-a-ply-file
       the point cloud structure looks like this:
       >>> cloud_data.elements
       (PlyElement('vertex', 
           (PlyProperty('x', 'float'), 
            PlyProperty('y', 'float'), 
            PlyProperty('z', 'float'), 
            PlyProperty('red', 'uchar'), 
            PlyProperty('green', 'uchar'), 
            PlyProperty('blue', 'uchar')), count=42454, comments=[]),)
    
    and numpy way to convert ndarray to strucutred array:
       https://stackoverflow.com/questions/3622850/converting-a-2d-numpy-array-to-a-structured-array
    
    and method to merge to structured arrays
       https://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays
    """

    # convert to strucutrre array
    struct_points = np.core.records.fromarrays(points.T, names="x, y, z")
    struct_colors = np.core.records.fromarrays(colors.T, names="red, green, blue")

    # merge 
    struct_merge = rfn.merge_arrays([struct_points, struct_colors], flatten=True, usemask=False)

    # convert to PlyFile data type
    el = PlyElement.describe(struct_merge, 'vertex', 
                             comments=[f'Created by EasyIDP v{idp.__version__}', 
                                       f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}'])  

    # save to file
    if binary:
        PlyData([el]).write(ply_path)
    else:
        PlyData([el], text=True).write(ply_path)

def write_laz(points, colors, laz_path, offset=np.array([0., 0., 0.])):
    las = laspy.create()
    las.xyz = points
    las.points['red'] = colors[:,0]
    las.points['green'] = colors[:,1]
    las.points['blue'] = colors[:,2]
    las.header.offsets = offset

    las.write(laz_path)

def write_las(points, colors, las_path, offset=np.array([0., 0., 0.])):
    write_laz(points, colors, las_path, offset)