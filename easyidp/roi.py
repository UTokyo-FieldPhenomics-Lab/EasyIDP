import os
import pyproj
import warnings
import numpy as np
from .shp import read_proj, read_shp, convert_proj, show_shp_fields
from .reconstruct import Container
from .jsonfile import read_json

class ROI(Container):
    """
    Summary APIs of each objects, often read from shp file.
    """

    def __init__(self, target_path=None):
        super().__init__()
        # in super
        # self.id_item = {}
        # self.item_label = {}
        # if has CRS -> GPS coordiantes -> geo2pix convert
        self.crs = None   # default -> pixel coords
        self.source = target_path

        if target_path is not None:
            self.open(target_path)
            

    def __setitem__(self, key, item):
        idx = len(self.id_item)
        self.id_item[idx] = item
        self.item_label[key] = idx


    def open(self, target_path):
        ext = os.path.splitext(target_path)[-1]
        if ext == ".shp":
            self.read_shp(target_path)
        elif ext == ".json":
            self.read_labelme_json(target_path)


    def read_shp(self, shp_path, shp_proj=None, name_field=None, include_title=False, encoding='utf-8'):
        # if geotiff_proj is not None and shp_proj is not None and shp_proj.name != geotiff_proj.name:
        # shp.convert_proj()
        roi_dict, crs = read_shp(shp_path, shp_proj, name_field, include_title, encoding, return_proj=True)

        self.source = shp_path

        self.crs = crs
        self.id_item = {}
        self.item_label = {}

        for k, v in roi_dict.items():
            self[k] = v

    def read_labelme_json(self, json_path):
        js_dict = read_json(json_path)

        # check if is labelme json
        if all(x in js_dict.keys() for x in ["version", "flags", "shapes", "imagePath", "imageHeight"]):

            # init values
            self.crs = None
            self.id_item = {}
            self.item_label = {}

            for shapes in js_dict["shapes"]:
                if shapes["shape_type"] == "polygon":
                    label = shapes["label"]
                    poly = shapes["points"]

                    self[label] = np.array(poly)
                else:
                    warnings.warn(
                        f"Only labelme [polygon] shape are accepted, not [{shapes['shape_type']}] of [{shapes['label']}]")
        else:
            raise TypeError(f"It seems [{json_path}] is not a Labelme json file.")

    def change_crs(self, target_crs):
        if self.crs is None:
            raise FileNotFoundError(
                "Current ROI does not have CRS, can not convert "
                "(Is it a pixel coordinate?)")

        if  not isinstance(self.crs, pyproj.CRS) or \
            not isinstance(target_crs, pyproj.CRS):
            raise TypeError(
                f"Both self.crs <{type(self.crs)}> and target_crs "
                f"<{type(target_crs)}> should be <pyproj.CRS> type"
            )

        self.id_item = convert_proj(self.id_item, self.crs, target_crs)
        self.crs = target_crs
        

    def get_z_from_dsm(self, dsm_path, mode="face", kernel="mean", buffer=0):
        """get the z values (heights) from DSM for 2D polygon

        Parameters
        ----------
        dsm_path : str | <GeoTiff> object
            the path of dsm, or the GeoTiff object from idp.GeoTiff()
        mode : str, optional
            - point: get height on each vertex, result in different values for each vertex
            - face: get height on polygon face, result in the same value for each vertex
        kernal : str, optional
            THe math kernal to calculate the z value.
            ["local", "mean", "min", "max", "all"], by default 'mean'
            - "local" using the z value of where boundary points located, each point will get different z-values
                    -> this will get a 3D curved mesh of ROI
            - "mean": using the mean value of boundary points closed part.
            - "min": 5th percentile mean height (the mean value of all pixels < 5th percentile)
            - "max": 95th percentile mean height (the mean value of all pixels > 95th percentile)
            - "all": using the mean value of whole DSM as the same z-value for all boundary points
                    -> this will get a 2D plane of ROI
        buffer : int, optional
            the buffer of ROI, by default 0
            it is suitable when the given ROI is points rather than polygons. Given this paramter will generate a round buffer
                polygon first, then extract the z-value by this region, but the return will only be a single point
            The unit of buffer follows the ROI coordinates, either pixel or meter.
        """

    def crop(self, target):
        # call related function
        pass

    def back2raw(self, chunks):
        # call related function
        # need check alt exists, the alt is changing for different dsm?
        pass


def read_cc_txt(txt_path):
    """Read the point cloud annotation made by cloudcompare

    Parameters
    ----------
    txt_path : str
        The path to cloudcompare annotation txt file
    """
    if os.path.exists(txt_path):
        try:
            # try to analysis 'x,y,z' type 
            test_data = np.loadtxt(txt_path, delimiter=',')
        except ValueError as e:
            # means it is 'label, x, y, z' type
            # line i -> Point #0, x, y, z
            # ValueError: could not convert string to float: 'Point '
            # it also view # as comment sign.
            test_data = np.loadtxt(txt_path, delimiter=',', comments="@", usecols=[1,2,3])
    else:
        raise FileNotFoundError(f"Could not find file [{txt_path}]")

    # get points A,B,C,D, but polygon needs A,B,C,D,A
    poly_data = np.append(test_data, test_data[0,:][None,:], axis = 0)

    return poly_data