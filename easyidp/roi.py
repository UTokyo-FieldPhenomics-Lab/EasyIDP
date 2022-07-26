import os
import pyproj
import warnings
import numpy as np
from copy import copy as ccopy
from shapely.geometry import Point, Polygon

import easyidp as idp


class ROI(idp.Container):
    """
    Summary APIs of each objects, often read from shp file.
    """

    def __init__(self, target_path=None, **kwargs):
        super().__init__()
        # in super
        # self.id_item = {}
        # self.item_label = {}
        # if has CRS -> GPS coordiantes -> geo2pix convert
        self.crs = None   # default -> pixel coords
        self.source = target_path

        if target_path is not None:
            self.open(target_path, **kwargs)

    def __setitem__(self, key, item):
        idx = len(self.id_item)
        self.id_item[idx] = item
        self.item_label[key] = idx

    def is_geo(self):
        """Returns True if the ROI is geo coordinate.

        Returns
        -------
        bool
        """
        if self.crs is None:
            return False
        else:
            return True


    def open(self, target_path, **kwargs):
        """An advanced wrapper to open ROI without dealing with format

        Parameters
        ----------
        target_path : str
            the path to roi files, current support \*.shp and labelme.json

        Notes
        -----
        You can also pass several control parameters in this function, please refer see also for more information

        See also
        --------
        read_shp, read_labelme_json

        """
        ext = os.path.splitext(target_path)[-1]
        if ext == ".shp":
            self.read_shp(target_path, **kwargs)
        elif ext == ".json":
            self.read_labelme_json(target_path)


    def read_shp(self, shp_path, shp_proj=None, name_field=None, include_title=False, encoding='utf-8'):
        """read ROI from shp file

        Parameters
        ----------
        shp_path : str
            the file path of \*.shp
        shp_proj : str | pyproj object
            by default None, will read automatically from prj file with the same name of shp filename, 
            or give manually by ``read_shp(..., shp_proj=pyproj.CRS.from_epsg(4326), ...)`` or 
            ``read_shp(..., shp_proj=r'path/to/{shp_name}.prj', ...)`` 
        name_field : str or int or list[ str|int ], optional
            by default None, the id or name of shp file fields as output dictionary keys
        include_title : bool, optional
            by default False, whether add column name to roi key.
        encoding : str
            by default 'utf-8', for some chinese characters, 'gbk' may required

        Notes
        -----
        For details of this parameters, please refer to see also.

        See also
        --------
        easyidp.shp.read_shp
        """
        # if geotiff_proj is not None and shp_proj is not None and shp_proj.name != geotiff_proj.name:
        # shp.convert_proj()
        roi_dict, crs = idp.shp.read_shp(shp_path, shp_proj, name_field, include_title, encoding, return_proj=True)

        self.source = shp_path

        self.crs = crs
        self.id_item = {}
        self.item_label = {}

        for k, v in roi_dict.items():
            self[k] = v

    def read_labelme_json(self, json_path):
        js_dict = idp.jsonfile.read_json(json_path)

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

        self.id_item = idp.shp.convert_proj(self.id_item, self.crs, target_crs)
        self.crs = target_crs

    def _get_z_input_check(self, obj, mode, kernel, buffer, func="dsm"):
        # check if has CRS (GeoROI), otherwise stop
        if not self.is_geo():
            raise TypeError("Could not operate without CRS specified")

        # check input type
        if mode not in ["point", "face"]:
            raise KeyError(
                f"The param 'mode' only accept 'point' or 'face', not '{mode}'"
            )

        if kernel not in [
            "mean", "min", "max", "pmin5", "pmin10", "pmax5", "pmax10"
        ]:
            raise KeyError(f"The param 'kernal' only accept "
                f"'mean', 'min', 'max', 'pmin5', 'pmin10', 'pmax5', 'pmax10'"
                f" not '{kernel}'"
            )

        if not isinstance(buffer, (int, float)):
            raise TypeError(
                f"Only 'int' and 'float' are acceptable for 'buffer', not "
                f"{type(buffer)} [{buffer}]."
            )

        # convert input objects
        if isinstance(obj, str) and os.path.exists(obj):
            if func == "dsm":
                return idp.GeoTiff(obj)
            else:
                return idp.PointCloud(obj)
        elif isinstance(obj, (idp.GeoTiff, idp.PointCloud)):
            return obj
        else:
            if func == "dsm":
                raise TypeError(
                    f"Only geotiff path <str> and <easyidp.GeoTiff> object"
                    f"are accepted, not {type(obj)}")
            else:
                raise TypeError(
                    f"Only geotiff path <str> and <easyidp.PointCloud> object"
                    f"are accepted, not {type(obj)}")


    def get_z_from_dsm(self, dsm, mode="face", kernel="mean", buffer=0, keep_crs=False):
        """get the z values (heights) from DSM for 2D polygon

        Parameters
        ----------
        dsm : str | <GeoTiff> object
            the path of dsm, or the GeoTiff object from idp.GeoTiff()
        mode : str, optional
            the mode to calculate z values, option in "point" and "face"
            **point**: get height on each vertex, result in different values for each vertex
            **face**: get height on polygon face, result in the same value for each vertex
        kernal : str, optional
            The math kernal to calculate the z value.
            ["mean", "min", "max", "pmin5", "pmin10", "pmax5", "pmax10"], by default 'mean'
        buffer : float, optional
            the buffer of ROI, by default 0 (no buffer),
            can be positive values or -1 (using all map), 
            please check the Notes section for more details
        keep_crs : bool, optional
            When the crs is not the save with DSM crs, where change the ROI crs to fit DSM.
            **False** (default): change ROI's CRS;
            **True**: not change ROI's CRS, only attach the z value to current coordinate. 

        Notes
        -----

        **Option details for** ``kernal`` **parameter**

        - "mean": the mean value inside polygon
        - "min": the minimum value inside polygon
        - "max": the maximum value inside polygon
        - "pmin5": 5th *percentile mean* inside polygon
        - "pmin10": 10th *percentile mean* inside polygon
        - "pmax5": 95th *percentile mean* inside polygon
        - "pmax10": 90th *percentile mean* inside polygon

        percentile mean: the mean value of all pixels over/under xth percentile threshold

        **Option details for** ``buffer`` **parameter**

        - 0: not using buffer
        - -1: ignore given polygon, using the full dsm to calculate the height
        - float: buffer distance, the unit of buffer follows the ROI coordinates, either pixel or meter.

        If mode is "point", will generate a round buffer polygon first, then extract the z-value by this region, but the return will only be a single point.

        If mode is "face", will buffer the polygon and then calculate the height inside the buffered polygon

        .. image:: ../_static/images/python_api/roi_crop_mode.png
            :alt: roi_crop_mode.png


        Examples
        --------

        Combine 

        See also
        --------
        easyidp.GeoTiff.math_polygon
        """
        dsm = self._get_z_input_check(dsm, mode, kernel, buffer, func="dsm")

        # check is the dsm, not RGB or multiband GeoTiff.
        if dsm.header["dim"] != 1:
            raise TypeError(
                f"Only one layer geotiff (DSM) are accepted, current "
                f"layer is {dsm.header['dim']}")

        # using the full map to calculate
        if buffer == -1 or buffer == -1.0:
            global_z = dsm.math_polygon(polygon_hv="all", kernel=kernel)
        else:
            global_z = None

        # convert CRS is necessary
        if self.crs.name != dsm.header["crs"].name and not keep_crs:
            self.change_crs(dsm.header["crs"])
            poly_dict = self.id_item.copy()
        else:
            poly_dict = idp.shp.convert_proj(self.id_item, self.crs, dsm.header["crs"])

        for key, poly in poly_dict.items():
            # only get the x and y of coords
            poly = poly[:, 0:2]

            # using the full map
            if global_z is not None:
                poly3d = np.insert(self.id_item[key], obj=2, values=global_z, axis=1)
            else:
                if mode == "face":    # using the polygon as uniform z values
                    # need do buffer
                    if buffer != 0 or buffer != 0.0:
                        p = Polygon(poly)
                        p_buffer = p.buffer(buffer)
                        poly = np.array(p_buffer.exterior.coords)

                    poly_z = dsm.math_polygon(poly, is_geo=True, kernel=kernel)
                    
                    poly3d = np.insert(self.id_item[key], obj=2, values=poly_z, axis=1)

                else:    # using each point own z values
                    if buffer != 0 or buffer != 0.0:
                        z_values = []
                        for po in poly:
                            p = Point(po)
                            p_buffer = p.buffer(buffer)
                            p_buffer_np = np.array(p_buffer.exterior.coords)

                            poly_z = dsm.math_polygon(p_buffer_np, is_geo=True, kernel=kernel)
                            z_values.append(poly_z)

                        z_values = np.array(z_values)
                    else:
                        # just qurey pixel value 
                        z_values = dsm.point_query(poly, is_geo=True)

                    poly3d = np.concatenate([self.id_item[key], z_values[:, None]], axis=1)

            self.id_item[key] = poly3d


    def get_z_from_pcd(self, pcd, mode="face", kernel="mean", buffer=0):
        # if mode = point, buffer > 0, otherwise raise error
        pcd = self._get_z_input_check(pcd, mode, kernel, buffer, func="pcd")
        raise NotImplementedError("Will be implemented in the future.")

    def crop(self, target, save_folder=""):
        """Crop several ROIs from the geotiff by given <ROI> object with several polygons and polygon names

        Parameters
        ----------
        target : str | <GeoTiff> object
            the path of dsm, or the GeoTiff object from idp.GeoTiff()
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_folder : str, optional
            the folder to save cropped images, use ROI indices as file_names, by default "", means not save.

        Returns
        -------
        dict,
            The dictionary with key=id and value=ndarray data

        See also
        --------
        easyidp.GeoTiff.crop
        """
        if not self.is_geo():
            raise TypeError("Could not operate without CRS specified")
            
        if isinstance(target, str) and os.path.exists(target):
            ext = os.path.splitext(target)[-1]
            if ext == ".tif":
                target = idp.GeoTiff(target)
            elif ext in [".ply", ".laz", ".las"]:
                target = idp.PointCloud(target)
        elif isinstance(target, (idp.GeoTiff, idp.PointCloud)):
            pass
        else:
            raise TypeError(
                f"Only file path <str> or <easyidp.GeoTiff> object or <easyidp.PointCloud> object "
                f"are accepted, not {type(target)}"
            )

        if isinstance(target, idp.GeoTiff):
            out = target.crop(self, is_geo=True, save_folder=save_folder)
        elif isinstance(target, idp.PointCloud):
            out = target.crop(self, save_folder=save_folder)

        return out

    def back2raw(self, chunks):
        # call related function
        # need check alt exists, the alt is changing for different dsm?
        pass

    def copy(self):
        """make a deep copy of current file

        Returns
        -------
        easyidp.ROI
        """
        ctn = ROI()
        ctn.id_item = self.id_item.copy()
        ctn.item_label = self.item_label.copy()
        ctn.crs = ccopy(self.crs)
        ctn.source = ccopy(self.source)

        return ctn


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