import os
import pyproj
import warnings
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from pathlib import Path

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
        if isinstance(obj, (Path, str)):
            if not Path(obj).exists():
                raise FileNotFoundError(f"Could not find file {obj}.")

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

        .. image:: ../../_static/images/python_api/roi_crop_mode.png
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
            global_z = dsm.math_polygon(polygon_hv="full_map", kernel=kernel)
        else:
            global_z = None

        # convert CRS if necessary
        if self.crs.name == dsm.header["crs"].name:
            poly_dict = self.id_item.copy()
        elif self.crs.name != dsm.header["crs"].name and not keep_crs:
            self.change_crs(dsm.header["crs"])
            poly_dict = self.id_item.copy()
        else:
            poly_dict = idp.shp.convert_proj(self.id_item, self.crs, dsm.header["crs"])

        pbar = tqdm(poly_dict.items(), desc=f"Read z values of roi from DSM [{dsm.file_path.name}]")
        for key, poly in pbar:
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

    def crop(self, target, save_folder=None):
        """Crop several ROIs from the geotiff by given <ROI> object with several polygons and polygon names

        Parameters
        ----------
        target : str | <GeoTiff> object
            the path of dsm, or the GeoTiff object from idp.GeoTiff()
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_folder : str, optional
            the folder to save cropped images, use ROI indices as file_names, by default None, means not save.

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
            
        if isinstance(target, (Path, str)) and Path(target).exists():
            ext = Path(target).suffix
            if ext == ".tif":
                target = idp.GeoTiff(target)
            elif ext in [".ply", ".laz", ".las"]:
                target = idp.PointCloud(target)
            else:
                raise TypeError(f"Only [.tif, .ply, .laz, .las] are supported, not [{ext}]")
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

    def back2raw(self, recons, save_folder=None, **kwargs):
        """Projects several GIS coordintates ROIs (polygons) to all images

        Parameters
        ----------
        roi : easyidp.ROI | dict
            the <ROI> object created by easyidp.ROI() or dictionary
        save_folder : str, optional
            the folder to save projected preview images and json files, by default ""
        distortion_correct : bool, optional
            Whether do distortion correction, by default True (back to raw image with lens distortion);
            If back to software corrected images without len distortion, set it to False. 
            (Pix4D support do this operation, seems metashape not supported yet.)
        ignore : str | None, optional
            None: strickly in image area;
            'x': only y (vertical) in image area, x can outside image;
            'y': only x (horizontal) in image area, y can outside image.
        log : bool, optional
            whether print log for debugging, by default False

        Returns
        -------
        out_dict : dict
            The 2-layer dictionary, the first layer is the roi id, the second layer is the image names contains each roi ( ``out_dict['roi_id']['image_name']`` )

            .. code-block:: python

                >>> out_dict = roi.back2raw(...)
                # find all available images with specified roi (plot), e.g. roi named 'N1W1':
                >>> one_roi_dict = out_dict['N1W1']  # roi id
                {'IMG_3457': ..., 'IMG_3458': ..., ...}
                # find the sepecific roi on specific image, e.g. roi named 'N1W1' on image 'IMG_3457':
                >>> one_roi_one_img_coord = out_dict["N1W1']['IMG_3457']
                array([[  43.9388228 , 1247.0474214 ],
                       [  69.04076173,  972.90860296],
                       [ 353.26968458,  993.31308291],
                       [ 328.12327606, 1267.41006845],
                       [  43.9388228 , 1247.0474214 ]])

        See also
        --------
        easyidp.pix4d.back2raw, easyidp.metashape.back2raw
        """
        # call related function
        # need check alt exists, the alt is changing for different dsm?
        if not self.is_geo():
            raise TypeError("Could not operate without CRS specified")

        # if is one chunk
        if isinstance(recons, (idp.Pix4D, idp.Metashape)):
            out_dict = recons.back2raw(self, **kwargs)
        # several chunks
        if isinstance(recons, idp.ProjectPool):
            out_dict = {}
            for chunk in recons:
                if isinstance(save_folder, str) and os.path.isdir(save_folder):
                    save_path = os.path.join(save_folder, chunk.label)
                else:
                    save_path = None

                out_dict[chunk.label] = chunk.back2raw(self, save_folder=save_path, **kwargs)

        return out_dict


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


# Load detections for backwards projection 

def load_detections(path):
    """Load a csv file of bounding box detections
    CSV takes the format xmin, ymin, xmax, ymax, image_path, label. The bounding box corners are in the image coordinate system,
    the image_path is the expected to be the full path, and the label is the character label for each box. One detection per row in the csv.
    
    Args:
        path: path on local disk
    Returns:
        boxes: a pandas dataframe of detections
    """
    
    # boxes = pd.read_csv(path)
    if not all([x in ["image_path","xmin","ymin","xmax","ymax","image_path","label"] for x in boxes.columns]):
        raise IOError("{} is expected to be a .csv with columns, xmin, ymin, xmax, ymax, image_path, label for each detection")
        
    return None