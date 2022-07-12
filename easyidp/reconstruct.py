import os
import pyproj
import numpy as np
from .geotiff import GeoTiff
from .pointcloud import PointCloud


class Container(dict):
    # a dict-like class, to contain items like {"id": item.label}
    # but enable using both [item.id] and [item.label] to fetch items
    # https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict

    def __init__(self):
        super().__init__()
        self.id_item = {}
        self.item_label = {}

    def __setitem__(self, key, item):
        self.id_item[key] = item
        self.item_label[item.label] = key

    def __getitem__(self, key):
        print(type(key))
        if isinstance(key, int):  # index by photo order
            return self.id_item[key]
        elif isinstance(key, str):  # index by photo name
            return self.id_item[self.item_label[key]]
        # elif isinstance(key, slice):
        #     return list(self.id_item.values())[key]
        else:
            raise KeyError(f"Key should be 'int', 'str', 'slice', not {key}")

    def __repr__(self):
        return repr(self.id_item)

    def __len__(self):
        return len(self.id_item)

    def __delitem__(self, key):
        del self.item_label[self.id_item[key]]
        del self.id_item[key]

    def __iter__(self):
        return iter(self.id_item.values())

    def keys(self):
        return self.id_item.keys()

    def values(self):
        return self.id_item.values()

    def items(self):
        return self.id_item.items()


class ProjectPool(Container):

    def __init__(self) -> None:
        super().__init__()
        self.id_item = {}
        self.item_label = {}

    def add_pix4d(self, paths):
        # proj.add_pix4d(["aaa.p4d", "bbb.p4d", ...]) 
        pass

    def add_metashape(self, paths):
        # proj.add_metashape(["aaa.psx", "bbb.psx"]) support using list to give time-series data
        pass


class Recons(object):
    """
    Equals to each individual Pix4D project & each chunk in Metashape project

    Coordinate systems used:
    internal   coordinate (local)
        the coordinate used in current chunk, often the center of model as initial point
    geocentric coordinate (world)
        use the earth's core as initial point, also called world coordinate
    geographic coordinate (crs):
        coordinate reference system (CRS) to locate geographical entities. Common used:
        WGS84 (EPSG: 4326)  | xyz = longitude, latitude, altitude
        WGS84/ UTM Zone xxx | e.g. UTM Zone 54N -> Tokyo area.
        ...
    """
    def __init__(self):

        self.label = ""
        self.meta = {}
        self.enabled = True

        self.sensors = Container()
        self.photos = Container()

        self.world_crs = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})
        self.crs = None

        self._dom = GeoTiff()
        self._dsm = GeoTiff()
        self._pcd = PointCloud()

    @property
    def dom(self):
        # default None
        if self._dom.has_data():
            return self._dom
        else:
            return None

    @dom.setter
    def dom(self, p):
        if isinstance(p, str):
            if os.path.exists(p):
                self._dom.read_geotiff(p)
            else:
                raise FileNotFoundError(f"Given DOM file [{p}] does not exists")
        elif isinstance(p, GeoTiff):
            self._dom = p
        else:
            raise TypeError(f"Please either specify DOM file path (str) or idp.GeoTiff objects, not {type(p)}")

    @property
    def dsm(self):
        if self._dsm.has_data():
            return self._dsm
        else:
            return None

    @dsm.setter
    def dsm(self, p):
        if isinstance(p, str):
            if os.path.exists(p):
                self._dsm.read_geotiff(p)
            else:
                raise FileNotFoundError(f"Given DSM file [{p}] does not exists")
        elif isinstance(p, GeoTiff):
            self._dsm = p
        else:
            raise TypeError(f"Please either specify DSM file path (str) or idp.GeoTiff objects, not {type(p)}")

    @property
    def pcd(self):
        if self._pcd.has_points():
            return self._pcd
        else:
            return None

    @pcd.setter
    def pcd(self, p):
        if isinstance(p, str):
            if os.path.exists(p):
                self._pcd.read_point_cloud(p)
            else:
                raise FileNotFoundError(f"Given pointcloud file [{p}] does not exists")
        elif isinstance(p, PointCloud):
            self._pcd = p
        else:
            raise TypeError(f"Please either specify pointcloud file path (str) or idp.PointCloud objects, not {type(p)}")


class Sensor:

    def __init__(self):
        self.id = 0
        self.label = ""
        # Sensor type in [frame, fisheye, spherical, rpc]
        self.type = "frame"
        self.width = 0  # in int
        self.height = 0  # in int

        # sensor actual size
        self.w_mm = 0   
        self.h_mm = 0 
        
        # pixel scale in mm
        self.pixel_width = 0.0
        self.pixel_height = 0.0
        self.pixel_size = []

        self.focal_length = 0.0  # in mm

        self.calibration = Calibration()

    def in_img_boundary(self, polygon_hv, ignore=None, log=False):
        """Judge whether given polygon is in the image area, and move points to boundary if specify ignore.

        Parameters
        ----------
        polygon_hv : numpy nx2 array
            [horizontal, vertical] points
        ignore : str | None, optional
            None: strickly in image area
            'x': only y (vertical) in image area, x can outside image
            'y': only x (horizontal) in image area, y can outside image
        log : bool, optional
            whether print log for debugging, by default False

        Returns
        -------
        None | polygon_hv
        """
        w, h = self.width, self.height
        coord_min = polygon_hv.min(axis=0)
        x_min, y_min = coord_min[0], coord_min[1]
        coord_max= polygon_hv.max(axis=0)
        x_max, y_max = coord_max[0], coord_max[1]

        if ignore is None:
            if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                if log: print('X ', (x_min, x_max, y_min, y_max))
                return None
            else:
                if log: print('O ', (x_min, x_max, y_min, y_max))
                return polygon_hv
        elif ignore=='x':
            if y_min < 0 or y_max > h:
                if log: print('X ', (x_min, x_max, y_min, y_max))
                return None
            else:
                # replace outside points to image boundary
                polygon_hv[polygon_hv[:, 0] < 0, 0] = 0
                polygon_hv[polygon_hv[:, 0] > w, 0] = w
                if log: print('O ', (x_min, x_max, y_min, y_max))
                return polygon_hv
        elif ignore=='y':
            if x_min < 0 or x_max > w:
                if log: print('X ', (x_min, x_max, y_min, y_max))
                return None
            else:
                # replace outside point to image boundary
                polygon_hv[polygon_hv[:, 1] < 0, 1] = 0
                polygon_hv[polygon_hv[:, 1] > h, 1] = h
                if log: print('O ', (x_min, x_max, y_min, y_max))
                return polygon_hv


class Photo:

    """
    This are used for pix4d
    class Image:
        def __init__(self, name, path, w, h, pmat, cam_matrix, rad_distort, tan_distort, cam_pos, cam_rot):
            # external parameters
            self.name = name   -> label
            self.path = path
            self.w = w    -> sensor property
            self.h = h    -> sensor property
            self.pmat = pmat   # transform matrix? 3x4 -> 4x4
            self.cam_matrix = cam_matrix
            #self.rad_distort = rad_distort   # seems not used
            #self.tan_distort = tan_distort
            self.cam_pos = cam_pos   -> location
            self.cam_rot = cam_rot   -> rotation
    """

    def __init__(self):
        self.id = 0
        self.path = ""
        self._path = ""
        self.label = ""
        self.sensor_id = 0
        self.enabled = False

        # reconstruction info in local coord
        self.cam_matrix = None # np.zeros(3,3) -> K
        self.location = None  # np.zeros(3,) -> t
        self.rotation = None  # np.zeros(3,3) -> R
        # metashape: 4x4 matrix describing photo location in the chunk coordinate system -> K[R t]
        # pix4d: 3x4 pmatrix
        self.transform = None  # 
        self.translation = None  # np.zeros(3,)

        # output infomation
        self.position = None # -> in outputs geo_coordiantes

        # meta info, not necessary in current version
        #self.time = ""
        #self.gps = {"altitude": 0.0, "latitude": 0.0, "longitude": 0.0}
        #self.xyz = {"X": 0, "Y": 0, "Z": 0}
        #self.orientation = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


    def img_exists(func):
        # the decorator to check if image exists
        def wrapper(self, *args, **kwargs):
            if self.path != "" or not os.path.exists(self.path):
                raise FileNotFoundError("Could not operate if not specify correct image file path")
            return func(self, *args, **kwargs)

        return wrapper

    @img_exists
    def get_imarray(self, roi=None):
        pass

    @property
    def center(self):
        # the camera center? is the last column of transform matrix
        # correct!
        return self.transform[0:3, 3]

    #def get_rotation_r(self):
        # the camera rotation R is the first 3x3 part of transform matrix, ideally,
        # but actually is self.rotation, not the same
        # return self.transform[0:3, 0:3]
    #    pass


class Calibration:

    def __init__(self):
        self.software = "metashape"
        self.type = "frame"
        # focal length
        self.f = 0.0  # unit is px, not mm for pix4d
        #self.f_unit = "px"

        # principle point offset
        # metashape: In the older versions Cx and Cy were given in pixels from the top-left corner of the image,
        #            in the latest release version they are measured as offset from the image center,
        #            https://www.agisoft.com/forum/index.php?topic=5827.0
        self.cx = 0.0  # pix4d -> px
        #self.cx_unit = "px" 
        self.cy = 0.0
        #self.cy_unit = "px"

        # [metashape only] affinity and non-orthogonality (skew) coefficients (in pixels)
        self.b1 = 0.0
        self.b2 = 0.0

        # pix4d -> Symmetrical Lens Distortion Coeffs
        # metashape -> radial distortion coefficients (dimensionless)
        self.k1 = 0.0
        self.k2 = 0.0
        self.k3 = 0.0
        self.k4 = 0.0

        # pix4d -> Tangential Lens Distortion Coeffs
        # metashape -> tangential distortion coefficient
        self.t1 = self.p1 = 0.0  # metashape -> p1
        self.t2 = self.p2 = 0.0  # metashape -> p2
        self.t3 = self.p3 = 0.0  # metashape -> p3
        self.t4 = self.p4 = 0.0  # metashape -> p4

    def calibrate(self, u, v):
        if self.software == "pix4d":
            if self.type == "frame":
                return self._calibrate_pix4d_frame(u, v)
            else:
                raise NotImplementedError(
                    f"Can not calibrate camera type [{self.type}], "
                    "only support [frame] currently"
                )
        elif self.software == "metashape":
            if self.type == "frame":
                return self._calibrate_metashape_frame(u, v)
            else:
                raise NotImplementedError(
                    f"Can not calibrate camera type [{self.type}], "
                    "only support [frame] currently"
                )
        else:
            raise TypeError(
                f"Could only handle [pix4d | metashape] projects, "
                "not {self.type}")

    def _calibrate_pix4d_frame(self, u, v):
        """Convert pix4d produced undistorted images pixel coordinate
        to original image pixel coordinate

        Parameters
        ----------
        u : ndarray
            the x pixel coordinate
        v : ndarray
            the y pixel coordinate

        Returns
        -------
        xb, yb: 
            the pixel coordinate on the raw image

        Notes
        -----
        Formula please refer: #2.1.2 in
        https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        """
        f = self.f
        cx = self.cx
        cy = self.cy

        xh = (u - cx) / f
        yh = (v - cy) / f

        r2 = xh ** 2 + yh ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3
        a1 = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        xhd = a1 * xh + 2 * self.t1 * xh * yh + self.t2 * (r2 + 2 * xh ** 2)
        yhd = a1 * yh + 2 * self.t2 * xh * yh + self.t1 * (r2 + 2 * yh ** 2)

        xb = f * xhd + cx
        yb = f * yhd + cy

        return xb, yb

    def _calibrate_metashape_frame(self, u, v):
        pass


class ChunkTransform:

    def __init__(self):
        self.matrix = None
        self.rotation = None
        self.translation = None
        self.scale = None
        self.matrix_inv = None


def filter_closest_img(p4d, img_dict, plot_geo, dist_thresh=None, num=None):
    """[summary]

    Parameters
    ----------
    img_dict : dict
        The outputs dict of geo2raw.get_img_coords_dict()
    plot_geo : nx3 ndarray
        The plot boundary polygon vertex coordinates
    num : None or int
        Keep the closest {x} images
    dist_thresh : None or float
        If given, filter the images smaller than this distance first

    Returns
    -------
    dict
        the same structure as output of geo2raw.get_img_coords_dict()
    """
    dist_geo = []
    dist_name = []
    
    img_dict_sort = {}
    for img_name, img_coord in img_dict.items():

        xmin_geo, ymin_geo = plot_geo[:,0:2].min(axis=0)
        xmax_geo, ymax_geo = plot_geo[:,0:2].max(axis=0)
        xctr_geo = (xmax_geo + xmin_geo) / 2
        yctr_geo = (ymax_geo + ymin_geo) / 2

        ximg_geo, yimg_geo, _ = p4d.img[img_name].cam_pos

        image_plot_dist = np.sqrt((ximg_geo-xctr_geo) ** 2 + (yimg_geo - yctr_geo) ** 2)

        if dist_thresh is not None and image_plot_dist > dist_thresh:
            # skip those image-plot geo distance greater than threshold
            continue
        else:
            # if not given dist_thresh, record all
            dist_geo.append(image_plot_dist)
            dist_name.append(img_name)

    if num is None:
        # not specify num, use all
        num = len(dist_name)
    else:
        num = min(len(dist_name), num)

    dist_geo_idx = np.asarray(dist_geo).argsort()[:num]
    img_dict_sort = {dist_name[idx]:img_dict[dist_name[idx]] for idx in dist_geo_idx}
    
    return img_dict_sort