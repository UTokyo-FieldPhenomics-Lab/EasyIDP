import numpy as np
from easyidp.core.math import apply_transform

class ReconsProject:
    """
    Equals to each individual Pix4D project & each chunk in Metashape project
    """

    def __init__(self, software="metashape"):
        """
        Parameters
        ----------
        software: str
            choose from ["agisoft", "Agisoft", "metashape", "MetaShape", "Metashape", "PhotoScan", "photoscan"
                         "pix4d", "Pix4D", "Pix4DMapper", "Pix4Dmapper"]
        """
        if software in ["agisoft", "Agisoft", "metashape", "MetaShape", "Metashape", "PhotoScan", "photoscan"]:
            self.software = "metashape"
        elif software in ["pix4d", "Pix4D", "Pix4DMapper", "Pix4Dmapper"]:
            self.software = "pix4d"
        else:
            raise LookupError("Only [pix4d] and [metashape] are supported at current stage")

        self.label = ""
        self.enabled = True

        self.sensors = {}
        self.photos = {}

        # metashape chunk.transform.matrix
        # from kunihiro kodama's Metashape API usage <kkodama@kazusa.or.jp>
        # >>> transm = chunk.transform.matrix
        # >>> invm = Metashape.Matrix.inv(chunk.transform.matrix)
        # invm.mulp(local_vec) --> transform chunk local coord to world coord(if you handle vec in local coord)
        # how to calculate from xml data:
        # https://www.agisoft.com/forum/index.php?topic=6176.0
        self.transform = MetashapeChunkTransform()

        self.crs_str = ""


    def world2local(self, points_np):
        return apply_transform(self.transform.matrix, points_np)


    def local2world(self, points_np):
        if self.transform.matrix_inv is None:
            self.transform.matrix_inv = np.linalg.inv(self.transform.matrix)

        return apply_transform(self.transform.matrix_inv, points_np)


    def project_world_points_on_raw(self, points, photo_id, distortion_correct=False):
        if self.software == "metashape":
            return self._metashape_project_world_points_on_raw(points, photo_id, distortion_correct)
        elif self.software == "pix4d":
            return self._pix4d_project_world_points_on_raw(points, photo_id, distortion_correct)
        else:
            raise KeyError("Current version only support [metashape] & [pix4d]")


    def _metashape_project_world_points_on_raw(self, points, photo_id, distortion_correct=False):
        dim = len(points.shape)
        camera_i = self.photos[photo_id]
        T = camera_i.get_camera_center()
        R = camera_i.transform[0:3, 0:3]
        XYZ = (points - T).dot(R)

        if dim == 1:
            X, Y, Z = XYZ
        elif dim == 2:
            X = XYZ[:, 0]
            Y = XYZ[:, 1]
            Z = XYZ[:, 2]
        else:
            raise ValueError("only 1x3 single point or nx3 multiple points are accepted")

        x = X / Z
        y = Y / Z

        w  = self.sensors[camera_i.sensor_idx].width
        h  = self.sensors[camera_i.sensor_idx].height
        f  = self.sensors[camera_i.sensor_idx].calibration.f
        cx = self.sensors[camera_i.sensor_idx].calibration.cx
        cy = self.sensors[camera_i.sensor_idx].calibration.cy

        # without distortion
        if not distortion_correct:
            K = np.asarray([[f, 0, w / 2 + cx, 0],
                            [0, f, h / 2 + cy, 0],
                            [0, 0, 1         , 0]])

            if dim == 1:
                Pch = np.asarray([x, y, 1, 1])
                Ppix = Pch.dot(K.T)
                return Ppix[0:2]
            elif dim == 2:
                # make [x, y, 1, 1] for multiple points
                Pch = np.vstack([x, y, np.ones(len(x)), np.ones(len(x))]).T
                Ppix = Pch.dot(K.T)
                return Ppix[:, 0:2]
            else:
                raise ValueError("only 1x3 single point or nx3 multiple points are accepted")
        else:
            # with distortion
            r2 = x**2 + y**2
            r4 = r2 ** 2
            r6 = r2 ** 3
            r8 = r2 ** 4

            k1 = self.sensors[camera_i.sensor_idx].calibration.k1
            k2 = self.sensors[camera_i.sensor_idx].calibration.k2
            k3 = self.sensors[camera_i.sensor_idx].calibration.k3
            k4 = self.sensors[camera_i.sensor_idx].calibration.k4

            p1 = self.sensors[camera_i.sensor_idx].calibration.t1
            p2 = self.sensors[camera_i.sensor_idx].calibration.t2
            b1 = self.sensors[camera_i.sensor_idx].calibration.b1
            b2 = self.sensors[camera_i.sensor_idx].calibration.b2

            eq_part1 = (1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8)

            x_prime = x * eq_part1 + (p1 * (r2 + 2 * x ** 2) + 2 * p2 * x * y)
            y_prime = y * eq_part1 + (p2 * (r2 + 2 * y ** 2) + 2 * p1 * x * y)

            u = w * 0.5 + cx + x_prime * f + x_prime * b1 + y_prime * b2
            v = h * 0.5 + cy + y_prime * f

            if dim == 1:
                return np.asarray([u, v])
            elif dim == 2:
                return np.vstack([u, v]).T
            else:
                return None

    def _pix4d_project_world_points_on_raw(self, poitns, photo_id, distortion_correct=False):
        pass


class MetashapeChunkTransform:

    def __init__(self):
        self.matrix = None
        self.rotation = None
        self.translation = None
        self.scale = None
        self.matrix_inv = None

class Sensor:

    def __init__(self):
        self.idx = 0
        self.label = ""
        # Sensor type in [frame, fisheye, spherical, rpc]
        self.type = "frame"
        self.width = 0
        self.width_unit = "px"
        self.height = 0
        self.height_unit = "px"

        # pixel scale in mm
        self.pixel_width = 0.0
        self.pixel_width_unit = "mm"
        self.pixel_height = 0.0
        self.pixel_height_unit = "mm"

        self.focal_length = 0.0
        self.focal_length_unit = "mm"

        self.calibration = Calibration()


class Calibration:

    def __init__(self):
        self.software = "metashape"
        # focal length
        self.f = 0.0
        self.f_unit = "px"

        # principle point offset
        # metashape: In the older versions Cx and Cy were given in pixels from the top-left corner of the image,
        #            in the latest release version they are measured as offset from the image center,
        #            https://www.agisoft.com/forum/index.php?topic=5827.0
        self.cx = 0.0         # pix4d -> px
        self.cx_unit = "px"   # pix4d -> mm
        self.cy = 0.0
        self.cy_unit = "px"

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
        self.t1 = 0.0    # metashape -> p1
        self.t2 = 0.0    # metashape -> p2
        self.t3 = 0.0    # metashape -> p3
        self.t4 = 0.0    # metashape -> p4

    def calibrate(self):
        pass


class Photo:

    def __init__(self):
        self.idx = 0
        self.path = ""
        self.label = ""
        self.sensor_idx = 0
        self.enabled = False

        # reconstruction info
        self.location = None      # np.zeros(3)
        self.rotation = None      # np.zeros(3)
        self.transform = None     # 4x4 matrix describing photo location in the chunk coordinate system
        self.translation = None   # np.zeros(3)

        # meta info, not necessary in current version
        # todo: support reading these meta info
        self.time = ""
        self.gps = {"altitude": 0.0, "latitude": 0.0, "longitude": 0.0}
        self.xyz = {"X": 0, "Y": 0, "Z": 0}
        self.orientation = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    def get_image(self, roi=None):
        pass

    def get_camera_center(self):
        # the camera center is the last column of transform matrix
        # correct!
        return self.transform[0:3, 3]

    def get_rotation_r(self):
        # the camera rotation R is the first 3x3 part of transform matrix, ideally,
        # but actually is self.rotation, not the same
        # return self.transform[0:3, 0:3]
        pass


class Image:
    # number of wxhxn matrix to describe image data (ndarray)
    # can be used in Camera & GeoTiff
    def __init__(self):
        self.values = np.zeros((0, 0, 0))
        self.mask = np.zeros((0, 0))

    def copy(self):
        pass

    def resize(self, w, h):
        pass

    def save(self, path):
        pass

    def clip_by_roi(self, roi):
        # return <class Image>, offsets
        pass

    def clip_by_mask(self, mask):
        # return <class Image>, offsets
        pass


class GeoTiff:

    def __init__(self):
        pass


class ShapeFile:

    def __init__(self):
        pass


class ROI:

    def __init__(self):
        pass