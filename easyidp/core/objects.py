import numpy as np


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


class MetashapeChunkTransform:

    def __init__(self):
        self.matrix = None
        self.rotation = None
        self.translation = None
        self.scale = None


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