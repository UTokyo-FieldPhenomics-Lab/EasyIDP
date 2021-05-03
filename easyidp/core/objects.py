import numpy as np


class ReconsProject:

    def __init__(self, software="metashape"):
        """
        Parameters
        ----------
        software: str
            choose from ["agisoft", "Agisoft", "metashape", "MetaShape", "Metashape",
                         "pix4d", "Pix4D", "Pix4DMapper", "Pix4Dmapper"]
        """
        if software in ["agisoft", "Agisoft", "metashape", "MetaShape", "Metashape"]:
            self.software = "metashape"
        elif software in ["pix4d", "Pix4D", "Pix4DMapper", "Pix4Dmapper"]:
            self.software = "pix4d"
        else:
            raise LookupError("Only [pix4d] and [metashape] are supported at current stage")


class Camera:

    def __init__(self):
        self.name = ""
        self.type = "perspective"
        self.enabled = False
        self.image = Image()
        self.location_covariance = None
        self.location = np.zeros(3)
        self.meta = dict()
        self.rotation = np.zeros(3)
        self.transform = np.zeros((4, 4))  # 4x4 matrix describing photo location in the chunk coordinate system


class Photos:

    def __init__(self):
        self.path = ""
        self.camera = None
        # meta info
        self.time = ""
        self.gps = {"altitude": 0.0, "latitude": 0.0, "longitude": 0.0}
        self.xyz = {"X": 0, "Y": 0, "Z": 0}
        self.orientation = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    def get_image(self, roi=None):
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