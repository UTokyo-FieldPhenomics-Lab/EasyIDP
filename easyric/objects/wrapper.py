from copy import copy
from easyric.io import pix4d
from easyric.objects.software import Pix4dFiles, PhotoScanFiles, OpenSfMFiles


class EasyParams:

    def __init__(self):
        self.offset = None
        self.img = None
        # from cicp file
        self.F = None
        self.Px = None
        self.Py = None
        self.K1 = None
        self.K2 = None
        self.K3 = None
        self.T1 = None
        self.T2 = None

    def from_pix4d_files(self, pix4dfiles):
        # --------------------
        # >>> p4d.offset.x
        # 368109.00
        # >>> p4d.Py
        # 3.9716578516421746
        # >>> p4d.img[0].name
        # ''DJI_0172.JPG''
        # >>> p4d.img['DJI_0172.JPG']
        # <class Image>
        # >>> p4d.img[0].pmat
        # pmat_ndarray
        # --------------------
        self.offset = OffSet(pix4dfiles.get_offsets())
        vars(self).update(pix4dfiles.get_cicp_dict())
        self.img = ImageSet(pix4dfiles.raw_img_path, pix4dfiles.get_ccp_dict())

    def from_photoscan_files(self):
        pass

    def from_opensfm_files(self):
        pass


class OffSet:

    def __init__(self, offsets):
        self.x = offsets[0]
        self.y = offsets[1]
        self.z = offsets[2]


class ImageSet:

    def __init__(self, img_path, ccp_dict):
        self.names = list(ccp_dict.keys())
        self.img = []
        for img_name in self.names:
            temp = copy(ccp_dict[img_name])
            temp['name'] = img_name
            # temp['pmat'] = pmat_dict[img_name]
            temp['path'] = f"{img_path}/{img_name}"
            self.img.append(Image(**temp))

    def __getitem__(self, key):
        if isinstance(key, int):  # index by photo name
            return self.img[key]
        elif isinstance(key, str):  # index by photo order
            return self.img[self.names.index(key)]
        elif isinstance(key, slice):
            return self.img[key]
        else:
            print(key)
            return None


class Image:

    def __init__(self, name, path, w, h, cam_matrix, rad_distort, tan_distort, cam_pos, cam_rot):
        self.name = name
        self.path = path
        self.w = w
        self.h = h
        # self.pmat = pmat
        self.cam_matrix = cam_matrix
        self.rad_distort = rad_distort
        self.tan_distort = tan_distort
        self.cam_pos = cam_pos
        self.cam_rot = cam_rot
