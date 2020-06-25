from copy import copy
import numpy as np


class OffSet:

    def __init__(self, offsets):
        self.x = offsets[0]
        self.y = offsets[1]
        self.z = offsets[2]
        self.np = np.asarray(offsets)

class ImageSet:

    def __init__(self, img_path, pmat_dict, ccp_dict):
        # container for external camera parameters for all raw images
        self.names = list(ccp_dict.keys())
        self.img = []
        for img_name in self.names:
            temp = copy(ccp_dict[img_name])
            temp['name'] = img_name
            temp['pmat'] = pmat_dict[img_name]
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

    def __init__(self, name, path, w, h, pmat, cam_matrix, rad_distort, tan_distort, cam_pos, cam_rot):
        # external parameters
        self.name = name
        self.path = path
        self.w = w
        self.h = h
        self.pmat = pmat
        self.cam_matrix = cam_matrix
        self.rad_distort = rad_distort
        self.tan_distort = tan_distort
        self.cam_pos = cam_pos
        self.cam_rot = cam_rot
