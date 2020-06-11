from copy import copy
from easyric.io import pix4d

class Params:

    def __init__(self, param_path, project_name, raw_img_folder, dom_path, dsm_path):
        self.xyz_file  = f"{param_path}/{project_name}_offset.xyz"
        self.pmat_file = f"{param_path}/{project_name}_pmatrix.txt"
        self.cicp_file = f"{param_path}/{project_name}_pix4d_calibrated_internal_camera_parameters.cam"
        self.ccp_file  = f"{param_path}/{project_name}_calibrated_camera_parameters.txt"

        self.img_path  = raw_img_folder
        self.dom_file  = dom_path
        self.dsm_file  = dsm_path

        # for direct usage (img_name as index)
        # ----------------------------------
        # >>> proj.pmat_dict['DJI_0172.JPG']
        # pmat_ndarray
        # >>> proj.cicp_dict['Py']
        # 3.9716578516421746
        # ----------------------------------
        self.offset_dict = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.offset_dict['x'], self.offset_dict['y'], self.offset_dict['z'] = pix4d.read_xyz(self.xyz_file)
        self.pmat_dict = pix4d.read_pmat(self.pmat_file)
        self.ccp_dict = pix4d.read_ccp(self.ccp_file)
        self.cicp_dict = pix4d.read_cicp(self.cicp_file)

        # pythonic useage
        # --------------------
        # >>> p4d.offset_x
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
        self.offset_x, self.offset_y, self.offset_z = pix4d.read_xyz(self.xyz_file)
        vars(self).update(self.cicp_dict)
        self.img = ImageSet(self.img_path, self.pmat_dict, self.ccp_dict)


class ImageSet:

    def __init__(self, img_path, pmat_dict, ccp_dict):
        self.names = list(pmat_dict.keys())
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