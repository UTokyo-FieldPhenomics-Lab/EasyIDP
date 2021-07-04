import os
import numpy as np
from copy import copy
from easyric.io import pix4d, geotiff, shp, plot
from easyric.calculate import geo2raw, geo2tiff, raw2raw

####################
# Software wrapper #
####################

class Pix4D:

    def __init__(self, project_path, raw_img_path=None, project_name=None,
                 param_folder=None, dom_path=None, dsm_path=None, ply_path=None):

        ######################
        # Project Attributes #
        ######################
        self.project_path = self._get_full_path(project_path)
        sub_folder = os.listdir(self.project_path)

        if project_name is None:
            self.project_name = os.path.basename(self.project_path)
        else:
            self.project_name = project_name

        if raw_img_path is not None:
            self.raw_img_path = os.path.normpath(raw_img_path)
        else:
            self.raw_img_path = None

        #################
        # Project Files #
        #################
        self.xyz_file = None
        self.pmat_file = None
        self.cicp_file = None
        self.ccp_file = None
        self.campos_file = None
        self.ply_file = None
        self.dom_file = None
        self.dsm_file = None
        self.dom_header = None
        self.dsm_header = None

        if '1_initial' in sub_folder:
            self._original_specify()
        else:
            if param_folder is None:
                raise FileNotFoundError(f'[Wrapper][Pix4D] Current folder |{self.project_path}| is not a standard '
                                        f'pix4d default projects, "1_initial" folder not found and `param_folder` not specified')
            else:
                self._manual_specify(param_folder, dom_path, dsm_path, ply_path)

        if self.dom_file is not None:
            self.dom_header = geotiff.get_header(self.dom_file)
        if self.dsm_file is not None:
            self.dsm_header = geotiff.get_header(self.dsm_file)

        ###############
        # Init Params #
        ###############
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

        self.offset = OffSet(self._get_offsets())
        self.img_pos = self._get_campos_df()
        vars(self).update(self._get_cicp_dict())
        self.img = ImageSet(img_path=self.raw_img_path,
                            pmat_dict=self._get_pmat_dict(),
                            ccp_dict=self._get_ccp_dict(),
                            img_pos=self.img_pos)

    def _original_specify(self):
        sub_folder = os.listdir(self.project_path)

        self.xyz_file = f"{self.project_path}/1_initial/params/{self.project_name}_offset.xyz"
        self.pmat_file = f"{self.project_path}/1_initial/params/{self.project_name}_pmatrix.txt"
        self.cicp_file = f"{self.project_path}/1_initial/params/{self.project_name}_pix4d_calibrated_internal_camera_parameters.cam"
        self.ccp_file = f"{self.project_path}/1_initial/params/{self.project_name}_calibrated_camera_parameters.txt"
        self.campos_file = f"{self.project_path}/1_initial/params/{self.project_name}_calibrated_images_position.txt"

        if self.raw_img_path is None:
            undistorted_path = f"{self.project_path}/1_initial/images/undistorted_images"
            if os.path.exists(undistorted_path):
                self.raw_img_path = undistorted_path
            else:
                raise FileNotFoundError("raw image file not given, and could not find undistorted images outputs in Pix4D project")

        self.ply_file = None
        if '2_densification' in sub_folder:
            dens_folder = f"{self.project_path}/2_densification/point_cloud"

            self.ply_file = self._check_end(dens_folder, '.ply')

        self.dom_file = None
        self.dsm_file = None
        if '3_dsm_ortho' in sub_folder:
            dsm_folder = f"{self.project_path}/3_dsm_ortho/1_dsm"
            dom_folder = f"{self.project_path}/3_dsm_ortho/2_mosaic"

            self.dsm_file = self._check_end(dsm_folder, '.tif')
            self.dom_file = self._check_end(dom_folder, '.tif')

    def _manual_specify(self, param_folder, dom_path=None, dsm_path=None, ply_path=None):
        self.xyz_file = self._get_full_path(f"{param_folder}/{self.project_name}_offset.xyz")
        self.pmat_file = self._get_full_path(f"{param_folder}/{self.project_name}_pmatrix.txt")
        self.cicp_file = self._get_full_path(f"{param_folder}/{self.project_name}_pix4d_calibrated_internal_camera_parameters.cam")
        self.ccp_file = self._get_full_path(f"{param_folder}/{self.project_name}_calibrated_camera_parameters.txt")
        self.campos_file = self._get_full_path(f"{param_folder}/{self.project_name}_calibrated_images_position.txt")

        if ply_path is None:
            try_ply = f"{self.project_name}_group1_densified_point_cloud.ply"
            self.ply_file = self._get_full_path(f"{self.project_path}/{try_ply}")
            if self.ply_file is not None:
                print(f"[Init][Pix4D] No ply given, however find '{try_ply}' at current project folder")
        else:
            self.ply_file = self._get_full_path(ply_path)

        if dom_path is None:
            try_dom = f"{self.project_name}_transparent_mosaic_group1.tif"
            self.dom_file = self._get_full_path(f"{self.project_path}/{try_dom}")
            if self.dom_file is not None:
                print(f"[Init][Pix4D] No dom given, however find '{try_dom}' at current project folder")
        else:
            self.dom_file = self._get_full_path(dom_path)

        if dsm_path is None:
            try_dsm = f"{self.project_name}_dsm.tif"
            self.dsm_file = self._get_full_path(f"{self.project_path}/{try_dsm}")
            if self.dsm_file is not None:
                print(f"[Init][Pix4D] No dsm given, however find '{try_dsm}' at current project folder")
        else:
            self.dsm_file = self._get_full_path(dsm_path)

    @staticmethod
    def _check_end(folder, ext):
        find_path = None
        if os.path.exists(folder):
            # find the first ply file as output (may cause problem)
            for file in os.listdir(folder):
                if file.endswith(ext):
                    find_path = f"{folder}/{file}"
                    break

        return find_path

    @staticmethod
    def _get_full_path(short_path):
        if isinstance(short_path, str):
            return os.path.abspath(os.path.normpath(short_path)).replace('\\', '/')
        else:
            return None

    def _get_offsets(self):
        return pix4d.read_xyz(self.xyz_file)

    def _get_pmat_dict(self):
        return pix4d.read_pmat(self.pmat_file)

    def _get_ccp_dict(self):
        return pix4d.read_ccp(self.ccp_file)

    def _get_cicp_dict(self):
        return pix4d.read_cicp(self.cicp_file)

    def _get_campos_df(self):
        return pix4d.read_cam_pos(self.campos_file)

    #################
    # Easy use apis #
    #################

    # ======== io.shp =========
    def read_shp2d(self, shp_path, shp_proj=None, geotiff_proj=None):
        if geotiff_proj is None:
            proj = self.dsm_header['proj']
        elif geotiff_proj == 'Null':   # the special params to do noting transfrom
            proj = None
        else:
            proj = geotiff_proj

        shp_dict = shp.read_shp2d(shp_path, shp_proj=shp_proj, geotiff_proj=proj)

        return shp_dict

    def read_shp3d(self, shp_path, get_z_by='mean', shp_proj=None, geotiff_proj=None):
        shp_dict = shp.read_shp3d(shp_path, self.dsm_file, get_z_by, shp_proj, geotiff_proj, geo_head=self.dsm_header)
        return shp_dict

    # ======== io.geotiff =========


    # ======== io.plot =========


    # ======== calculate.geo2raw =========


    # ======== calculate.geo2tiff =========


    # ======== calculate.raw2raw =========



class PhotoScan:
    pass


class OpenSfM:
    pass

#################
# Used Objects  #
#################
class OffSet:

    def __init__(self, offsets):
        self.x = offsets[0]
        self.y = offsets[1]
        self.z = offsets[2]
        self.np = np.asarray(offsets)

class ImageSet:

    def __init__(self, img_path, pmat_dict, ccp_dict, img_pos):
        # container for external camera parameters for all raw images
        pix4d_used = list(ccp_dict.keys())
        self.names = []
        self.img = []
        
        # in case the img_path has subfolders
        for fpathe, dirs, fs in os.walk(img_path):
            for f in fs:
                full_path = os.path.join(fpathe,f)
                if f in pix4d_used:
                    # f is img_name
                    temp = copy(ccp_dict[f])
                    temp['name'] = f
                    temp['pmat'] = pmat_dict[f]
                    temp['path'] = full_path
                    temp["cam_pos"] = img_pos.loc[f, :].values
                    self.img.append(Image(**temp))
                    self.names.append(f)

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
