import os
from easyric.io import pix4d
from easyric.objects.wrapper import OffSet, ImageSet

class Pix4D:

    def __init__(self, project_path, raw_img_path, project_name=None,
                 param_folder=None, dom_path=None, dsm_path=None, ply_path=None):

        ######################
        # Project Attributes #
        ######################
        self.project_path = self._get_full_path(project_path)
        if project_name is None:
            self.project_name = os.path.basename(self.project_path)
        else:
            self.project_name = project_name

        self.raw_img_path = os.path.normpath(raw_img_path)

        sub_folder = os.listdir(self.project_path)

        #################
        # Project Files #
        #################
        self.xyz_file = None
        self.pmat_file = None
        self.cicp_file = None
        self.ccp_file = None
        self.ply_file = None
        self.dom_file = None
        self.dsm_file = None

        if '1_initial' in sub_folder:
            self._original_specify()
        else:
            if param_folder is None:
                raise FileNotFoundError(f'[Wrapper][Pix4D] Current folder |{self.project_path}| is not a standard '
                                        f'pix4d default projects, "1_initial" folder not found and `param_folder` not specified')
            else:
                self._manual_specify(param_folder, dom_path, dsm_path, ply_path)

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

        self.offset = OffSet(self.get_offsets())
        vars(self).update(self.get_cicp_dict())
        self.img = ImageSet(img_path=self.raw_img_path,
                            pmat_dict=self.get_pmat_dict(),
                            ccp_dict=self.get_ccp_dict())

    def _original_specify(self):
        sub_folder = os.listdir(self.project_path)

        self.xyz_file = f"{self.project_path}/1_initial/params/{self.project_name}_offset.xyz"
        self.pmat_file = f"{self.project_path}/1_initial/params/{self.project_name}_pmatrix.txt"
        self.cicp_file = f"{self.project_path}/1_initial/params/{self.project_name}_pix4d_calibrated_internal_camera_parameters.cam"
        self.ccp_file = f"{self.project_path}/1_initial/params/{self.project_name}_calibrated_camera_parameters.txt"

        self.ply_file = None
        if '2_densification' in sub_folder:
            dens_folder = f"{self.project_path}/2_densification/point_cloud"

            self.plyfile = self._check_end(dens_folder, '.ply')

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

        if ply_path is None:
            try_ply = f"{self.project_name}_group1_densified_point_cloud.ply"
            self.ply_file = self._get_full_path(f"{self.project_path}/{try_ply}")
            if self.ply_file is not None:
                print(f"[Wrapper][Pix4D] No ply given, however find '{try_ply}' at current project folder")
        else:
            self.ply_file = self._get_full_path(ply_path)

        if dom_path is None:
            try_dom = f"{self.project_name}_transparent_mosaic_group1.tif"
            self.dom_file = self._get_full_path(f"{self.project_path}/{try_dom}")
            if self.dom_file is not None:
                print(f"[Wrapper][Pix4D] No dom given, however find '{try_dom}' at current project folder")
        else:
            self.dom_file = self._get_full_path(dom_path)

        if dsm_path is None:
            try_dsm = f"{self.project_name}_dsm.tif"
            self.dsm_file = self._get_full_path(f"{self.project_path}/{try_dsm}")
            if self.dsm_file is not None:
                print(f"[Wrapper][Pix4D] No dsm given, however find '{try_dsm}' at current project folder")
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

    def get_offsets(self):
        return pix4d.read_xyz(self.xyz_file)

    def get_pmat_dict(self):
        return pix4d.read_pmat(self.pmat_file)

    def get_ccp_dict(self):
        return pix4d.read_ccp(self.ccp_file)

    def get_cicp_dict(self):
        return pix4d.read_cicp(self.cicp_file)


class PhotoScan:
    pass


class OpenSfM:
    pass