import os
from easyric.io import pix4d
from easyric.objects.wrapper import EasyParams

class Pix4dFiles:

    def __init__(self, project_path, raw_img_path, project_name=None):
        """
        The function to automatic drag necessary files from Pix4D default project folders
        :param project_path: The folder path of pix4d project
        :param raw_img_path: THe raw image folder produced pix4d project
        """
        self.project_path = self._get_full_path(project_path)
        if project_name is None:
            self.project_name = os.path.basename(self.project_path)
        else:
            self.project_name = project_name

        self.raw_img_path = os.path.normpath(raw_img_path)

        sub_folder = os.listdir(self.project_path)

        if '1_initial' in sub_folder:
            self._original_specify()
        else:
            print(f'Could not find "1_initial" folder in "{self.project_path}" which means not a Pix4D project folder\n'
                  f'  Please manual given each file paths by self.manual_specify(param_folder, dom_path, etc.)')

        self.xyz_file = None
        self.pmat_file = None
        self.cicp_file = None
        self.ccp_file = None
        self.ply_file = None
        self.dom_file = None
        self.dsm_file = None

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

    def manual_specify(self, param_folder, dom_path=None, dsm_path=None, ply_path=None):
        self.xyz_file = self._get_full_path(f"{param_folder}/{self.project_name}_offset.xyz")
        self.pmat_file = self._get_full_path(f"{param_folder}/{self.project_name}_pmatrix.txt")
        self.cicp_file = self._get_full_path(f"{param_folder}/{self.project_name}_pix4d_calibrated_internal_camera_parameters.cam")
        self.ccp_file = self._get_full_path(f"{param_folder}/{self.project_name}_calibrated_camera_parameters.txt")

        self.ply_file = self._get_full_path(ply_path)
        self.dom_file = self._get_full_path(dom_path)
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


class PhotoScanFiles:
    pass


class OpenSfMFiles:
    pass