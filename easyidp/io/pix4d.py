import os
import numpy as np
from easyidp.core.objects import ReconsProject, Sensor, Calibration, Photo


def _match_suffix(folder, ext):
    """
    find the first file by given suffix, e.g. *.tif
    Parameters
    ----------
    folder: str
        the path of folder want to find
    ext: str | list
        str -> e.g. "tif"
        list -> e.g. ["ply", "laz"]
            will find all suffix files

    Returns
    -------
    find_path: str | None
        if find -> str of path
        if not find -> None
    """
    find_path = None
    if os.path.exists(folder):
        if isinstance(ext, str):
            one_ext = True
        else:
            one_ext = False

        if one_ext:
            for file in os.listdir(folder):
                if file.endswith(ext):
                    find_path = f"{folder}/{file}"
                    return find_path
        else:
            for ex in ext:
                for file in os.listdir(folder):
                    if file.endswith(ex):
                        find_path = f"{folder}/{file}"
                        return find_path

    return find_path


def _get_full_path(short_path):
    if isinstance(short_path, str):
        return os.path.abspath(os.path.normpath(short_path)).replace('\\', '/')
    else:
        return None


def get_project_structure(project_path:str, project_name=None, force_find=False):
    """
    A fuction to automatically analyze related subfiles in pix4d project folder

    Parameters
    ----------
    project_path: str
        the path to pix4d project file, that folder should contains the following sub-folder:

        \project_path
        |--- 1_initial\
        |--- 2_densification\
        |___ 3_dsm_ortho\

    project_name: str, optional
        by default, the project_name is the same as given project path
        e.g. project_path = xxxx/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d
             then the project name is the same to "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"
        but sometimes, the project_folder is not the project_name, so it can be specified by user:
        e,g, project_path = xxxx/20210303
             but all the files beneath that folder have the prefix of "maize_tanashi"_dsm.tif, then
             the project_name should be "maize_tanashi"

    force_find: bool
        if False, then only choose the files fit pix4d default name rule.
            -> "{dom_folder}/{project_name}_transparent_mosaic_group1.tif"
        But sometimes, those files are renamed and only one file is under certain folder
            -> "{dom_folder}/the_renamed_dom.tif"
        Give True, then it will find the first file that suits the suffix.

    Returns
    -------
    p4d: dict
        a python dictionary that contains the path to each file.
        {
            "param": the folder of parameters
            "pcd": the point cloud file
            "dom": the digital orthomosaic file
            "dsm": the digital surface model file
            "undist_raw": the undistorted images corrected by software (when original image unable to find)
        }
    """
    p4d = {"param": None, "pcd": None, "dom": None, "dsm": None, "undist_raw": None, "project_name": None}

    project_path = _get_full_path(project_path)
    sub_folder = os.listdir(project_path)

    if project_name is None:
        project_name = os.path.basename(project_path)

    p4d["project_name"] = project_name

    param_path = f"{project_path}/1_initial/params"
    undist_folder = f"{project_path}/1_initial/images/undistorted_images"

    dense_folder = f"{project_path}/2_densification/point_cloud"
    ply_file = f"{dense_folder}/{project_name}_group1_densified_point_cloud.ply"
    laz_file = f"{dense_folder}/{project_name}_group1_densified_point_cloud.laz"
    las_file = f"{dense_folder}/{project_name}_group1_densified_point_cloud.las"

    dsm_folder = f"{project_path}/3_dsm_ortho/1_dsm"
    dsm_file = f"{dsm_folder}/{project_name}_dsm.tif"

    dom_folder = f"{project_path}/3_dsm_ortho/2_mosaic"
    dom_file = f"{dom_folder}/{project_name}_transparent_mosaic_group1.tif"

    if '1_initial' not in sub_folder:
        raise FileNotFoundError(f"[io][pix4D] Current folder |{project_path}| is not a standard pix4d projects folder")

    if os.path.exists(param_path):
        p4d["param"] = param_path
    else:
        raise FileNotFoundError("[io][pix4d] can not find important parameter folder ['1_initial/params']")

    if os.path.exists(undist_folder):
        p4d["undist_raw"] = undist_folder

    if os.path.exists(dense_folder):
        if os.path.exists(ply_file):
            p4d["pcd"] = ply_file
        elif os.path.exists(las_file):
            p4d["pcd"] = las_file
        elif os.path.exists(laz_file):
            p4d["laz"] = laz_file
        else:
            if force_find:
                force = _match_suffix(dense_folder, ["ply", "las", "laz"])
                p4d["pcd"] = force

    if os.path.exists(dsm_folder):
        if os.path.exists(dsm_file):
            p4d["dsm"] = dsm_file
        else:
            if force_find:
                force = _match_suffix(dsm_folder, "tif")
                p4d["dsm"] = force

    if os.path.exists(dom_folder):
        if os.path.exists(dom_file):
            p4d["dom"] = dom_file
        else:
            if force_find:
                force = _match_suffix(dom_folder, "tif")
                p4d["dom"] = force

    return p4d


def open_project(project_path, project_name=None, param_folder=None):
    if project_name is None:
        project_name = os.path.basename(project_path)

    recons_proj = ReconsProject(software="pix4d")
    recons_proj.label = project_name

    offset    = read_xyz (f"{param_folder}/{project_name}_offset.xyz")
    pmat_dict = read_pmat(f"{param_folder}/{project_name}_pmatrix.txt")
    cicp_dict = read_cicp(f"{param_folder}/{project_name}_calibrated_internal_camera_parameters.cam")
    ccp_dict  = read_ccp (f"{param_folder}/{project_name}_calibrated_camera_parameters.txt")


def read_xyz(xyz_path):
    """
    read pix4d file PROJECTNAME_offset.xyz
    Parameters
    ----------
    xyz_path: str
        the path to target offset.xyz file

    Returns
    -------
    x, y, z: float
    """
    with open(xyz_path, 'r') as f:
        x, y, z = f.read().split(' ')
    return float(x), float(y), float(z)


def read_pmat(pmat_path):
    """
    read pix4d file PROJECTNAME_pmatrix.txt
    Parameters
    ----------
    pmat_path: str
        the path of pmatrix file.type

    Returns
    -------
    pmat_dict: dict
        pmat_dict = {"DJI_0000.JPG": nparray(3x4), ... ,"DJI_9999.JPG": nparray(3x4)}

    """
    pmat_nb = np.loadtxt(pmat_path, dtype=float, delimiter=None, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,))
    pmat_names = np.loadtxt(pmat_path, dtype=str, delimiter=None, usecols=0)

    pmat_dict = {}

    for i, name in enumerate(pmat_names):
        pmat_dict[name] = pmat_nb[i, :].reshape((3, 4))

    return pmat_dict


def read_cicp(cicp_path):
    """
    Read PROJECTNAME_pix4d_calibrated_internal_camera_parameters.cam file
    Parameters
    ----------
    cicp_path: str

    Returns
    -------
    cicp_dict: dict
    """
    with open(cicp_path, 'r') as f:
        key_pool = ['F', 'Px', 'Py', 'K1', 'K2', 'K3', 'T1', 'T2']
        cam_dict = {}
        for line in f.readlines():
            sp_list = line.split(' ')
            if len(sp_list) == 2:  # lines with param.name in front
                lead, contents = sp_list[0], sp_list[1]
                if lead in key_pool:
                    cam_dict[lead] = float(contents[:-1])
            elif len(sp_list) == 9:
                # Focal Length mm assuming a sensor width of 12.82x8.55mm\n
                w_h = sp_list[8].split('x')
                cam_dict['w_mm'] = float(w_h[0])  # extract e.g. 12.82
                cam_dict['h_mm'] = float(w_h[1][:-4])  # extract e.g. 8.55
    return cam_dict


def read_ccp(ccp_path):
    """
    Read PROJECTNAME_calibrated_camera_parameters.txt
    Parameters
    ----------
    ccp_path: str

    Returns
    -------
    img_configs: dict
    """
    with open(ccp_path, 'r') as f:
        '''
        # for each block
        1   fileName imageWidth imageHeight 
        2-4 camera matrix K [3x3]
        5   radial distortion [3x1]
        6   tangential distortion [2x1]
        7   camera position t [3x1]
        8-10   camera rotation R [3x3]
        '''
        lines = f.readlines()

    img_configs = {}

    file_name = ""
    cam_mat_line1 = ""
    cam_mat_line2 = ""
    cam_rot_line1 = ""
    cam_rot_line2 = ""
    for i, line in enumerate(lines):
        if i < 8:
            pass
        else:
            block_id = (i - 7) % 10
            if block_id == 1:  # [line]: fileName imageWidth imageHeight
                file_name, w, h = line[:-1].split()  # ignore \n character
                img_configs[file_name] = {}
                img_configs['w'] = int(w)
                img_configs['h'] = int(h)
            elif block_id == 2:
                cam_mat_line1 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 3:
                cam_mat_line2 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 4:
                cam_mat_line3 = np.fromstring(line, dtype=float, sep=' ')
                img_configs[file_name]['cam_matrix'] = np.vstack([cam_mat_line1, cam_mat_line2, cam_mat_line3])
            elif block_id == 5:
                img_configs[file_name]['rad_distort'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 6:
                img_configs[file_name]['tan_distort'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 7:
                img_configs[file_name]['cam_pos'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 8:
                cam_rot_line1 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 9:
                cam_rot_line2 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 0:
                cam_rot_line3 = np.fromstring(line, dtype=float, sep=' ')
                cam_rot = np.vstack([cam_rot_line1, cam_rot_line2, cam_rot_line3])
                img_configs[file_name]['cam_rot'] = cam_rot

    return img_configs