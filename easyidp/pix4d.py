import os
import numpy as np
import warnings

####################
# code for file IO #
####################

def _match_suffix(folder, ext):
    """
    find the *first* file by given suffix, e.g. *.tif
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


def parse_p4d_param_folder(param_path:str):
    param_dict = {}
    # keys = ["project_name", "xyz", "pmat", "cicp", "ccp"]

    param_files = os.listdir(param_path)

    if len(param_files) < 4:
        raise FileNotFoundError(
            f"Given param folder [{_get_full_path(param_path)}] "
            "does not have enough param files to parse"
        )

    project_name = os.path.commonprefix(param_files)
    # > "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_"
    if project_name[-1] == '_':
        project_name = project_name[:-1]
    param_dict["project_name"] = project_name

    xyz_file = f"{param_path}/{project_name}_offset.xyz"
    if os.path.exists(xyz_file):
        param_dict['xyz'] = xyz_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{xyz_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path or "
            "`project_name` is correct."
        )

    pmat_file = f"{param_path}/{project_name}_pmatrix.txt"
    if os.path.exists(pmat_file):
        param_dict['pmat'] = pmat_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{pmat_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path or "
            "`project_name` is correct."
        )

    cicp_file = f"{param_path}/{project_name}_pix4d_calibrated_internal_camera_parameters.cam"
    # two files with the same string
    # {project_name}_      calibrated_internal_camera_parameters.cam
    # {project_name}_pix4d_calibrated_internal_camera_parameters.cam
    if os.path.exists(cicp_file):
        param_dict['cicp'] = cicp_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{cicp_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path or "
            "`project_name` is correct."
        )

    ccp_file = f"{param_path}/{project_name}_calibrated_camera_parameters.txt"
    if os.path.exists(ccp_file):
        param_dict['ccp'] = ccp_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{ccp_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path or "
            "`project_name` is correct."
        )

    campos_file = f"{param_path}/{project_name}_calibrated_images_position.txt"
    if os.path.exists(campos_file):
        param_dict['campos'] = campos_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{campos_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path or "
            "`project_name` is correct."
        )

    return param_dict


def parse_p4d_project(project_path:str, param_folder=None):
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

    param_folder: str, default None
        if not given, it will parse as a standard pix4d project, and trying
            to get the project name from `1_initial/param` folder
        if it is not a standard pix4d project (re-orgainzed folder), need manual
            specify the path to param folder, in order to parse project_name
            for later usage.

    Returns
    -------
    p4d: dict
        a python dictionary that contains the path to each file.
        {
            "project_name": the prefix of whole project file.
            "param": the folder of parameters
            "pcd": the point cloud file
            "dom": the digital orthomosaic file
            "dsm": the digital surface model file
            "undist_raw": the undistorted images corrected by the pix4d software (when original image unable to find)
        }

    Notes
    -----
    Project_name can be extracted from parameter folder prefix in easyidp 2.0, no need manual specify.
    To find the outputs, it will pick the first file that fits the expected file format.
    """
    p4d = {"param": None, "pcd": None, "dom": None, "dsm": None, "undist_raw": None, "project_name": None}

    project_path = _get_full_path(project_path)
    sub_folder = os.listdir(project_path)

    #################################
    # parse defalt 1_initial folder #
    #################################
    if param_folder is None:
        param_folder = f"{project_path}/1_initial/params"

    undist_folder = f"{project_path}/1_initial/images/undistorted_images"

    # check whether a correct pix4d project folder
    if '1_initial' not in sub_folder and param_folder is None:
        raise FileNotFoundError(f"Current folder [{project_path}] is not a standard pix4d projects folder, please manual speccify `param_folder`")

    if os.path.exists(param_folder):
        param = parse_p4d_param_folder(param_folder)
        p4d["param"] = param
        project_name = param["project_name"]
        p4d["project_name"] = param["project_name"]
    else:
        raise FileNotFoundError(
            "Can not find pix4d parameter in given project folder"
        )

    if os.path.exists(undist_folder):
        p4d["undist_raw"] = undist_folder

    ######################
    # parse output files #
    ######################

    # point cloud file
    pcd_folder = f"{project_path}/2_densification/point_cloud"
    ply_file = f"{pcd_folder}/{project_name}_group1_densified_point_cloud.ply"
    laz_file = f"{pcd_folder}/{project_name}_group1_densified_point_cloud.laz"
    las_file = f"{pcd_folder}/{project_name}_group1_densified_point_cloud.las"

    if os.path.exists(pcd_folder):
        if os.path.exists(ply_file):
            p4d["pcd"] = ply_file
        elif os.path.exists(las_file):
            p4d["pcd"] = las_file
        elif os.path.exists(laz_file):
            p4d["pcd"] = laz_file
        else:
            force = _match_suffix(pcd_folder, ["ply", "las", "laz"])
            if force is not None:
                p4d["pcd"] = force
            else:
                warnings.warn(
                    f"Unable to find any point cloud output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{pcd_folder}]. Please specify manually."
                )

    # digital surface model DSM file
    dsm_folder = f"{project_path}/3_dsm_ortho/1_dsm"
    dsm_file = f"{dsm_folder}/{project_name}_dsm.tif"
    if os.path.exists(dsm_folder):
        if os.path.exists(dsm_file):
            p4d["dsm"] = dsm_file
        else:
            force = _match_suffix(dsm_folder, "tif")
            if force is not None:
                p4d["dsm"] = force
            else:
                warnings.warn(
                    f"Unable to find any DSM output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{dense_folder}]. Please specify manually."
                )

    dom_folder = f"{project_path}/3_dsm_ortho/2_mosaic"
    dom_file = f"{dom_folder}/{project_name}_transparent_mosaic_group1.tif"
    if os.path.exists(dom_folder):
        if os.path.exists(dom_file):
            p4d["dom"] = dom_file
        else:
            force = _match_suffix(dom_folder, "tif")
            if force is not None:
                p4d["dom"] = force
            else:
                warnings.warn(
                    f"Unable to find any DOM output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{dense_folder}]. Please specify manually."
                )

    return p4d


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
    """Read PROJECTNAME_pix4d_calibrated_internal_camera_parameters.cam file

    Parameters
    ----------
    cicp_path: str

    Returns
    -------
    cicp_dict: dict

    Notes
    -----
    It is the info about sensor
    maize_cicp.txt:
        Pix4D camera calibration file 0
        #Focal Length mm assuming a sensor width of 17.49998592000000030566x13.12498944000000200560mm
        F 15.01175404934517487732
        #Principal Point mm
        Px 8.48210511970419922534
        Py 6.33434629978042273990
        #Symmetrical Lens Distortion Coeffs
        K1 0.03833474118270804865
        K2 -0.01750917966495743258
        K3 0.02049798716391852335
        #Tangential Lens Distortion Coeffs
        T1 0.00240851666319534747
        T2 0.00292562392135245920
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
                # one example:
                # > Focal Length mm assuming a sensor width of 12.82x8.55mm\n
                w_h = sp_list[8].split('x')
                cam_dict['w_mm'] = float(w_h[0])  # extract e.g. 12.82
                cam_dict['h_mm'] = float(w_h[1][:-4])  # extract e.g. 8.55
    return cam_dict


def read_ccp(ccp_path):
    """Read PROJECTNAME_calibrated_camera_parameters.txt

    Parameters
    ----------
    ccp_path: str

    Returns
    -------
    img_configs: dict
        {'Image1.JPG': 
            {'cam_matrix': array([[...]]), 
             'rad_distort': array([ 0.03833474, ...02049799]),
             'tan_distort': array([0.00240852, 0...00292562]), 
             'cam_pos': array([ 21.54872207,...8570281 ]), 
             'cam_rot': array([[ 0.78389904,...99236  ]])}, 
             'w': 4608, 'h': 3456, 
         'Image2.JPG':
            ...
        }

    Notes
    -----
    It is the camera position info
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


def read_campos_geo(campos_path):
    """Read PROJECTNAME_calibrated_images_position.txt

    Parameters
    ----------
    campos_path : str

    Notes
    -----
    the geo position of each camera
        DJI_0954.JPG,368030.548722,3955824.412658,127.857028
        DJI_0955.JPG,368031.004387,3955824.824967,127.381322
        DJI_0956.JPG,368033.252520,3955826.479610,127.080709
        DJI_0957.JPG,368032.022104,3955826.060493,126.715974
        DJI_0958.JPG,368031.901165,3955826.109158,126.666393
        DJI_0959.JPG,368030.686490,3955830.981070,127.327741

    Returns
    -------
    campos_dict:
        {"Image1.JPG": np.array([x, y ,z]), 
         "Image2.JPG": ...
         ...}
    """
    with open(campos_path, 'r') as f:
        cam_dict = {}
        for line in f.readlines():
            sp_list = line.split(',')
            if len(sp_list) == 4: 
                cam_dict[sp_list[0]] = np.array(sp_list[1:], dtype=np.float)

    return cam_dict


################################
# code for reverse calculation #
################################