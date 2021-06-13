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
    p4d = {"param":None, "pcd":None, "dom":None, "dsm":None, "undist_raw":None, "project_name":None}

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
