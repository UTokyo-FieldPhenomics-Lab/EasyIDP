import os


def _split_project_path(path: str):
    """
    Get project name, current folder, extension, etc. from given project path.

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------
    folder_path: str
        e.g. "/root/to/metashape/"
    project_name: str
        e.g. "test_proj"
    ext: str:
        e.g. "psx"
    """
    folder_path, file_name = os.path.split(path)
    project_name, ext = os.path.splitext(file_name)

    return folder_path, project_name, ext


def _check_is_software(path: str):
    """
    Check if given path is metashape project structure

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------
    judge: bool
        true: is metashape projects (has complete project structure)
        false: not a metashape projects (missing some files or path not found)
    """
    judge = False
    if not os.path.exists(path):
        print(f"Could not find Metashape project file [{path}]")
        return judge

    folder_path, project_name, ext = _split_project_path(path)
    data_folder = os.path.join(folder_path, project_name + ".files")

    if not os.path.exists(data_folder):
        print(f"Could not find Metashape project file [{data_folder}]")
        return judge

    judge = True
    return judge


def open_project_path(path: str):
    """
    Read project data by given Metashape project path.

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------

    """
    pass