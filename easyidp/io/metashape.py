import os
import zipfile


def _split_project_path(path: str):
    """
    [inner_function] Get project name, current folder, extension, etc. from given project path.

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
    [inner function] Check if given path is metashape project structure

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


def _get_xml_str_from_zip_file(zip_file, xml_file):
    """
    [inner function] read xml file in zip file

    Parameters
    ----------
    zip_file: str
        the path to zip file, e.g. "data/metashape/goya_test.files/project.zip"
    xml_file: str
        the file name in zip file, e.g. "doc.xml" in previous zip file

    Returns
    -------
    xml_str: str
        the string of readed xml file
    """
    with zipfile.ZipFile(zip_file) as zfile:
        xml = zfile.open(xml_file)
        xml_str = xml.read().decode("utf-8")

    return xml_str


def _get_project_zip_xml(project_folder, project_name):
    """
    [inner function] read project.zip xml files to string
    project path = "/root/to/metashape/test_proj.psx"
    -->  project_folder = "/root/to/metashape/"
    -->  project_name = "test_proj"

    Parameters
    ----------
    project_folder: str
    project_name: str

    Returns
    -------
    xml_str: str
        the string of loaded "doc.xml"
    """
    zip_file = f"{project_folder}/{project_name}.files/project.zip"
    return _get_xml_str_from_zip_file(zip_file, "doc.xml")


def _get_chunk_zip_xml(project_folder, project_name, chunk_id):
    """
    [inner function] read chunk.zip xml file in given chunk
        project path = "/root/to/metashape/test_proj.psx"
    -->  project_folder = "/root/to/metashape/"
    -->  project_name = "test_proj"

    Parameters
    ----------
    project_folder: str
    project_name: str
    chunk_id: int or str
        the chunk id start from 0 of chunk.zip

    Returns
    -------
    xml_str: str
        the string of loaded "doc.xml"
    """
    zip_file = f"{project_folder}/{project_name}.files/{chunk_id}/chunk.zip"
    return _get_xml_str_from_zip_file(zip_file, "doc.xml")


def _get_frame_zip_xml(project_folder, project_name, chunk_id):
    """
    [inner function] read frame.zip xml file in given chunk
    project path = "/root/to/metashape/test_proj.psx"
    -->  project_folder = "/root/to/metashape/"
    -->  project_name = "test_proj"

    Parameters
    ----------
    project_folder: str
    project_name: str
    chunk_id: int or str
        the chunk id start from 0 of chunk.zip

    Returns
    -------
    xml_str: str
        the string of loaded "doc.xml"
    """
    zip_file = f"{project_folder}/{project_name}.files/{chunk_id}/0/frame.zip"
    return _get_xml_str_from_zip_file(zip_file, "doc.xml")


def open_project(path: str):
    """
    Read project data by given Metashape project path.

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------

    """
    if _check_is_software(path):
        folder_path, project_name, ext = _split_project_path(path)
        project_xml_str = _get_project_zip_xml(folder_path, project_name)
        # todo: the function to decode project_xml to know the number of chunks
        # chunk_xml_str = get_chunk_zip_xml(folder_path, project_name)