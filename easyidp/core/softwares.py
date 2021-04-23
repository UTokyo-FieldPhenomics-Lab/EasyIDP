import os

class MetaShape:
    """Lite version of Metashape python API

    Notes
    -----
    This module is a lite version of the Metashape Python API which reads
    data directly from project xml files without Metashape Pro installed.
    This module is used to getting necessary data for reverse calculation.

    It equals to Metashape API -> Metashape.Document
    """

    def __init__(self, print_progress=False):
        self.print_progress = print_progress
        self.chunks = []
        self.meta = None
        self.path = None
        self.project_name = None

    def open(self, path: str):
        """
        Read project data by given Metashape project path.

        Parameters
        ----------
        path: str
            e.g. proj_path="/root/to/metashape/test_proj.psx"
        Returns
        -------

        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find Metashape project file [{path}]")

        folder_path, file_name = os.path.split(path)
        project_name, ext = os.path.splitext(file_name)

        self.path = path
        self.project_name = project_name

        data_folder = os.path.join(folder_path, project_name + ".files")
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Could not find Metashape project file [{data_folder}]")


class Pix4D:

    def __init__(self):
        pass