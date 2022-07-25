import os
import sys
import pathlib
import zipfile
import gdown
import subprocess

def user_data_dir(file_name=""):
    r"""Get OS specific data directory path for SwagLyrics.
    Parameters
    ----------
    file_name : str
        file to be fetched from the data dir

    Returns
    -------
    str,
        full path to the user-specific data dir

    Notes
    -----
    Typical user data directories are:
        macOS:    ~/Library/Application Support/SwagLyrics
        Unix:     ~/.local/share/SwagLyrics   # or in $XDG_DATA_HOME, if defined
        Win 10:   C:\Users\<username>\AppData\Local\SwagLyrics
    For Unix, we follow the XDG spec and support $XDG_DATA_HOME if defined.

    Referenced from stackoverflow [1]_ then get github [2]_ .

    References
    ----------
    .. [1] Python: Getting AppData folder in a cross-platform way `url <https://stackoverflow.com/questions/19078969/python-getting-appdata-folder-in-a-cross-platform-way>`_
    .. [2] SwagLyrics-For-Spotify/swaglyrics/__init__.py `url <https://github.com/SwagLyrics/SwagLyrics-For-Spotify/blob/master/swaglyrics/__init__.py#L8-L32>`_
    """
    # get os specific path
    if sys.platform.startswith("win"):
        os_path = os.getenv("LOCALAPPDATA")
    elif sys.platform.startswith("darwin"):
        os_path = "~/Library/Application Support"
    else:
        # linux
        os_path = os.getenv("XDG_DATA_HOME", "~/.local/share")

    # join with SwagLyrics dir
    path = pathlib.Path(os_path) / "easyidp.data"

    return path.expanduser() / file_name

def show_data_dir():
    """open the cached data files in cross-platform [1]_ system default viewer.

    Reference
    ---------
    .. [1] Python: Opening a folder in Explorer/Nautilus/Finder `url <https://stackoverflow.com/questions/6631299/python-opening-a-folder-in-explorer-nautilus-finder>`_
    """
    path=user_data_dir()

    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

class EasyidpDataSet():

    def __init__(self, name="", url="", size=""):
        self.name = name
        self.url = url
        self.size = size
        self.data_dir = user_data_dir(self.name)
        self.zip_file = user_data_dir(self.name + ".zip")

        self.pix4d = self.ReconsProj()
        self.metashape = self.ReconsProj()

    def load_data(self):
        if not os.path.exists(self.data_dir):
            out = self.download_data()

            if os.path.exists(self.zip_file):
                self.unzip_data()
            else:
                raise FileNotFoundError(
                    f"Could not find the downloaded file [{self.zip_file}], "
                    f"please call ().load_data() to download again.\n"
                    f"Tips: ensure you can access Google Drive url."
                )
        
    def download_data(self):
        # Download; extract data to disk.
        # Raise an exception if the link is bad, or we can't connect, etc.
        output = gdown.download(url=self.url, output=str(self.zip_file), quiet=False, fuzzy=True)

        return output

    def unzip_data(self):
        with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        # already extracted
        if os.path.exists(self.data_dir):
            os.remove(self.zip_file)

    class ReconsProj():

        def __init__(self) -> None:
            self.proj = ""
            self.param = ""
            self.photo = ""
            self.dom = ""
            self.dsm = ""
            self.pcd = ""


class TanashiLotus2017(EasyidpDataSet):

    def __init__(self):
        url = "https://drive.google.com/file/d/1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j/view?usp=sharing"
        super().__init__(name="2017_tanashi_lotus", url=url, size="3.3GB")

        super().load_data()

        self.pix4d.proj = str(self.data_dir / "20170531")
        self.pix4d.param = str(self.data_dir / "20170531" / "params")
        self.pix4d.photo = str(self.data_dir / "20170531" / "photos")
        self.pix4d.dom = str(
            self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_transparent_mosaic_group1.tif"
        )
        self.pix4d.dsm = str(
            self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_dsm.tif"
        )
        self.pix4d.pcd = str(
            self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_group1_densified_point_cloud.ply"
        )

        self.metashape.proj = str(self.data_dir / "170531.Lotus.psx")
        self.metashape.param = str(self.data_dir / "170531.Lotus.files")
        self.metashape.photo = str(self.data_dir / "20170531" / "photos")
        self.metashape.dom = str(self.data_dir / "170531.Lotus.outputs" / "170531.Lotus_dom.tif")
        self.metashape.dsm = str(self.data_dir / "170531.Lotus.outputs" / "170531.Lotus_dsm.tif")
        self.metashape.dom = str(self.data_dir / "170531.Lotus.outputs" / "170531.Lotus.laz")

        
class GDownTest(EasyidpDataSet):

    def __init__(self):
        url = "https://drive.google.com/file/d/1yWvIOYJ1ML-UGleh3gT5b7dxXzBuSPgQ/view?usp=sharing"
        super().__init__("gdown_test", url, "0.2KB")
        super().load_data()

        self.pix4d.proj = str(self.data_dir / "file1.txt")
        self.metashape.param = str(self.data_dir / "folder1")




    