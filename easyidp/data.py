import os
import sys
import pathlib
import zipfile
import gdown
import requests
import subprocess
import webbrowser

def user_data_dir(file_name=""):
    r"""Get OS specific data directory path for EasyIDP.
    
    Parameters
    ----------
    file_name : str
        file to be fetched from the data dir

    Returns
    -------
    str
        full path to the user-specific data dir

    Notes
    -----
    Typical user data directories are:

    .. code-block:: text

        macOS:    ~/Library/Application Support/easyidp.data
        Unix:     ~/.local/share/easyidp.data   # or in $XDG_DATA_HOME, if defined
        Win 10:   C:\Users\<username>\AppData\Local\easyidp.data

    For Unix, we follow the XDG spec and support ``$XDG_DATA_HOME`` if defined.

    Referenced from stackoverflow [1]_ then get github [2]_ .

    References
    ----------
    .. [1] Python: Getting AppData folder in a cross-platform way https://stackoverflow.com/questions/19078969/python-getting-appdata-folder-in-a-cross-platform-way
    .. [2] SwagLyrics-For-Spotify/swaglyrics/__init__.py https://github.com/SwagLyrics/SwagLyrics-For-Spotify/blob/master/swaglyrics/__init__.py#L8-L32

    """
    # get os specific path
    if sys.platform.startswith("win"):
        os_path = os.getenv("LOCALAPPDATA")
    elif sys.platform.startswith("darwin"):
        os_path = "~/Library/Application Support"
    else:
        # linux
        os_path = os.getenv("XDG_DATA_HOME", "~/.local/share")

    # join with easyidp.data dir
    path = pathlib.Path(os_path) / "easyidp.data"

    return path.expanduser() / file_name

def show_data_dir():
    """open the cached data files in cross-platform system default viewer.

    It modified from this [1]_ webpage.

    References
    ----------
    .. [1] Python: Opening a folder in Explorer/Nautilus/Finder https://stackoverflow.com/questions/6631299/python-opening-a-folder-in-explorer-nautilus-finder

    """
    path = user_data_dir()

    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def url_checker(url):
    r"""Check if download url is accessable or not.
    
    Modified from this [1]_ link.

    Parameters
    ----------
    url : str
        The url need to be checked

    References
    ----------
    .. [1] https://pytutorial.com/check-url-is-reachable
    """
    try:
        #Get Url
        get = requests.get(url, timeout=3)
        # if the request succeeds 
        if get.status_code == 200:
            return True
        else:
            return False

    #Exception
    except requests.exceptions.RequestException as e:
        # print URL with Errs
        return False


class EasyidpDataSet():

    def __init__(self, name="", url_list=[], size="",):
        """The dataset has the following properties (almost in string type)

        name
            The dataset name
        url_list
            The possible download urls
        size
            Ths size of zipped file
        data_dir
            The final dataset directory
        zip_file
            The *temporary* downloaded zip file path (won't be used in)
        shp
            The path to plot ROI shapefile (\*.shp)
        photo
            The folder path to the raw photos

        pix4d.proj
            The pix4d project folder
        pix4d.param
            The parameter folder of pix4d porject
        pix4d.dom
            The generated DOM path of plot map
        pix4d.dsm
            The generated DSM path of plot map
        pix4d.pcd
            The generated pointcloud path of plot map

        metashape.proj
            The metashape project file
        metashape.param
            The parameter folder of metashape porject
        metashape.dom
            The generated DOM path of plot map
        metashape.dsm
            The generated DSM path of plot map
        metashape.pcd
            The generated pointcloud path of plot map
        

        """
        self.name = name
        self.url_list = url_list
        self.size = size
        self.data_dir = user_data_dir(self.name)
        self.zip_file = user_data_dir(self.name + ".zip")

        self.pix4d = self.ReconsProj()
        self.metashape = self.ReconsProj()

    def load_data(self):
        r"""Download dataset from Google Drive to user AppData folder

        Caution
        -------
        For users in China mainland, please either find a way to access google drive,
        or download from `CowTransfer Link <https://cowtransfer.com/s/4cc3d3f199824b>`_ 

        Save and extract all \*.zip file to folder ``idp.data.show_data_dir()``.

        The data structure should like this:

        
        .. tab:: Windows

            .. code-block:: text

                . C:\Users\<user>\AppData\Local\easyidp.data
                |-- 2017_tanashi_lotus
                |-- gdown_test
                |-- ...

        .. tab:: MacOS

            .. code-block:: text

                . ~/Library/Application Support/easyidp.data
                |-- 2017_tanashi_lotus
                |-- gdown_test
                |-- ...

        .. tab:: Linux/BSD

            .. code-block:: text

                . ~/.local/share/easyidp.data   # or in $XDG_DATA_HOME, if defined
                |-- 2017_tanashi_lotus
                |-- gdown_test
                |-- ...

        """

        if not os.path.exists(self.data_dir):
            out = self._download_data()

            if os.path.exists(self.zip_file):
                self._unzip_data()
            else:
                raise FileNotFoundError(
                    f"Could not find the downloaded file [{self.zip_file}], "
                    f"please call ().load_data() to download again.\n"
                    f"Tips: ensure you can access Google Drive url."
                )
        
    def _download_data(self):
        """using gdown to download dataset from Google Drive to user AppData folder
        """
        # Download; extract data to disk.
        # Raise an exception if the link is bad, or we can't connect, etc.

        if url_checker(self.url_list[0]):   # google drive 
            output = gdown.download(url=self.url_list[0], output=str(self.zip_file), quiet=False, fuzzy=True)
        elif url_checker(self.url_list[1]):  # cowtransfer
            print(
                f"Could not access to default google drive download link <{self.url_list[1]}>."
                f"Please download the file in browser and unzip to the popup folder "
                f"[{str(user_data_dir())}]"
            )
            # open url
            webbrowser.open(self.url_list[1], new=0, autoraise=True)
            # open folder in file explorer
            show_data_dir()
        else:
            raise ConnectionError("Could not find any downloadable link. Please contact the maintainer via github.")

        return output

    def _unzip_data(self):
        """Unzip downloaded zip data and remove after decompression
        """
        with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        # already extracted
        if os.path.exists(self.data_dir):
            os.remove(self.zip_file)
        else:
            raise FileNotFoundError("Seems fail to unzip, please check whether the zip file is fully downloaded.")

    class ReconsProj():


        def __init__(self) -> None:
            self.proj = ""
            self.param = ""
            self.dom = ""
            self.dsm = ""
            self.pcd = ""


class Lotus(EasyidpDataSet):

    def __init__(self):
        url_list = [
            "https://drive.google.com/file/d/1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j/view?usp=sharing",
            "https://cowtransfer.com/s/03355b0b684442"
        ]
        super().__init__(name="2017_tanashi_lotus", url_list=url_list, size="3.6GB")

        super().load_data()

        self.pix4d.photo = str(self.data_dir / "20170531" / "photos")
        self.shp = str(self.data_dir / "plots.shp")

        self.pix4d.proj = str(self.data_dir / "20170531")
        self.pix4d.param = str(self.data_dir / "20170531" / "params")
        
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
        self.metashape.dom = str(self.data_dir / "170531.Lotus.outputs" / "170531.Lotus_dom.tif")
        self.metashape.dsm = str(self.data_dir / "170531.Lotus.outputs" / "170531.Lotus_dsm.tif")
        self.metashape.pcd = str(self.data_dir / "170531.Lotus.outputs" / "170531.Lotus.laz")

    def load_data(self):
        return super().load_data()

        
class GDownTest(EasyidpDataSet):

    def __init__(self):
        url_list = [
            "https://drive.google.com/file/d/1yWvIOYJ1ML-UGleh3gT5b7dxXzBuSPgQ/view?usp=sharing",
            "https://cowtransfer.com/s/20fe3984fb9a47"
        ]
        super().__init__("gdown_test", url_list, "0.2KB")
        super().load_data()

        self.pix4d.proj = str(self.data_dir / "file1.txt")
        self.metashape.param = str(self.data_dir / "folder1")

    def load_data(self):
        return super().load_data()




    