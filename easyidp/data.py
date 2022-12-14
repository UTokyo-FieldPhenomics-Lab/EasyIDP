import os
import sys
import shutil
import zipfile
import gdown
import requests
import subprocess
import webbrowser
from pathlib import Path

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
    path = Path(os_path) / "easyidp.data"

    add_usr = path.expanduser()

    if not os.path.exists(str(add_usr)):
        os.mkdir(str(add_usr))

    return add_usr / file_name

def show_data_dir():
    """open the cached data files in cross-platform system default viewer.

    Notes
    -----
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


def download_all():
    """download all datasets
    """
    lotus = Lotus()
    gd = GDownTest()
    test = TestData()


class EasyidpDataSet():
    """The base class for Dataset
    """

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
        or download from `CowTransfer Link <https://fieldphenomics.cowtransfer.com/s/25f92eb0585b4d>`_ 

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
                print("Successfully downloaded, start unzipping ...")
                self._unzip_data()
                print("Successfully unzipped, the cache zip file has been removed.")
            else:
                raise FileNotFoundError(
                    f"Could not find the downloaded file [{self.zip_file}], "
                    f"please call ().load_data() to download again.\n"
                    f"Tips: ensure you can access Google Drive url."
                )

    def reload_data(self):
        """remove local data and redownload again
        """
        self.remove_data()
        self.load_data()

    def remove_data(self):
        """remove local cached data file
        """
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        
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
            self.project = ""
            self.param = ""
            self.dom = ""
            self.dsm = ""
            self.pcd = ""


class Lotus(EasyidpDataSet):
    """The dataset for lotus plot in Tanashi, Tokyo.

    .. image:: ../../_static/images/data/2017_tanashi_lotus.png 
        :width: 600
        :alt: 2017_tanashi_lotus.png 

    - **Crop** : lotus
    - **Location** : Tanashi, Nishi-Tokyo, Japan
    - **Flight date** : May 31, 2017
    - **UAV model** : DJI Inspire 1
    - **Flight height** : 30m
    - **Image number** :142
    - **Image size** : 4608 x 3456
    - **Software** : Pix4D, Metashape
    - **Outputs** : DOM, DSM, PCD
    """

    def __init__(self):
        """
        Containts the following arguments, you can access by:

        .. code-block:: python

            >>> lotus = idp.data.Lotus()
            >>> lotus.photo
            'C:\\Users\\<user>\\AppData\\Local\\easyidp.data\\2017_tanashi_lotus\\20170531\\photos'

        - ``.photo`` : the folder containts raw images
        - ``.shp`` : the plot roi shapefile
        - ``.pix4d.project`` : the pix4d project folder
        - ``.pix4d.param`` : the pix4d project parameter folder
        - ``.pix4d.dom`` : the pix4d produced orthomosaic
        - ``.pix4d.dsm`` : the pix4d produced digial surface del
        - ``.pix4d.pcd`` : the pix4d produced point cloud
        - ``.metashape.project`` : the metashape project file
        - ``.metashape.param`` : the metashape project folder
        - ``.metashape.dom`` : the metashape produced orthomosaic
        - ``.metashape.pcd`` : the metashape produced point cloud
        - ``.metashape.dsm`` : the metashape produced digial surface model

        See also
        --------
        EasyidpDataSet
        """

        url_list = [
            "https://drive.google.com/file/d/1SJmp-bG5SZrwdeJL-RnnljM2XmMNMF0j/view?usp=sharing",
            "https://fieldphenomics.cowtransfer.com/s/9a87698f8d3242"
        ]
        super().__init__(name="2017_tanashi_lotus", url_list=url_list, size="3.6GB")

        super().load_data()

        self.photo = self.data_dir / "20170531" / "photos"
        self.shp = self.data_dir / "plots.shp"

        self.pix4d.project = self.data_dir / "20170531"
        self.pix4d.param = self.data_dir / "20170531" / "params"
        
        self.pix4d.dom = self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_transparent_mosaic_group1.tif"
        self.pix4d.dsm = self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_dsm.tif"
        self.pix4d.pcd = self.data_dir / "20170531" / "hasu_tanashi_20170531_Ins1RGB_30m_group1_densified_point_cloud.ply"

        self.metashape.project = self.data_dir / "170531.Lotus.psx"
        self.metashape.param = self.data_dir / "170531.Lotus.files"
        self.metashape.dom = self.data_dir / "170531.Lotus.outputs" / "170531.Lotus_dom.tif"
        self.metashape.dsm = self.data_dir / "170531.Lotus.outputs" / "170531.Lotus_dsm.tif"
        self.metashape.pcd = self.data_dir / "170531.Lotus.outputs" / "170531.Lotus.laz"

        
class GDownTest(EasyidpDataSet):

    def __init__(self):
        url_list = [
            "https://drive.google.com/file/d/1yWvIOYJ1ML-UGleh3gT5b7dxXzBuSPgQ/view?usp=sharing",
            "https://fieldphenomics.cowtransfer.com/s/b5a469fab5dc48"
        ]
        super().__init__("gdown_test", url_list, "0.2KB")
        super().load_data()

        self.pix4d.proj = self.data_dir / "file1.txt"
        self.metashape.param = self.data_dir / "folder1"



class TestData(EasyidpDataSet):
    """The data for developer and package testing.
    """

    def __init__(self, test_out="./tests/out"):
        """
        Containts the following arguments, you can access by:
        
        **json test module** 

        * ``.json.for_read_json``
        * ``.json.labelme_demo``
        * ``.json.labelme_warn``
        * ``.json.labelme_err``

        **shp test module** 

        * ``.shp.lotus_shp`` 
        * ``.shp.lotus_prj`` 
        * ``.shp.complex_shp``
        * ``.shp.complex_prj``
        * ``.shp.lonlat_shp``
        * ``.shp.utm53n_shp``
        * ``.shp.utm53n_prj``
        * ``.shp.rice_shp``
        * ``.shp.rice_prj``
        * ``.shp.roi_shp``
        * ``.shp.roi_prj``
        * ``.shp.testutm_shp``
        * ``.shp.testutm_prj``

        **pcd test module** 

        * ``.pcd.lotus_las``
        * ``.pcd.lotus_laz``
        * ``.pcd.lotus_pcd``
        * ``.pcd.lotus_las13``
        * ``.pcd.lotus_laz13``
        * ``.pcd.lotus_ply_asc``
        * ``.pcd.lotus_ply_bin``
        * ``.pcd.maize_las``
        * ``.pcd.maize_laz``
        * ``.pcd.maize_ply``

        **roi test module** 

        * ``.roi.dxf``
        * ``.roi.lxyz_txt``
        * ``.roi.xyz_txt``

        **geotiff test module**

        * ``.tiff.soyweed_part``
        * ``.tiff.out``

        **metashape test module** 

        * ``.metashape.goya_psx``
        * ``.metashape.goya_param``
        * ``.metashape.lotus_psx``
        * ``.metashape.lotus_param``
        * ``.metashape.lotus_dsm``
        * ``.metashape.wheat_psx``
        * ``.metashape.wheat_param``

        **pix4d test module** 

        * ``.pix4d.lotus_folder``
        * ``.pix4d.lotus_param``
        * ``.pix4d.lotus_photos``
        * ``.pix4d.lotus_dom``
        * ``.pix4d.lotus_dsm``
        * ``.pix4d.lotus_pcd``
        * ``.pix4d.lotus_dom_part``
        * ``.pix4d.lotus_dsm_part``
        * ``.pix4d.lotus_pcd_part``
        * ``.pix4d.maize_folder``
        * ``.pix4d.maize_dom``
        * ``.pix4d.maize_dsm``
        * ``.pix4d.maize_noparam``
        * ``.pix4d.maize_empty``
        * ``.pix4d.maize_noout``

        **cvtools test module** 

        * ``.cv.out``
        
        **visualize test module** 

        * ``.vis.out``


        Parameters
        ----------
        test_out : str, optional
            The folder for saving temporary outputs, by default "./tests/out"

        See also
        --------
        EasyidpDataSet
        """
        url_list = [
            "https://drive.google.com/file/d/17b_17CofqIuCVOWMnD67_wOnWMtwF8bw/view?usp=sharing",
            "https://fieldphenomics.cowtransfer.com/s/edaf0826b02548"
        ]

        self.name = "data_for_tests"
        self.url_list = url_list
        self.size = "344MB"
        self.data_dir = user_data_dir(self.name)
        self.zip_file = user_data_dir(self.name + ".zip")

        super().load_data()

        self.json = self.JsonDataset(self.data_dir, test_out)
        self.metashape = self.MetashapeDataset(self.data_dir, test_out)
        self.pix4d = self.Pix4Dataset(self.data_dir)
        self.shp = self.ShapefileDataset(self.data_dir, test_out)
        self.pcd = self.PointCloudDataset(self.data_dir, test_out)
        self.roi = self.ROIDataset(self.data_dir)
        self.tiff = self.TiffDataSet(self.data_dir, test_out)

        self.cv = self.CVDataset(self.data_dir, test_out)
        self.vis = self.VisualDataset(self.data_dir, test_out)


    class MetashapeDataset():

        def __init__(self, data_dir, test_out):
            if isinstance(test_out, str):
                test_out = Path(test_out)

            self.goya_psx    = data_dir / "metashape" / "goya_test.psx"
            self.goya_param  = data_dir / "metashape" / "goya_test.files"

            self.lotus_psx   = data_dir / "metashape" / "Lotus.psx"
            self.lotus_param = data_dir / "metashape" / "Lotus.files"
            self.lotus_dsm   = data_dir / "metashape" / "Lotus.files" / "170531.Lotus_dsm.tif"

            self.wheat_psx   = data_dir / "metashape" / "wheat_tanashi.psx"
            self.wheat_param = data_dir / "metashape" / "wheat_tanashi.files"


    class Pix4Dataset():

        def __init__(self, data_dir):

            # here is reorgainzed pix4d project
            self.lotus_folder   = data_dir / "pix4d" / "lotus_tanashi_full"
            self.lotus_param    = self.lotus_folder / "params"
            self.lotus_photos   = self.lotus_folder / "photos"
            self.lotus_dom      = self.lotus_folder / "hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"
            self.lotus_dsm      = self.lotus_folder / "hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"
            self.lotus_pcd      = self.lotus_folder / "hasu_tanashi_20170525_Ins1RGB_30m_group1_densified_point_cloud.ply"
            self.lotus_dom_part = self.lotus_folder / "plot_dom.tif"
            self.lotus_dsm_part = self.lotus_folder / "plot_dsm.tif"
            self.lotus_pcd_part = self.lotus_folder / "plot_pcd.ply"

            # here is standard pix4d project
            self.maize_folder  = data_dir / "pix4d" / "maize_tanashi" / "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"
            self.maize_dom = self.maize_folder / "3_dsm_ortho" / "2_mosaic" / \
                "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_transparent_mosaic_group1.tif"
            self.maize_dsm = self.maize_folder / "3_dsm_ortho" / "1_dsm" / \
                "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_dsm.tif"

            self.maize_noparam = data_dir / "pix4d" / "maize_tanashi" / "maize_tanashi_no_param"
            self.maize_empty   = data_dir / "pix4d" / "maize_tanashi" / "maize_tanashi_raname_empty_test"
            self.maize_noout   = data_dir / "pix4d" / "maize_tanashi" / "maize_tanashi_raname_no_outputs"

    class JsonDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir
            self.for_read_json = data_dir / "json_test" / "for_read_json.json"
            self.labelme_demo  = data_dir / "json_test" / "labelme_demo_img.json"
            self.labelme_warn  = data_dir / "json_test" / "labelme_warn_img.json"
            self.labelme_err   = data_dir / "json_test" / "for_read_json.json"

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "json_test"

        def __truediv__(self, other):
            return self.data_dir / "json_test" / other


    class ShapefileDataset():

        def __init__(self, data_dir, test_out):

            self.data_dir = data_dir
            self.lotus_shp = data_dir / "shp_test" / "lotus_plots.shp"
            self.lotus_prj = data_dir / "shp_test" / "lotus_plots.prj"

            self.complex_shp = data_dir / "shp_test" / "complex_shp_review.shp"
            self.complex_prj = data_dir / "shp_test" / "complex_shp_review.prj"

            self.lonlat_shp = data_dir / "shp_test" / "lon_lat.shp"

            self.utm53n_shp = data_dir / "shp_test" / "lon_lat_utm53n.shp"
            self.utm53n_prj = data_dir / "shp_test" / "lon_lat_utm53n.prj"

            self.rice_shp = data_dir / "shp_test" / "rice_ind_duplicate.shp"
            self.rice_prj = data_dir / "shp_test" / "rice_ind_duplicate.prj"

            self.roi_shp = data_dir / "shp_test" / "roi.shp"
            self.roi_prj = data_dir / "shp_test" / "roi.prj"

            self.testutm_shp = data_dir / "shp_test" / "test_utm.shp"
            self.testutm_prj = data_dir / "shp_test" / "test_utm.prj"

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "shp_test"

        def __truediv__(self, other):
            return self.data_dir / "shp_test" / other


    class PointCloudDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            self.lotus_las = data_dir / "pcd_test" / "hasu_tanashi.las"
            self.lotus_laz = data_dir / "pcd_test" / "hasu_tanashi.laz"
            self.lotus_pcd = data_dir / "pcd_test" / "hasu_tanashi.pcd"

            self.lotus_las13 = data_dir / "pcd_test" / "hasu_tanashi_1.3.las"
            self.lotus_laz13 = data_dir / "pcd_test" / "hasu_tanashi_1.3.laz"

            self.lotus_ply_asc = data_dir / "pcd_test" / "hasu_tanashi_ascii.ply"
            self.lotus_ply_bin = data_dir / "pcd_test" / "hasu_tanashi_binary.ply"

            self.maize_las = data_dir / "pcd_test" / "maize3na_20210614_15m_utm.las"
            self.maize_laz = data_dir / "pcd_test" / "maize3na_20210614_15m_utm.laz"
            self.maize_ply = data_dir / "pcd_test" / "maize3na_20210614_15m_utm.ply"

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "pcd_test"

        def __truediv__(self, other):
            return self.data_dir / "pcd_test" / other


    class ROIDataset():

        def __init__(self, data_dir):
            self.data_dir = data_dir

            self.dxf  = data_dir / "roi_test" / "hasu_tanashi_ccroi.dxf"
            self.lxyz_txt = data_dir / "roi_test" / "hasu_tanashi_lxyz.txt"
            self.xyz_txt  = data_dir / "roi_test" / "hasu_tanashi_xyz.txt"


    class TiffDataSet():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            self.soyweed_part = data_dir / "tiff_test" / "2_12.tif"

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "tiff_test"

        def __truediv__(self, other):
            return self.data_dir / "tiff_test" / other


    class CVDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "cv_test"

        def __truediv__(self, other):
            return self.data_dir / "cv_test" / other


    class VisualDataset():

        def __init__(self, data_dir, test_out):
            self.data_dir = data_dir

            if isinstance(test_out, str):
                test_out = Path(test_out)
            self.out = test_out / "visual_test"

        def __truediv__(self, other):
            return self.data_dir / "visual_test" / other
