import os
import pyproj
import warnings
import numpy as np
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from pathlib import Path

import easyidp as idp


class ROI(idp.Container):
    """
    A Region of Interest (ROI) object, can be either 2D or 3D, often read from shp file.
    """

    def __init__(self, target_path=None, **kwargs):
        """The method to initialize the ROI class

        Parameters
        ----------
        target_path : str | pathlib.Path, optional
            the path to roi file, often is shp file path, by default None

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            [shp][proj] Use projection [WGS 84] for loaded shapefile [lotus_plots.shp]
            Read shapefile [lotus_plots.shp]: 100%|███████████| 112/112 [00:00<00:00, 2559.70it/s]

            >>> roi
            <easyidp.ROI> with 112 items
            [0]     N1W1
            array([[139.54052962,  35.73475194],
                   [139.54055106,  35.73475596],
                   [139.54055592,  35.73473843],
                   [139.54053438,  35.73473446],
                   [139.54052962,  35.73475194]])
            [1]     N1W2
            array([[139.54053488,  35.73473289],
                   [139.54055632,  35.73473691],
                   [139.54056118,  35.73471937],
                   [139.54053963,  35.73471541],
                   [139.54053488,  35.73473289]])
            ...
            [110]   S4E6
            array([[139.54090456,  35.73453742],
                   [139.540926  ,  35.73454144],
                   [139.54093086,  35.7345239 ],
                   [139.54090932,  35.73451994],
                   [139.54090456,  35.73453742]])
            [111]   S4E7
            array([[139.54090986,  35.73451856],
                   [139.54093129,  35.73452258],
                   [139.54093616,  35.73450504],
                   [139.54091461,  35.73450107],
                   [139.54090986,  35.73451856]])
        """
        super().__init__()
        # in super
        # self.id_item = {}
        # self.item_label = {}
        # if has CRS -> GPS coordiantes -> geo2pix convert

        #: the CRS that current ROI used.
        self.crs = None   # default -> pixel coords
        #: the source file path of current ROI.
        self.source = target_path

        if target_path is not None:
            self.open(target_path, **kwargs)


    def is_geo(self):
        """Returns True if the ROI is geo coordinate.

        Returns
        -------
        bool
        """
        if self.crs is None:
            return False
        else:
            return True


    def open(self, target_path, **kwargs):
        """An advanced wrapper to open ROI without dealing with file format, current support shapefile.shp and labelme.json

        Parameters
        ----------
        target_path : str
            the path to roi files, current support shapefile.shp and labelme.json

        Example
        -------

        Initialize an empty object:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> roi = idp.ROI()

        Then you can open a ROI by:

        .. code-block:: python

            >>> roi.read_shp(test_data.shp.lotus_shp, name_field=0)

            >>> roi.read_labelme_json(test_data.json.labelme_demo)

        Or using this short function:

        .. code-block:: python

            >>> roi.open(test_data.shp.lotus_shp, name_field=0)

            >>> roi.open(test_data.json.labelme_demo)

        Notes
        -----
        You can also pass several control parameters in this function, please refer to :func:`read_shp` and :func:`read_labelme_json` for more information

        """
        ext = os.path.splitext(target_path)[-1]
        if ext == ".shp":
            self.read_shp(target_path, **kwargs)
        elif ext == ".json":
            self.read_labelme_json(target_path)


    def read_shp(self, shp_path, shp_proj=None, name_field=None, include_title=False, encoding='utf-8'):
        """read ROI from shp file

        Parameters
        ----------
        shp_path : str
            the file path of \*.shp
        shp_proj : str | pyproj object
            | by default None, will read automatically from prj file with the same name of shp filename, 
            | or give manually by ``read_shp(..., shp_proj=pyproj.CRS.from_epsg(4326), ...)`` or 
            | ``read_shp(..., shp_proj=r'path/to/{shp_name}.prj', ...)`` 
        name_field : str or int or list[ str|int ], optional
            by default None, the id or name of shp file fields as output dictionary keys
        include_title : bool, optional
            by default False, whether add column name to roi key.
        encoding : str
            by default 'utf-8', for some chinese characters, 'gbk' may required

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> roi = idp.ROI()

        Then you can open a ROI by:

        .. code-block:: python

            >>> roi.read_shp(test_data.shp.lotus_shp, name_field=0)
            [shp][proj] Use projection [WGS 84] for loaded shapefile [lotus_plots.shp]
            Read shapefile [lotus_plots.shp]: 100%|███████████| 112/112 [00:00<00:00, 2559.70it/s]
            >>> roi
            <easyidp.ROI> with 112 items
            [0]     N1W1
            array([[139.54052962,  35.73475194],
                   [139.54055106,  35.73475596],
                   [139.54055592,  35.73473843],
                   [139.54053438,  35.73473446],
                   [139.54052962,  35.73475194]])
            [1]     N1W2
            array([[139.54053488,  35.73473289],
                   [139.54055632,  35.73473691],
                   [139.54056118,  35.73471937],
                   [139.54053963,  35.73471541],
                   [139.54053488,  35.73473289]])
            ...
            [110]   S4E6
            array([[139.54090456,  35.73453742],
                   [139.540926  ,  35.73454144],
                   [139.54093086,  35.7345239 ],
                   [139.54090932,  35.73451994],
                   [139.54090456,  35.73453742]])
            [111]   S4E7
            array([[139.54090986,  35.73451856],
                   [139.54093129,  35.73452258],
                   [139.54093616,  35.73450504],
                   [139.54091461,  35.73450107],
                   [139.54090986,  35.73451856]])

        Notes
        -----
        For more details of these parameters, please refer to :func:`easyidp.shp.read_shp`

        See also
        --------
        easyidp.shp.read_shp

        """
        # if geotiff_proj is not None and shp_proj is not None and shp_proj.name != geotiff_proj.name:
        # shp.convert_proj()
        roi_dict, crs = idp.shp.read_shp(shp_path, shp_proj, name_field, include_title, encoding, return_proj=True)

        self.source = shp_path

        self.crs = crs
        self.id_item = {}
        self.item_label = {}

        for k, v in roi_dict.items():
            self[k] = v

    def read_labelme_json(self, json_path):
        """read roi from labelme marked json file

        Parameters
        ----------
        json_path : str
            the path to labelme json file.

        Example
        -------
        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> json_path = test_data.json.labelme_demo
            PosixPath('/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/json_test/labelme_demo_img.json')

            >>> roi = idp.ROI(json_path)
            >>> roi
            <easyidp.ROI> with 1 items
            [0]     1
            array([[2447.2392638 , 1369.32515337],
                   [2469.93865031, 1628.2208589 ],
                   [2730.06134969, 1605.52147239],
                   [2703.06748466, 1348.46625767]])

        """
        js_dict = idp.jsonfile.read_json(json_path)

        # check if is labelme json
        if all(x in js_dict.keys() for x in ["version", "flags", "shapes", "imagePath", "imageHeight"]):

            # init values
            self.crs = None
            self.id_item = {}
            self.item_label = {}
            self.source = json_path

            for shapes in js_dict["shapes"]:
                if shapes["shape_type"] == "polygon":
                    label = shapes["label"]
                    poly = shapes["points"]

                    self[label] = np.array(poly)
                else:
                    warnings.warn(
                        f"Only labelme [polygon] shape are accepted, not [{shapes['shape_type']}] of [{shapes['label']}]")
        else:
            raise TypeError(f"It seems [{json_path}] is not a Labelme json file.")
        
    def read_geojson(self, geojson_path, name_field=None, include_title=False):
        """read ROI from geojson file

        Parameters
        ----------
        geojson_path : str
            the file path of \*.geojson
        name_field : str or int or list[ str|int ], optional
            by default None, the id or name of shp file fields as output dictionary keys
        include_title : bool, optional
            by default False, whether add column name to roi key.

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> roi = idp.ROI()

        Then you can open a ROI by:

        .. code-block:: python

            >>> roi.read_geojson(test_data.json.geojson_soy, name_field="FID")
            Read geojson [2023_soybean_field.geojson]: 100%|███████████| 260/260         [00:00<00:00, 218234.75it/s]
        
            >>> roi
            <easyidp.ROI> with 260 items
            [0]	65
            array([[-26384.952573, -28870.678514],
                   [-26384.269447, -28870.522501],
                   [-26385.160022, -28866.622912],
                   [-26385.843163, -28866.778928],
                   [-26384.952573, -28870.678514]])
            [1]	97
            array([[-26386.065868, -28865.804036],
                   [-26385.382668, -28865.648006],
                   [-26386.273244, -28861.748416],
                   [-26386.956458, -28861.90445 ],
                   [-26386.065868, -28865.804036]])
            ...
            [258]	4
            array([[-26404.447166, -28860.770249],
                   [-26405.337854, -28856.870669],
                   [-26406.020223, -28857.026509],
                   [-26405.129644, -28860.926114],
                   [-26404.447166, -28860.770249]])
            [259]	1
            array([[-26393.693576, -28844.979604],
                   [-26394.58426 , -28841.08004 ],
                   [-26395.26665 , -28841.235885],
                   [-26394.375966, -28845.135449],
                   [-26393.693576, -28844.979604]])
        
        Notes
        -----
        For more details of these parameters, please refer to :func:`easyidp.jsonfile.read_geojson`

        See also
        --------
        easyidp.jsonfile.show_geojson_fields
        """
        pass

        geojson_dict, crs_proj = idp.jsonfile.read_geojson(
            geojson_path, name_field, include_title, return_proj=True)

        self.source = geojson_path

        self.crs = crs_proj
        self.id_item = {}
        self.item_label = {}

        for k, v in geojson_dict.items():
            self[k] = v


    def change_crs(self, target_crs):
        """Change the geo coordinates of roi to another crs.

        Parameters
        ----------
        target_crs : pyproj.CRS
            the CRS want convert to.

        Example
        -------
        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

        Read roi with lon and lat CRS (WGS84)

        .. code-block:: python

            >>> roi = idp.ROI(test_data.shp.lotus_shp)
            [shp][proj] Use projection [WGS 84] for loaded shapefile [lotus_plots.shp]
            >>> roi.crs
            <Geographic 2D CRS: EPSG:4326>
            Name: WGS 84
            Axis Info [ellipsoidal]:
            - Lat[north]: Geodetic latitude (degree)
            - Lon[east]: Geodetic longitude (degree)
            Area of Use:
            - name: World.
            - bounds: (-180.0, -90.0, 180.0, 90.0)
            Datum: World Geodetic System 1984 ensemble
            - Ellipsoid: WGS 84
            - Prime Meridian: Greenwich

        Check the roi coordinates

        .. code-block:: python

            >>> roi[0]
            array([[139.54052962,  35.73475194],
                   [139.54055106,  35.73475596],
                   [139.54055592,  35.73473843],
                   [139.54053438,  35.73473446],
                   [139.54052962,  35.73475194]])

        Read a geotiff with different CRS (UTM 54N)

        .. code-block:: python

            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> target_crs = dom.header["crs"]
            >>> target_crs
            <Derived Projected CRS: EPSG:32654>
            Name: WGS 84 / UTM zone 54N
            Axis Info [cartesian]:
            - E[east]: Easting (metre)
            - N[north]: Northing (metre)
            Area of Use:
            - name: Between 138°E and 144°E, northern hemisphere between equator and 84°N, onshore and offshore. Japan. Russian Federation.
            - bounds: (138.0, 0.0, 144.0, 84.0)
            Coordinate Operation:
            - name: UTM zone 54N
            - method: Transverse Mercator
            Datum: World Geodetic System 1984 ensemble
            - Ellipsoid: WGS 84
            - Prime Meridian: Greenwich

        Change the roi crs (coordiante) from WGS84 to UTM 54N

        .. code-block:: python

            >>> roi.change_crs(target_crs)
            >>> roi.crs
            <Derived Projected CRS: EPSG:32654>
            Name: WGS 84 / UTM zone 54N
            Axis Info [cartesian]:
            - E[east]: Easting (metre)
            - N[north]: Northing (metre)
            Area of Use:
            - name: Between 138°E and 144°E, northern hemisphere between equator and 84°N, onshore and offshore. Japan. Russian Federation.
            - bounds: (138.0, 0.0, 144.0, 84.0)
            Coordinate Operation:
            - name: UTM zone 54N
            - method: Transverse Mercator
            Datum: World Geodetic System 1984 ensemble
            - Ellipsoid: WGS 84
            - Prime Meridian: Greenwich

        Check the converted coordiante values

        .. code-block:: python

            >>> roi[0]
            array([[ 368017.7565143 , 3955511.08102276],
                   [ 368019.70190232, 3955511.49811902],
                   [ 368020.11263046, 3955509.54636219],
                   [ 368018.15769062, 3955509.13563382],
                   [ 368017.7565143 , 3955511.08102276]])
        """
        if self.crs is None:
            raise FileNotFoundError(
                "Current ROI does not have CRS, can not convert "
                "(Is it a pixel coordinate?)")

        if  not isinstance(self.crs, pyproj.CRS) or \
            not isinstance(target_crs, pyproj.CRS):
            raise TypeError(
                f"Both self.crs <{type(self.crs)}> and target_crs "
                f"<{type(target_crs)}> should be <pyproj.CRS> type"
            )

        self.id_item = idp.geotools.convert_proj(self.id_item, self.crs, target_crs)
        self.crs = target_crs

    def _get_z_input_check(self, obj, mode, kernel, buffer, func="dsm"):
        # check if has CRS (GeoROI), otherwise stop
        if not self.is_geo():
            raise TypeError("Could not operate without CRS specified")

        # check input type
        if mode not in ["point", "face"]:
            raise KeyError(
                f"The param 'mode' only accept 'point' or 'face', not '{mode}'"
            )

        if kernel not in [
            "mean", "min", "max", "pmin5", "pmin10", "pmax5", "pmax10"
        ]:
            raise KeyError(f"The param 'kernel' only accept "
                f"'mean', 'min', 'max', 'pmin5', 'pmin10', 'pmax5', 'pmax10'"
                f" not '{kernel}'"
            )

        if not isinstance(buffer, (int, float)):
            raise TypeError(
                f"Only 'int' and 'float' are acceptable for 'buffer', not "
                f"{type(buffer)} [{buffer}]."
            )

        # convert input objects
        if isinstance(obj, (Path, str)):
            if not Path(obj).exists():
                raise FileNotFoundError(f"Could not find file {obj}.")

            if func == "dsm":
                return idp.GeoTiff(obj)
            else:
                return idp.PointCloud(obj)
        elif isinstance(obj, (idp.GeoTiff, idp.PointCloud)):
            return obj
        else:
            if func == "dsm":
                raise TypeError(
                    f"Only geotiff path <str> and <easyidp.GeoTiff> object"
                    f"are accepted, not {type(obj)}")
            else:
                raise TypeError(
                    f"Only geotiff path <str> and <easyidp.PointCloud> object"
                    f"are accepted, not {type(obj)}")


    def get_z_from_dsm(self, dsm, mode="face", kernel="mean", buffer=0, keep_crs=False):
        """Get the z values (heights) from DSM for 2D polygon

        Parameters
        ----------
        dsm : str | <GeoTiff> object
            the path of dsm, or the GeoTiff object from idp.GeoTiff()
        mode : str, optional
            the mode to calculate z values, by default "face".
            
            - ``point``: get height on each vertex, result in different values for each vertex
            - ``face``: get height on polygon face, result in the same value for each vertex

        kernel : str, optional
            The math kernel to calculate the z value, by default 'mean'

            - ``mean``: the mean value inside polygon
            - ``min``: the minimum value inside polygon
            - ``max``: the maximum value inside polygon
            - ``pmin5``: 5th *percentile mean* inside polygon
            - ``pmin10``: 10th *percentile mean* inside polygon
            - ``pmax5``: 95th *percentile mean* inside polygon
            - ``pmax10``: 90th *percentile mean* inside polygon
            
            .. note::
            
                percentile mean: the mean value of all pixels over/under xth percentile threshold

                .. image:: ../../_static/images/python_api/percentile_mean.png
                    :alt: percentile_mean.png'
                    :scale: 35
        
        buffer : float, optional
            | the buffer of ROI, by default 0 (no buffer),

            - ``0``: not using buffer
            - ``-1``: ignore given polygon, using the full dsm to calculate the height
            - ``float``: buffer distance, the unit of buffer follows the DSM coordinates, either pixel or meter.

            .. note::

                If ``mode="point"`` , will generate a round buffer polygon first, then extract the z-value by this region, but the return will only be a single point.

                If ``mode="face"``, will buffer the polygon and then calculate the height inside the buffered polygon

                .. image:: ../../_static/images/python_api/roi_crop_mode.png
                    :alt: roi_crop_mode.png'
                    :scale: 30

        keep_crs : bool, optional
            When the crs is not the save with DSM crs, where change the ROI crs to fit DSM.

            - ``False`` (default): change ROI's CRS;
            - ``True``: not change ROI's CRS, only attach the z value to current coordinate. 

        Example
        -------

        Data prepare:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> roi = roi[0:3]
            >>> lotus_full_dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)

        .. caution:: 
        
            The ROI and DSM, did not share the same CRS.

        The ROI is in longitude-latitude coordinate system, unit is degree.

        .. code-block:: python

            >>> roi.crs
            <Geographic 2D CRS: EPSG:4326>
            Name: WGS 84
            ...

            >>> roi
            [0]     N1W1
            array([[139.54052962,  35.73475194],
                   [139.54055106,  35.73475596],
                   [139.54055592,  35.73473843],
                   [139.54053438,  35.73473446],
                   [139.54052962,  35.73475194]])
            [1]     N1W2
            array([[139.54053488,  35.73473289],
                   [139.54055632,  35.73473691],
                   [139.54056118,  35.73471937],
                   [139.54053963,  35.73471541],
                   [139.54053488,  35.73473289]])
            [2]     N1W3
            array([[139.54054017,  35.73471392],
                   [139.54056161,  35.73471794],
                   [139.54056647,  35.73470041],
                   [139.54054493,  35.73469644],
                   [139.54054017,  35.73471392]])

        While the DSM (and DOM) are in UTM zone 54N, unit is meter.

        .. code-block:: python

            >>> lotus_full_dsm.crs
            <Derived Projected CRS: EPSG:32654>
            Name: WGS 84 / UTM zone 54N
            ...

            >>> >>> lotus_full_dsm.header['tie_point']
            [368014.54157, 3955518.2747700005]


        .. admonition:: Different mode examples
            :class: important

            The point mode, each point has its unique z value.

            .. code-block:: python

                >>> roi_temp = roi.copy()
                >>> roi_temp.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=0, keep_crs=False)

                >>> roi_temp
                <easyidp ROI> with 3 items
                [0]     N1W1
                array([[ 368017.7565143 , 3955511.08102276,      97.63990021],
                       [ 368019.70190232, 3955511.49811902,      97.67140198],
                       [ 368020.11263046, 3955509.54636219,      97.75421143],
                       [ 368018.15769062, 3955509.13563382,      97.65288544],
                       [ 368017.7565143 , 3955511.08102276,      97.63990021]])
                [1]     N1W2
                array([[ 368018.20042946, 3955508.96051697,      97.65105438],
                       [ 368020.14581791, 3955509.37761334,      97.65817261],
                       [ 368020.55654627, 3955507.42585654,      97.63339996],
                       [ 368018.601606  , 3955507.01512806,      97.61153412],
                       [ 368018.20042946, 3955508.96051697,      97.65105438]])
                [2]     N1W3
                array([[ 368018.64801755, 3955506.84956301,      97.59950256],
                       [ 368020.59340644, 3955507.26665948,      97.64406586],
                       [ 368021.00413502, 3955505.31490271,      97.64678192],
                       [ 368019.04919431, 3955504.90417413,      97.63285828],
                       [ 368018.64801755, 3955506.84956301,      97.59950256]])

            The face mode, all points of one ROI share the same z value.

            .. code-block:: python

                >>> roi_temp = roi.copy()
                >>> roi_temp.get_z_from_dsm(lotus_full_dsm, mode="face", kernel="mean", buffer=0, keep_crs=False)

                >>> roi_temp
                <easyidp ROI> with 3 items
                [0]     N1W1
                array([[ 368017.7565143 , 3955511.08102276,      97.68352509],
                       [ 368019.70190232, 3955511.49811902,      97.68352509],
                       [ 368020.11263046, 3955509.54636219,      97.68352509],
                       [ 368018.15769062, 3955509.13563382,      97.68352509],
                       [ 368017.7565143 , 3955511.08102276,      97.68352509]])
                [1]     N1W2
                array([[ 368018.20042946, 3955508.96051697,      97.59811401],
                       [ 368020.14581791, 3955509.37761334,      97.59811401],
                       [ 368020.55654627, 3955507.42585654,      97.59811401],
                       [ 368018.601606  , 3955507.01512806,      97.59811401],
                       [ 368018.20042946, 3955508.96051697,      97.59811401]])
                [2]     N1W3
                array([[ 368018.64801755, 3955506.84956301,      97.6997757 ],
                       [ 368020.59340644, 3955507.26665948,      97.6997757 ],
                       [ 368021.00413502, 3955505.31490271,      97.6997757 ],
                       [ 368019.04919431, 3955504.90417413,      97.6997757 ],
                       [ 368018.64801755, 3955506.84956301,      97.6997757 ]])

        .. admonition:: Setting buffer
            :class: seealso

            You can using buffer to calculate z values from a larger area. This will decrease the effects of some extreme noise points on DSM. Especially for the point mode, which is more sensitive to such noise.

            .. caution:: 
            
                The value here share the same unit as DSM, if your DSM in lon-lat coordinate (e.g. WGS84, EPSG:4326), ``buffer=1.0`` will result in 1.0 degree in longitude and latitude, this is a very large area!

            .. code-block:: python

                >>> roi_temp = roi.copy()
                >>> roi_temp.get_z_from_dsm(lotus_full_dsm, mode="face", kernel="mean", buffer=1.0, keep_crs=False)

                >>> roi_temp
                <easyidp ROI> with 3 items
                [0]     N1W1
                array([[ 368017.7565143 , 3955511.08102276,      98.30323792],
                       [ 368019.70190232, 3955511.49811902,      98.30323792],
                       [ 368020.11263046, 3955509.54636219,      98.30323792],
                       [ 368018.15769062, 3955509.13563382,      98.30323792],
                       [ 368017.7565143 , 3955511.08102276,      98.30323792]])
                [1]     N1W2
                array([[ 368018.20042946, 3955508.96051697,      97.6088028 ],
                       [ 368020.14581791, 3955509.37761334,      97.6088028 ],
                       [ 368020.55654627, 3955507.42585654,      97.6088028 ],
                       [ 368018.601606  , 3955507.01512806,      97.6088028 ],
                       [ 368018.20042946, 3955508.96051697,      97.6088028 ]])
                [2]     N1W3
                array([[ 368018.64801755, 3955506.84956301,      97.5995636 ],
                       [ 368020.59340644, 3955507.26665948,      97.5995636 ],
                       [ 368021.00413502, 3955505.31490271,      97.5995636 ],
                       [ 368019.04919431, 3955504.90417413,      97.5995636 ],

        .. admonition:: keep_crs option
            :class: tip

            If not keep CRS, the ROI x and y values will also change to the same coordinate with DSM.

            If do not want the value change, please setting ``keep_crs=True``

            .. code-block:: python

                >>> roi_temp = roi.copy()
                >>> roi_temp.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=0, keep_crs=True)

                >>> roi_temp
                <easyidp ROI> with 3 items
                [0]     N1W1
                array([[139.54052962,  35.73475194,  97.63990021],
                       [139.54055106,  35.73475596,  97.67140198],
                       [139.54055592,  35.73473843,  97.75421143],
                       [139.54053438,  35.73473446,  97.65288544],
                       [139.54052962,  35.73475194,  97.63990021]])
                [1]     N1W2
                array([[139.54053488,  35.73473289,  97.65105438],
                       [139.54055632,  35.73473691,  97.65817261],
                       [139.54056118,  35.73471937,  97.63339996],
                       [139.54053963,  35.73471541,  97.61153412],
                       [139.54053488,  35.73473289,  97.65105438]])
                [2]     N1W3
                array([[139.54054017,  35.73471392,  97.59950256],
                       [139.54056161,  35.73471794,  97.64406586],
                       [139.54056647,  35.73470041,  97.64678192],
                       [139.54054493,  35.73469644,  97.63285828],
                       [139.54054017,  35.73471392,  97.59950256]])


        See also
        --------
        :func:`easyidp.GeoTiff.polygon_math <easyidp.geotiff.GeoTiff.polygon_math>`

        """
        dsm = self._get_z_input_check(dsm, mode, kernel, buffer, func="dsm")

        # check is the dsm, not RGB or multiband GeoTiff.
        if dsm.header["dim"] != 1:
            raise TypeError(
                f"Only one layer geotiff (DSM) are accepted, current "
                f"layer is {dsm.header['dim']}")

        # using the full map to calculate
        if buffer == -1 or buffer == -1.0:
            global_z = dsm.polygon_math(polygon_hv="full_map", kernel=kernel)
        else:
            global_z = None

        # convert CRS if necessary
        if self.crs.name == dsm.header["crs"].name:
            poly_dict = self.id_item.copy()
        elif self.crs.name != dsm.header["crs"].name and not keep_crs:
            self.change_crs(dsm.header["crs"])
            poly_dict = self.id_item.copy()
        else:
            poly_dict = idp.geotools.convert_proj(self.id_item, self.crs, dsm.header["crs"])

        nan_z_list = []
        pbar = tqdm(self.items(), desc=f"Read z values of roi from DSM [{dsm.file_path.name}]")
        for roi_name, val in pbar:
            poly = poly_dict[self.item_label[roi_name]]
            # only get the x and y of coords
            poly = poly[:, 0:2]

            # using the full map
            if global_z is not None:
                poly3d = _insert_z_value_for_roi(val, global_z)
            else:
                if mode == "face":    # using the polygon as uniform z values
                    # need do buffer
                    if buffer != 0 or buffer != 0.0:
                        p = Polygon(poly)
                        p_buffer = p.buffer(buffer)
                        poly = np.array(p_buffer.exterior.coords)

                    poly_z = dsm.polygon_math(poly, is_geo=True, kernel=kernel)
                    
                    poly3d = _insert_z_value_for_roi(val, poly_z)

                else:    # using each point own z values
                    if buffer != 0 or buffer != 0.0:
                        z_values = []
                        for po in poly:
                            p = Point(po)
                            p_buffer = p.buffer(buffer)
                            p_buffer_np = np.array(p_buffer.exterior.coords)

                            poly_z = dsm.polygon_math(p_buffer_np, is_geo=True, kernel=kernel)
                            z_values.append(poly_z)

                        z_values = np.array(z_values)
                    else:
                        # just qurey pixel value 
                        z_values = dsm.point_query(poly, is_geo=True)

                    poly3d = _insert_z_value_for_roi(val, z_values)

            # give warning if np.nan in z values (often caused by ROI outside DSM)
            if(np.isin(poly3d, dsm.header['nodata']).any()):
                nan_z_list.append(roi_name)

            self[roi_name] = poly3d

        if len(nan_z_list) > 0:
            warnings.warn(f"Z values contains empty attribute [{dsm.header['nodata']}] for {nan_z_list}, this may be caused by the ROI distribute inside the DSM no-value area, please double check the source shapefile and DOM in GIS software")


    def get_z_from_pcd(self, pcd, mode="face", kernel="mean", buffer=0):
        """Get the z values (heights) from Point cloud for 2D polygon

        .. attention:: This function has not been implemented.

        See also
        --------
        get_z_from_dsm
        """
        # if mode = point, buffer > 0, otherwise raise error
        pcd = self._get_z_input_check(pcd, mode, kernel, buffer, func="pcd")
        raise NotImplementedError("Will be implemented in the future.")

    def crop(self, target, save_folder=None):
        """Crop several ROIs from the geotiff by given <ROI> object with several polygons and polygon names

        Parameters
        ----------
        target : str | <GeoTiff> object
            the path of dsm, or the GeoTiff object from idp.GeoTiff()
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_folder : str, optional
            the folder to save cropped images, use ROI indices as file_names, by default None, means not save.

        Returns
        -------
        dict,
            The dictionary with key=id and value=ndarray data

        Example
        -------

        Data prepare:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> roi = idp.ROI(test_data.shp.lotus_shp , name_field=0)
            >>> roi = roi[0:3]
            >>> roi.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=0, keep_crs=False)

        Then crop the given DOM, DSM and PointCloud:

        .. code-block:: python

            >>> lotus_full_dsm = test_data.pix4d.lotus_dsm
            >>> lotus_full_pcd = test_data.pix4d.lotus_pcd 
            >>> lotus_full_dom = test_data.pix4d.lotus_dom 

            >>> out_dom = roi.crop(lotus_full_dom)
            >>> out_dsm = roi.crop(lotus_full_dsm)
            >>> out_pcd = roi.crop(lotus_full_pcd)

            >>> out_dsm
            {'N1W1': 
            array([[-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   ...,
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.]], dtype=float32), 
            'N1W2': 
            array([[-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   ...,
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.]], dtype=float32), 
            'N1W3': 
            array([[-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   ...,
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.],
                   [-10000., -10000., -10000., ..., -10000., -10000., -10000.]], dtype=float32)}

        Or you can specify the ``save_folder`` parameter to automatically save the cropped results

        .. code-block:: python

            >>> out_dom = roi.crop(lotus_full_dom, save_folder=r"path/to/save/outputs")

        See also
        --------
        :func:`easyidp.GeoTiff.crop_rois <easyidp.geotiff.GeoTiff.crop_rois>`

        """
        if not self.is_geo():
            raise TypeError("Could not operate without CRS specified")
            
        if isinstance(target, (Path, str)) and Path(target).exists():
            ext = Path(target).suffix
            if ext == ".tif":
                target = idp.GeoTiff(target)
            elif ext in [".ply", ".laz", ".las"]:
                target = idp.PointCloud(target)
            else:
                raise TypeError(f"Only [.tif, .ply, .laz, .las] are supported, not [{ext}]")
        elif isinstance(target, (idp.GeoTiff, idp.PointCloud)):
            pass
        else:
            raise TypeError(
                f"Only file path <str> or <easyidp.GeoTiff> object or <easyidp.PointCloud> object "
                f"are accepted, not {type(target)}"
            )

        if isinstance(target, idp.GeoTiff):
            out = target.crop_rois(self, is_geo=True, save_folder=save_folder)
        elif isinstance(target, idp.PointCloud):
            out = target.crop_rois(self, save_folder=save_folder)

        return out

    def back2raw(self, recons, **kwargs):
        """Projects several GIS coordintates ROIs (polygons) to all images

        Parameters
        ----------
        roi : easyidp.ROI | dict
            the <ROI> object created by easyidp.ROI() or dictionary
        save_folder : str, optional
            the folder to save projected preview images and json files, by default ""
        distortion_correct : bool, optional
            | Whether do distortion correction, by default True (back to raw image with lens distortion);
            | If back to software corrected images without len distortion, set it to False. 
            | (Pix4D support do this operation, seems metashape not supported yet.)
        ignore : str | None, optional
            Whether tolerate small parts outside image, check :func:`easyidp.reconstruct.Sensor.in_img_boundary` for more details.

            - ``None``: strickly in image area;
            - ``x``: only y (vertical) in image area, x can outside image;
            - ``y``: only x (horizontal) in image area, y can outside image.

        log : bool, optional
            whether print log for debugging, by default False

        Returns
        -------
        out_dict : dict
            The 2-layer dictionary, the first layer is the roi id, the second layer is the image names contains each roi ( ``out_dict['roi_id']['image_name']`` )

            .. code-block:: python

                >>> out_dict = roi.back2raw(...)
                # find all available images with specified roi (plot), e.g. roi named 'N1W1':
                >>> one_roi_dict = out_dict['N1W1']  # roi id
                {'IMG_3457': ..., 'IMG_3458': ..., ...}
                # find the sepecific roi on specific image, e.g. roi named 'N1W1' on image 'IMG_3457':
                >>> one_roi_one_img_coord = out_dict["N1W1']['IMG_3457']
                array([[  43.9388228 , 1247.0474214 ],
                       [  69.04076173,  972.90860296],
                       [ 353.26968458,  993.31308291],
                       [ 328.12327606, 1267.41006845],
                       [  43.9388228 , 1247.0474214 ]])

            .. caution::
                It is recommended to use dict.items() for iteration.

                .. code-block:: python

                    for roi_id, img_dict in out_dict.items():
                        # roi_id = 'N1W1'
                        # img_dict = out_dict[roi_id]
                        for img_name, coords in img_dict.items():
                            # img_name = "IMG_3457"
                            # coords = out_dict[roi_id][img_name]
                            print(coords)

                Not recommended to use in this way:

                .. code-block:: python
                
                    for roi_id in out_dict.keys()
                        img_dict = out_dict[roi_id]
                        for img_name in img_dict.keys():
                            coords = out_dict[roi_id][img_name]
                            print(coords)

        Example
        -------

        Data prepare:

        .. code-block:: python

            >>> import easyidp as idp
            >>> lotus = idp.data.Lotus()

            >>> p4d = idp.Pix4D(lotus.pix4d.project, lotus.photo, lotus.pix4d.param)

            >>> ms = idp.Metashape(project_path=lotus.metashape.project, chunk_id=0)

            >>> roi = idp.ROI(lotus.shp, name_field=0)
            >>> roi = roi[0:3]
            >>> roi.get_z_from_dsm(lotus.pix4d.dsm)

        Then do the backward calculation

        .. code-block:: python

            >>> out_p4d = roi.back2raw(p4d)

            >>> out_ms = roi.back2raw(ms)
            {'N1W1': 
                {'DJI_0479': array([[  43.91987253, 1247.04066872],
                                    [  69.0221046 ,  972.89938018],
                                    [ 353.25370817,  993.30409359],
                                    [ 328.10701394, 1267.40353364],
                                    [  43.91987253, 1247.04066872]]), 
                 'DJI_0480': array([[ 655.3678591 , 1273.01418098],
                                    [ 681.18303761,  996.4866665 ],
                                    [ 965.60719523, 1019.55346144],
                                    [ 939.89408896, 1296.05588162],
                                    [ 655.3678591 , 1273.01418098]]), 
                 'DJI_0481': array([[1024.43757205, 1442.10211955],
                                    [1043.51451272, 1159.41597   ],
                                    [1331.67724595, 1177.40543929],
                                    [1312.55275279, 1460.0493473 ],
                                    [1024.43757205, 1442.10211955]]), 
                 ...
                }

             'N1W2': 
                {...}
                
             ...
                
        See also
        --------
        easyidp.pix4d.back2raw, easyidp.metashape.back2raw

        """
        # call related function
        # need check alt exists, the alt is changing for different dsm?
        if not self.is_geo():
            raise TypeError("Could not operate without CRS specified")

        # check if is 3D roi
        # fix bugs #47
        for k, coord in self.items():
            dim = coord.shape[1]
            if dim != 3:
                raise ValueError(f"The back2raw function requires 3D roi with shape=(n, 3), but [{k}] is {coord.shape}")

        # if is one chunk
        if isinstance(recons, (idp.Pix4D, idp.Metashape)):
            out_dict = recons.back2raw(self, **kwargs)
        # several chunks
        # Todo, not supported
        if isinstance(recons, idp.ProjectPool):
            raise NotImplementedError("This Pool batch processing function has not been fully implemented")
            out_dict = {}
            for chunk in recons:
                if isinstance(save_folder, str) and os.path.isdir(save_folder):
                    save_path = os.path.join(save_folder, chunk.label)
                else:
                    save_path = None

                out_dict[chunk.label] = chunk.back2raw(self, save_folder=save_path, **kwargs)

        return out_dict


def _insert_z_value_for_roi(ndarray, z_value):
    # given a value list
    if isinstance(z_value, np.ndarray):
        # fix bug #72, the input z_value is already nx1 2d array
        if len(z_value.shape) == 1:
            # convert [1,2,3,4] -> [[1],[2],[3],[4]]
            z_value_2d = z_value[:, None]
        elif len(z_value.shape) == 2:
            z_value_2d = z_value
        else:
            raise ValueError(f"The expected z_value shape should be either (n) or (n, 1), not given {z_value.shape}")

        if ndarray.shape[1] == 2:
            ndarray_z = np.concatenate([ndarray, z_value_2d], axis=1)
        elif ndarray.shape[1] == 3:
            ndarray_z = np.concatenate([ndarray[:, 0:2], z_value_2d], axis=1)
        else:
            raise ValueError(f"The expected ROI shape should be (n, 3), not given {ndarray.shape}")
    else:  # share the same value
        if ndarray.shape[1] == 2:
            ndarray_z = np.insert(ndarray, obj=2, values=z_value, axis=1)
        elif ndarray.shape[1] == 3:
            ndarray_z = np.insert(ndarray[:,0:2], obj=2, values=z_value, axis=1)
        else:
            raise ValueError(f"The expected ROI shape should be (n, 3), not given {ndarray.shape}")

    return ndarray_z


def read_cc_txt(txt_path):
    """Read the point cloud annotation made by cloudcompare

    Parameters
    ----------
    txt_path : str
        The path to cloudcompare annotation txt file

    Note
    ----
    Please refer :ref:`make-roi-on-point-cloud` to know how to prepare txt roi annotation in CloudCompare for point cloud.
    """
    if os.path.exists(txt_path):
        try:
            # try to analysis 'x,y,z' type 
            test_data = np.loadtxt(txt_path, delimiter=',')
        except ValueError as e:
            # means it is 'label, x, y, z' type
            # line i -> Point #0, x, y, z
            # ValueError: could not convert string to float: 'Point '
            # it also view # as comment sign.
            test_data = np.loadtxt(txt_path, delimiter=',', comments="@", usecols=[1,2,3])
    else:
        raise FileNotFoundError(f"Could not find file [{txt_path}]")

    # get points A,B,C,D, but polygon needs A,B,C,D,A
    poly_data = np.append(test_data, test_data[0,:][None,:], axis = 0)

    return poly_data


# Load detections for backwards projection 

def load_detections(path):
    """Load a csv file of bounding box detections
    CSV takes the format xmin, ymin, xmax, ymax, image_path, label. The bounding box corners are in the image coordinate system,
    the image_path is the expected to be the full path, and the label is the character label for each box. One detection per row in the csv.
    
    Args:
        path: path on local disk
    Returns:
        boxes: a pandas dataframe of detections
    """
    
    # boxes = pd.read_csv(path)
    boxes = None
    if not all([x in ["image_path","xmin","ymin","xmax","ymax","image_path","label"] for x in boxes.columns]):
        raise IOError("{} is expected to be a .csv with columns, xmin, ymin, xmax, ymax, image_path, label for each detection")
        
    return None