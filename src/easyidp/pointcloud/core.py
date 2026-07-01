import os
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
from ..logger import logger

import numpy as np
import pyproj
from scipy.spatial import cKDTree

from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon

import easyidp as idp
from .geometry import query_indices_by_polygon, query_indices_by_multipolygon
from .compat import PointCloudCompatMixin


class PointCloud(PointCloudCompatMixin):
    """EasyIDP defined PointCloud class, consists by point coordinates, and optionally point colors and point normals."""

    def __init__(self, pcd_path="", offset=[0.0, 0.0, 0.0]):
        """The method to initialize the PointCloud class

        Parameters
        ----------
        pcd_path : str, optional
            The point cloud file path for loading/reading, by default "", means create an empty point cloud class
        offset : list, optional
            This parameter is used to specify your own offsets rather than the automatically calculated one.

            .. note::

                When the point cloud xyz value is too large, need to deduct duplicate values (minus offsets) to save the memory cost and increase the precision.

            .. caution::

                For some Pix4D produced pointcloud, the point cloud itself has been offseted, need manually add the offset value back.

        Example
        -------

        **Prepare**

        Cancel the numpy scientific counting method display:

        .. code-block:: python

            >>> import numpy as np
            >>> np.set_printoptions(suppress=True)

        Package loading:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

        **Read large xyz point cloud**

        Most point cloud use the CRS (GPS) coordianate as xyz values directly.

        .. code-block:: python

            >>> pcd = idp.PointCloud(test_data.pcd.maize_las)
            >>> pcd.points
            array([[ 367993.0206, 3955865.095 ,      57.9707],
                   [ 367993.146 , 3955865.3131,      57.9703],
                   [ 367992.6317, 3955867.2979,      57.9822],
                   ...,
                   [ 368014.7912, 3955879.4943,      58.0219],
                   [ 368014.1528, 3955883.5785,      58.0321],
                   [ 368016.7278, 3955874.1188,      57.9668]])

        If store these values directly, will cost a lot of memeory with precision loss. But with offsets, the data can be stored more neatly in the EasyIDP:

        .. code-block:: python

            >>> pcd.offset
            array([ 367900., 3955800.,       0.])
            >>> pcd._points
            array([[ 93.0206,  65.095 ,  57.9707],
                   [ 93.146 ,  65.3131,  57.9703],
                   [ 92.6317,  67.2979,  57.9822],
                   ...,
                   [114.7912,  79.4943,  58.0219],
                   [114.1528,  83.5785,  58.0321],
                   [116.7278,  74.1188,  57.9668]])

        **Manually specify offset**

        The previous offset is calculated automatically by EasyIDP, you can also manually specify the offset values:

        .. code-block:: python

            >>> pcd = idp.PointCloud(test_data.pcd.maize_las, offset=[367800, 3955700, 50])
            >>> pcd.offset
            array([ 367800., 3955700.,       50.])
            >>> pcd._points
            array([[193.0206, 165.095 ,   7.9707],
                   [193.146 , 165.3131,   7.9703],
                   [192.6317, 167.2979,   7.9822],
                   ...,
                   [214.7912, 179.4943,   8.0219],
                   [214.1528, 183.5785,   8.0321],
                   [216.7278, 174.1188,   7.9668]])

        Though the inner stored values changed, it does not affect the final point valus:

        .. code-block:: python

            >>> pcd.points
            array([[ 367993.0206, 3955865.095 ,      57.9707],
                   [ 367993.146 , 3955865.3131,      57.9703],
                   [ 367992.6317, 3955867.2979,      57.9822],
                   ...,
                   [ 368014.7912, 3955879.4943,      58.0219],
                   [ 368014.1528, 3955883.5785,      58.0321],
                   [ 368016.7278, 3955874.1188,      57.9668]])

        **Read Pix4D offseted point cloud and add offset back**

        If you read the Pix4D produced point cloud directly:

        .. code-block:: python

            >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
            >>> pcd
                         x        y        z  r    g    b        nx      ny      nz
                0  -18.908  -15.778   -0.779  123  103  79   nodata  nodata  nodata
                1  -18.908  -15.777   -0.78   124  104  81   nodata  nodata  nodata
                2  -18.907  -15.775   -0.802  123  103  80   nodata  nodata  nodata
              ...  ...      ...      ...      ...  ...  ...     ...     ...     ...
            42451  -15.789  -17.961   -0.847  116  98   80   nodata  nodata  nodata
            42452  -15.789  -17.939   -0.84   113  95   76   nodata  nodata  nodata
            42453  -15.786  -17.937   -0.833  115  97   78   nodata  nodata  nodata

        Here the xyz seems not the correct one, when we check the Pix4D project ``{name}_offset.xyz`` file in the param folders, we can find the offset values stored by Pix4D.

        .. code-block:: python

            >>> with open(test_data.pix4d.lotus_param / "hasu_tanashi_20170525_Ins1RGB_30m_offset.xyz", 'r') as f:
            ...     f.readlines()
            ['368043.000 3955495.000 98.000']

        This often requires user manually add that offset back to point cloud. But EasyIDP supports dealing with such situation easily:

        .. code-block:: python

            >>> p4d_offset_np = np.array([368043, 3955495,  98])
            >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin, p4d_offset_np)
            >>> pcd
                            x            y        z  r    g    b        nx      ny      nz
                0  368024.092  3955479.222   97.221  123  103  79   nodata  nodata  nodata
                1  368024.092  3955479.223   97.22   124  104  81   nodata  nodata  nodata
                2  368024.093  3955479.225   97.198  123  103  80   nodata  nodata  nodata
              ...     ...          ...      ...      ...  ...  ...     ...     ...     ...
            42451  368027.211  3955477.039   97.153  116  98   80   nodata  nodata  nodata
            42452  368027.211  3955477.061   97.16   113  95   76   nodata  nodata  nodata
            42453  368027.214  3955477.063   97.167  115  97   78   nodata  nodata  nodata

        .. note::

            You can also obtain the ``p4d_offset_np`` by :class:`easyidp.Pix4D <easyidp.pix4d.Pix4D>` object:

            .. code-block:: python

                >>> p4d = idp.Pix4D(project_path   = test_data.pix4d.lotus_folder,
                ...                 raw_img_folder = test_data.pix4d.lotus_photos,
                ...                 param_folder   = test_data.pix4d.lotus_param))
                >>> p4d.offset_np
                array([ 368043., 3955495.,      98.])

            And feed it to the previous function:

            .. code-block:: python

                >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin, p4d.offset_np)
        """
        #: the file path to the current point cloud file
        self.file_path = pcd_path
        #: the file extension to the current point cloud file
        self.file_ext = ".ply"

        self._points = None  # internal points with offsets to save memory
        #: The color (RGB) values of point cloud
        self.colors = None
        #: The normal vector values of point cloud
        self.normals = None
        #: The size of point cloud (xyz)
        self.shape = (0, 3)
        #: The CRS of point cloud
        self._crs = None
        #: The KDTree of point cloud
        self._tree = None

        self.offset = self._offset_type_check(offset)
        # BeatTiFul print strings for calling print() function
        self._btf_print = "<Empty easyidp.PointCloud object>"

        if pcd_path != "":
            self.read_point_cloud(pcd_path)

    def __str__(self) -> str:
        return self._btf_print

    def __repr__(self) -> str:
        return self._btf_print

    def __len__(self) -> int:
        return self.shape[0]

    def _update_btf_print(self):
        """Print Point Cloud in "DataFrame" beautiful way
        >>> print(pcd)
             x    y    z  r       g       b           nx      ny      nz
        0    1    2    3  nodata  nodata  nodata  nodata  nodata  nodata
        1    4    5    6  nodata  nodata  nodata  nodata  nodata  nodata
        2    7    8    9  nodata  nodata  nodata  nodata  nodata  nodata
        """
        head = ["", "x", "y", "z", "r", "g", "b", "nx", "ny", "nz"]
        data = []
        col_align = ["right"] + ["decimal"] * 3 + ["left"] * 3 + ["decimal"] * 3

        if self.shape[0] > 6:
            show_idx = [0, 1, 2, -3, -2, -1]
        else:
            show_idx = list(range(self.shape[0]))

        for i in show_idx:
            if self.has_points():
                xyz = np.around(self.points[i, :], decimals=3).tolist()
            else:
                xyz = ["nodata"] * 3

            if self.has_colors():
                rgb = self.colors[i, :].tolist()
            else:
                rgb = ["nodata"] * 3

            if self.has_normals():
                nxyz = self.normals[i, :].tolist()
            else:
                nxyz = ["nodata"] * 3

            if i >= 0:
                data.append([i] + xyz + rgb + nxyz)
            if i < 0:
                data.append([self.shape[0] + i] + xyz + rgb + nxyz)

        if self.shape[0] > 6:
            data.insert(3, ["..."] * 10)

        self._btf_print = tabulate(
            data, headers=head, tablefmt="plain", colalign=col_align
        )

    @property
    def points(self):
        """The xyz values of point cloud"""
        if self._points is None:
            return None
        else:
            return self._points + self._offset

    @points.setter
    def points(self, p):
        if not isinstance(p, np.ndarray):
            raise TypeError(
                "Only numpy ndarray object are acceptable for setting values"
            )
        elif self.shape != p.shape and self.shape != (0, 3):
            raise IndexError(
                f"The given shape [{p.shape}] does not match current point cloud shape [{self.shape}]"
            )
        else:
            self._points = p - self._offset
            self.shape = p.shape
            self._tree = None  # clear tree cache
            self._update_btf_print()

    @property
    def crs(self):
        """The Coordinate Reference System (CRS) of point cloud"""
        return self._crs

    @crs.setter
    def crs(self, c):
        if c is None:
            self._crs = None
        elif isinstance(c, pyproj.CRS):
            self._crs = c
        else:
            try:
                self._crs = pyproj.CRS.from_user_input(c)
            except pyproj.exceptions.CRSError:
                raise TypeError(
                    f"Only pyproj.CRS object or valid CRS string/int are acceptable, not {type(c)} [{c}]"
                )

    @property
    def tree(self):
        """The 2D KDTree of point cloud for fast spatial query"""
        if self._tree is None:
            if self.has_points():
                # self.points is property, will calculated with offset, it is slow
                # using self._points + self._offset to avoid data copy?
                # cKDTree need data copy? -> yes, it seems
                # build on 2D
                self._tree = cKDTree(self._points_xy)
            else:
                return None
        return self._tree

    @property
    def _points_xy(self):
        """Absolute XY coordinates without allocating the Z column.

        Returns ``self._points[:, :2] + self._offset[:2]`` which
        allocates only an (N, 2) array instead of the full (N, 3)
        produced by the public ``points`` property.  Used in KDTree-
        based crop hot paths where Z is not needed for the query.
        """
        return self._points[:, :2] + self._offset[:2]

    def select_by_index(self, indices, invert=False):
        """Select points by index and return a new PointCloud.

        Parameters
        ----------
        indices : ndarray of int
            Indices of points to select.
        invert : bool, optional
            If True, select all points NOT in ``indices``. Default False.

        Returns
        -------
        PointCloud
            A new PointCloud containing only the selected points.
            Empty selection produces an empty PointCloud with CRS and
            offset preserved.

        Examples
        --------
        >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        >>> sub = pcd.select_by_index(np.array([0, 5, 10]))
        >>> sub.shape
        (3, 3)
        """
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices, dtype=int)
        else:
            indices = indices.astype(int, copy=False)

        if invert:
            all_idx = np.arange(self.shape[0])
            indices = np.setdiff1d(all_idx, indices)

        result = PointCloud()
        result._offset = self._offset.copy()

        if self._crs is not None:
            result._crs = self._crs

        if len(indices) > 0 and self._points is not None:
            result._points = self._points[indices].copy()
            result.shape = result._points.shape
            if self.colors is not None:
                result.colors = self.colors[indices].copy()
            if self.normals is not None:
                result.normals = self.normals[indices].copy()
            result._update_btf_print()
        else:
            result._points = None
            result.colors = None
            result.normals = None
            result.shape = (0, 3)

        return result

    def crop(self, geometry):
        """Crop point cloud by a 2D geometry.

        Parameters
        ----------
        geometry : shapely.geometry.Polygon, shapely.geometry.MultiPolygon, \
                or easyidp.ROI
            The 2D crop geometry.
            - ``Polygon`` / ``MultiPolygon``: return a single PointCloud.
            - ``ROI``: return a dict of ``{label: PointCloud}``.

        Returns
        -------
        PointCloud or dict of {str: PointCloud}
            Cropped point cloud(s).

        Raises
        ------
        TypeError
            If ``geometry`` is an unsupported type, with migration guidance.

        Examples
        --------
        Crop by a Shapely polygon:

        >>> from shapely.geometry import Polygon
        >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        >>> polygon = Polygon([
        ...     [-18.5, -18.1], [-15.7, -18.1],
        ...     [-15.7, -15.6], [-18.5, -15.6],
        ... ])
        >>> cropped = pcd.crop(polygon)
        >>> isinstance(cropped, idp.PointCloud)
        True

        Crop by an ROI and keep ROI labels:

        >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
        >>> crops = pcd.crop(roi)
        >>> isinstance(crops, dict)
        True
        """
        if not self.has_points():
            if isinstance(geometry, idp.ROI):
                result = {}
                for key in geometry.keys():
                    empty = PointCloud()
                    empty._crs = self._crs
                    empty._offset = self._offset.copy()
                    result[key] = empty
                return result
            empty = PointCloud()
            empty._crs = self._crs
            empty._offset = self._offset.copy()
            return empty

        if isinstance(geometry, ShapelyPolygon):
            return self._crop_shapely_polygon(geometry)
        elif isinstance(geometry, ShapelyMultiPolygon):
            return self._crop_shapely_multipolygon(geometry)
        elif isinstance(geometry, idp.ROI):
            return self._crop_roi_internal(geometry)
        else:
            if isinstance(geometry, np.ndarray):
                hint = "Use .crop_polygon() for numpy arrays."
            elif isinstance(geometry, (list, tuple)):
                hint = "Use .crop_polygon() with a numpy array."
            else:
                hint = (
                    "Use Shapely Polygon, MultiPolygon, or EasyIDP ROI."
                )
            raise TypeError(
                f"Unsupported crop geometry type {type(geometry).__name__}. "
                f"{hint}"
            )

    def _crop_shapely_polygon(self, polygon):
        """Crop one Shapely polygon using the shared index query.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            2D polygon used to select points along the Z axis.

        Returns
        -------
        PointCloud
            New point cloud containing selected points.

        Notes
        -----
        This helper is for internal integration and Advanced API users.
        Ordinary users should call :meth:`crop`.
        """
        coords = np.array(polygon.exterior.coords)[:, 0:2]
        indices = query_indices_by_polygon(
            self._points_xy, coords, tree=self.tree,
        )
        return self.select_by_index(indices)

    def _crop_shapely_multipolygon(self, multipolygon):
        """Crop a Shapely MultiPolygon using the shared index query.

        Parameters
        ----------
        multipolygon : shapely.geometry.MultiPolygon
            2D multipolygon used to select points along the Z axis.

        Returns
        -------
        PointCloud
            New point cloud containing all points selected by any part.

        Notes
        -----
        This helper is for internal integration and Advanced API users.
        Ordinary users should call :meth:`crop`.
        """
        indices = query_indices_by_multipolygon(
            self._points_xy, multipolygon, tree=self.tree,
        )
        return self.select_by_index(indices)

    def _crop_roi_internal(self, roi):
        """Crop every polygon in an EasyIDP ROI.

        Parameters
        ----------
        roi : easyidp.ROI
            ROI collection whose labels are preserved in the result dict.

        Returns
        -------
        dict[str, PointCloud]
            Mapping from ROI label to cropped point cloud.

        Notes
        -----
        This helper powers both :meth:`PointCloud.crop` and
        :meth:`easyidp.ROI.crop` point-cloud workflows.
        """
        out_dict = {}
        pbar = tqdm(
            roi.items(),
            desc=f"Crop roi from point cloud [{os.path.basename(self.file_path)}]",
        )
        for k, polygon_hv in pbar:
            poly = ShapelyPolygon(polygon_hv[:, 0:2])
            out_dict[k] = self._crop_shapely_polygon(poly)
        return out_dict

    @property
    def offset(self):
        """The offset value of point cloud

        .. caution::

            If change this value directly, the xyz value of point cloud will also be changed, just like moving the whole point cloud.

        Example
        -------
        For example, the point cloud like:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> pts = idp.PointCloud(test_data.pcd.maize_las)
            >>> pts
                            x            y        z  r    g    b
                0  367993.021  3955865.095   57.971  28   21   17
                1  367993.146  3955865.313   57.97   28   23   19
                2  367992.632  3955867.298   57.982  29   22   18
                ...     ...          ...      ...      ...  ...  ...
            49655  368014.791  3955879.494   58.022  33   28   25
            49656  368014.153  3955883.578   58.032  30   40   26
            49657  368016.728  3955874.119   57.967  25   20   18
            >>> pts.offset
            array([ 367900., 3955800.,       0.])

        Change the offset directly:

        .. code-block:: python

            >>> pts.offset = [300, 200, 50]
            >>> pts
                            x        y        z  r    g    b
                0  393.021  265.095  107.971  28   21   17
                1  393.146  265.313  107.97   28   23   19
                2  392.632  267.298  107.982  29   22   18
                ...  ...      ...      ...      ...  ...  ...
            49655  414.791  279.494  108.022  33   28   25
            49656  414.153  283.578  108.032  30   40   26
            49657  416.728  274.119  107.967  25   20   18

        .. caution::

            If you want to change the offset without affecting the point xyz values, please use :func:`update_offset_value`

        """
        return self._offset

    @offset.setter
    def offset(self, o):
        # the point values will change:
        # --------------------------------
        # points =  _point + offset
        #   |         |         |
        # change   no change  change
        o = self._offset_type_check(o)
        self._offset = o
        self._tree = None
        if self._points is not None:
            self._update_btf_print()

    def update_offset_value(self, off_val):
        """Change the offset value without affecting the xyz point values.

        Parameters
        ----------
        off_val : list | tuple | ndarray
            The offset values want to set

        Example
        -------
        For example, the point cloud like:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> pts = idp.PointCloud(test_data.pcd.maize_las)
            >>> pts
                            x            y        z  r    g    b
                0  367993.021  3955865.095   57.971  28   21   17
                1  367993.146  3955865.313   57.97   28   23   19
                2  367992.632  3955867.298   57.982  29   22   18
                ...     ...          ...      ...      ...  ...  ...
            49655  368014.791  3955879.494   58.022  33   28   25
            49656  368014.153  3955883.578   58.032  30   40   26
            49657  368016.728  3955874.119   57.967  25   20   18
            >>> pts.offset
            array([ 367900., 3955800.,       0.])

        Change the offset without affecting the xyz values:

        .. code-block:: python

            >>> pts.update_offset_value([360000, 3955000, 50])

            >>> pts.offset
            array([ 360000., 3955000.,      50.])

            >>> pts.points
                            x            y        z  r    g    b                        nx                     ny                    nz
                0  367993.021  3955865.095   57.971  28   21   17    -0.031496062992125984    0.36220472440944884    0.9291338582677166
                1  367993.146  3955865.313   57.97   28   23   19     0.08661417322834646     0.07086614173228346    0.9921259842519685
                2  367992.632  3955867.298   57.982  29   22   18    -0.007874015748031496    0.26771653543307083    0.9606299212598425
              ...     ...          ...      ...      ...  ...  ...  ...                     ...                    ...
            49655  368014.791  3955879.494   58.022  33   28   25     0.44881889763779526    -0.14960629921259844    0.8740157480314961
            49656  368014.153  3955883.578   58.032  30   40   26     0.44881889763779526    -0.29133858267716534    0.8346456692913385
            49657  368016.728  3955874.119   57.967  25   20   18     0.3228346456692913      0.26771653543307083    0.8976377952755905

        .. caution::

            If you want to change the offset like moving point cloud (also change the xyz values), please use :func:`offset`
        """
        # the point values not change
        # --------------------------------
        # points =  _point + offset
        #   |         |         |
        # no change   change-   change+
        off_val = self._offset_type_check(off_val)

        if self._points is not None:
            self._points = self._points + self._offset - off_val
            self._offset = off_val
            self._update_btf_print()
        else:
            self._offset = off_val

    @staticmethod
    def _offset_type_check(off_val):
        """Validate and normalize a point-cloud offset.

        Parameters
        ----------
        off_val : list, tuple, or numpy.ndarray
            Three-dimensional offset value ``[x, y, z]``.

        Returns
        -------
        numpy.ndarray
            Float64 offset array with shape ``(3,)``.

        Raises
        ------
        ValueError
            If *off_val* is not a 3D list/tuple/array.

        Examples
        --------
        >>> idp.PointCloud._offset_type_check([368000, 3955000, 100])
        array([ 368000., 3955000.,     100.])
        """
        if len(off_val) == 3:
            if isinstance(off_val, (list, tuple)):
                return np.asarray(off_val, dtype=np.float64)
            elif isinstance(off_val, np.ndarray):
                return off_val.astype(np.float64)
            else:
                raise ValueError(
                    f"Only [x, y, z] list or np.array([x, y, z]) are acceptable, not {type(off_val)} type"
                )
        else:
            raise ValueError(
                f"Please give correct 3D coordinate [x, y, z], only {len(off_val)} was given"
            )

    def has_colors(self):
        """Returns True if the point cloud contains point colors.

        Returns
        -------
        bool

        Examples
        --------
        >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        >>> pcd.has_colors()
        True
        """
        if self.colors is None:
            return False
        else:
            return True

    def has_points(self):
        """Returns True if the point cloud contains points.

        Returns
        -------
        bool

        Examples
        --------
        >>> pcd = idp.PointCloud()
        >>> pcd.has_points()
        False
        """
        if self._points is None:
            return False
        else:
            return True

    def has_normals(self):
        """Returns True if the point cloud contains point normals.

        Returns
        -------
        bool

        Examples
        --------
        >>> pcd = idp.PointCloud()
        >>> pcd.has_normals()
        False
        """
        if self.normals is None:
            return False
        else:
            return True

    def to_crs(self, target_crs):
        """Convert point cloud to a new CRS, returning a new PointCloud.

        Parameters
        ----------
        target_crs : str or pyproj.CRS
            Target coordinate reference system.

        Returns
        -------
        PointCloud
            A new PointCloud with coordinates transformed to the target CRS.
            Colors and normals are preserved. Normals are copied without
            rotation (orientation is relative to the surface plane and stays
            valid after rigid-body CRS transform).

        Raises
        ------
        TypeError
            If ``self.crs`` is None.

        Examples
        --------
        >>> pcd.crs = "EPSG:32654"
        >>> pcd_wgs84 = pcd.to_crs("EPSG:4326")
        """
        if self._crs is None:
            raise TypeError(
                "Current PointCloud has no CRS, please specify it by "
                "`pcd.crs = 'current_EPSG'` first."
            )

        if not isinstance(target_crs, pyproj.CRS):
            target_crs = pyproj.CRS.from_user_input(target_crs)

        result = PointCloud()
        if self.colors is not None:
            result.colors = self.colors.copy()
        if self.normals is not None:
            result.normals = self.normals.copy()
        result._crs = target_crs

        if self._points is None:
            return result

        if self._crs.equals(target_crs):
            result._points = self._points.copy()
            result._offset = self._offset.copy()
            result.shape = self.shape
            result._update_btf_print()
            return result

        return self._transform_to(result, self._crs, target_crs)

    def _transform_to(self, result, src_crs, target_crs):
        """Transform absolute coordinates and set offset on *result*.

        Parameters
        ----------
        result : PointCloud
            Target point cloud object to receive transformed coordinates.
        src_crs : pyproj.CRS
            Source coordinate reference system.
        target_crs : pyproj.CRS
            Target coordinate reference system.

        Returns
        -------
        PointCloud
            The same *result* object after coordinate transformation.

        Notes
        -----
        This helper uses ``always_xy=True`` and recomputes the target
        offset after transforming absolute coordinates.
        """
        transformer = pyproj.Transformer.from_crs(
            src_crs, target_crs, always_xy=True
        )
        x, y, z = transformer.transform(
            self._points[:, 0] + self._offset[0],
            self._points[:, 1] + self._offset[1],
            self._points[:, 2] + self._offset[2],
        )
        new_pts = np.vstack([x, y, z]).T
        result._recompute_offset(new_pts)
        result._update_btf_print()
        return result

    def _recompute_offset(self, new_pts):
        """Set ``_points`` and ``_offset`` from new absolute coordinates.

        Parameters
        ----------
        new_pts : numpy.ndarray of shape (N, 3)
            Absolute XYZ coordinates.

        Returns
        -------
        None
            Updates the point cloud in place.

        Notes
        -----
        Coordinates with large absolute values are stored as local values
        plus a rounded offset to reduce precision loss.
        """
        if abs(np.max(new_pts)) > 65536:
            self._offset = np.floor(new_pts.min(axis=0) / 100) * 100
            self._points = new_pts - self._offset
        else:
            self._offset = np.array([0.0, 0.0, 0.0])
            self._points = new_pts
        self.shape = self._points.shape

    def change_crs(self, target_crs):
        """Change the point cloud coordinate system in place.

        Transforms absolute coordinates and automatically recomputes the
        offset for the new coordinate space. Colors and normals are
        preserved. Normals are kept without rotation (orientation is
        relative to the surface plane and stays valid after rigid-body
        CRS transform). The spatial tree is cleared.

        Parameters
        ----------
        target_crs : str or pyproj.CRS
            Target coordinate reference system.

        Returns
        -------
        None
            Modifies the PointCloud in place.

        Raises
        ------
        TypeError
            If ``self.crs`` is None.

        Notes
        -----
        If the source and target CRS are the same, a warning is logged
        and no transformation is performed.

        Examples
        --------
        >>> pcd = idp.PointCloud(test_data.pcd.maize_las)
        >>> pcd.crs = "EPSG:32654"
        >>> pcd.change_crs("EPSG:4326")
        >>> pcd.crs.to_epsg()
        4326
        """
        if self._crs is None:
            raise TypeError(
                "Current PointCloud has no CRS, please specify it by "
                "`pcd.crs = 'current_EPSG'` first."
            )

        if not isinstance(target_crs, pyproj.CRS):
            target_crs = pyproj.CRS.from_user_input(target_crs)

        if self._crs.equals(target_crs):
            logger.warning(
                f"The current CRS [{self._crs.name}] is same as target "
                f"CRS [{target_crs.name}], skip converting"
            )
            return

        if self._points is None:
            self._crs = target_crs
            return

        self._transform_to(self, self._crs, target_crs)
        self._crs = target_crs
        self._tree = None

    def clear(self):
        """Delete all points and make an empty point cloud.

        Returns
        -------
        None
            Clears point, color, normal, CRS, offset, and display state.

        Examples
        --------
        >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        >>> pcd.clear()
        >>> pcd.has_points()
        False
        """
        self._points = None  # internal points with offsets to save memory
        self.colors = None
        self.normals = None
        self.shape = (0, 3)
        self._crs = None
        self._tree = None

        self.offset = np.array([0.0, 0.0, 0.0])

        self.file_ext = ".ply"
        self.file_path = ""

    def save(self, pcd_path):
        """Save current point cloud to a file, support ply, las, laz format.

        Parameters
        ----------
        pcd_path : str
            The file path of saved point cloud, if file extention not given, will use parent point cloud file extention.

        See also
        --------
        write_point_cloud

        Examples
        --------
        >>> pcd = idp.PointCloud(test_data.pcd.lotus_ply_bin)
        >>> pcd.save("lotus_crop.ply")

        """

        from .io import write_point_cloud

        pcd_path = Path(pcd_path)
        file_format = None if pcd_path.suffix else self.file_ext.lstrip(".")
        return write_point_cloud(pcd_path, self, format=file_format)
