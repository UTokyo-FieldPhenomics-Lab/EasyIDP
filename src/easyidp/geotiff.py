import os
from functools import wraps
from pathlib import Path

from loguru import logger
import numpy as np
import psutil
import pyproj
import rasterio as rio
from rasterio.enums import ColorInterp
from rasterio.mask import mask as riomask
from shapely.geometry import mapping, Polygon
from skimage.io import imread
from skimage.transform import ProjectiveTransform, warp
from tqdm import tqdm

import easyidp as idp



class GeoTiff(object):
    """A easy GeoTiff class warpped on rasterio
    """

    def __init__(self, file_path:str|Path|None=None, imarray:np.ndarray|None=None, header:dict=None, mask:np.ndarray|None=None):
        """The method to initialize the GeoTiff class

        Parameters
        ----------
        tif_path : None| str | pathlib.Path, optional
            the path to geotiff file, by default None, specify if need to open existing geotiff file
        imarray : np.ndarray
            The pixel data of GeoTIFF if format of (height, width, bands)
        header : dict
            The profile / meta information of geotiff file, including size, bands, CRS, etc
        Transparent_layer: None | int, optional
            the transparent or alpha layer of given GeoTiff file, by default None
            |    None: no alpha layer
            |    0-x : specific alpha layer
            |    -1  : the last layer

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

        """

        self.file_path = Path(file_path) if file_path is not None else None
        """The file path of current GeoTiff, pathlib.Path object

        .. code-block:: python

            >>> dom.file_path
            PosixPath('/Users/<user>/Library/Application Support/easyidp.data/data_for_tests/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif')
        
        """

        self.header = header
        """The Geotiff meta infomation

        .. code-block:: python

            >>> dom.header
            {'height': 5752, 'width': 5490, 'dim': 4, 'nodata': 0, 'dtype': dtype('uint8'), 
             'scale': [0.00738, 0.00738], 'tie_point': [368014.54157, 3955518.2747700005], 
             'crs': <Derived Projected CRS: EPSG:32654>
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
             'profile': <dict `rasterio.io.DatasetReader.profile`>
            }
            >>> dom.header["height"]
            5752

        .. caution::

            Since v2.0.2, this function backend has been switched from `tifffile` to `rasterio` to improve the performance,
            Some of the key tags in `header` like 'tags', 'photometric', 'planarconfig', 'compress' has been deprecated.

        """

        self._imarray = imarray
        self._mask = mask

        #: The layer to represent transparency / alpha
        # self.transparent_layer = None

        if self.file_path is not None:
            self.open(self.file_path)


    @property
    def crs(self):
        """A quick access to ``self.header['crs']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'crs' in self.header.keys():
            return self.header['crs']
        else:
            return None
        
    @property
    def height(self):
        """A quick access to ``self.header['height']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'height' in self.header.keys():
            return self.header['height']
        else:
            return None
        
    @property
    def width(self):
        """A quick access to ``self.header['width']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'width' in self.header.keys():
            return self.header['width']
        else:
            return None
        
    @property
    def dim(self):
        """A quick access to ``self.header['dim']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'dim' in self.header.keys():
            return self.header['dim']
        else:
            return None
        
    @property
    def nodata(self):
        """A quick access to ``self.header['nodata']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'nodata' in self.header.keys():
            return self.header['nodata']
        else:
            return None
        
    @property
    def scale(self):
        """A quick access to ``self.header['scale']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'scale' in self.header.keys():
            return self.header['scale']
        else:
            return None
        
    @property
    def tie_point(self):
        """A quick access to ``self.header['tie_point']``, please access the ``header`` dict to change value"""
        if isinstance(self.header, dict) and 'tie_point' in self.header.keys():
            return self.header['tie_point']
        else:
            return None
        
    @property
    def has_alpha(self) -> bool:
        """Check if this GeoTiff has an alpha channel.
        
        This property reads the colorinterp (color interpretation) from the 
        GeoTiff header to determine if an alpha mask is present. The colorinterp 
        field is extracted from the TIFF metadata by rasterio, which maps to
        the GDAL/TIFF PHOTOMETRIC and EXTRASAMPLES tags.
        
        Returns
        -------
        bool
            True if the GeoTiff has an alpha band, False otherwise.
        
        Example
        -------
        .. code-block:: python
        
            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> dom.has_alpha
            True
            >>> dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)
            >>> dsm.has_alpha
            False
        """
        if isinstance(self.header, dict) and 'has_alpha' in self.header.keys():
            return self.header['has_alpha']
        else:
            return False
        
    @property
    def imarray(self):
        """Access to the pixel values in the type of numpy ndarray"""
        if self._imarray is None:
            # no self stored data, need to read the file at disk
            if not os.path.exists(self.file_path):
                logger.warning(f"Could not find file [{self.file_path}], skip loading")
                return None
            else:
                self._imarray = get_imarray(self.file_path)

        return self._imarray

    @property
    def mask(self) -> np.ndarray | None:
        """Boolean mask where True indicates valid (non-nodata) pixels.
        
        Shape is (height, width). For multi-band images with alpha channel,
        a pixel is considered valid if alpha > 0. For DSM (single band), 
        a pixel is valid if value != nodata.
        
        This property is lazy-computed and cached for efficiency.
        
        Returns
        -------
        np.ndarray | None
            Boolean mask array with shape (height, width), or None if no data.
        
        Example
        -------
        .. code-block:: python
        
            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)
            >>> mask = dsm.mask
            >>> mask.shape
            (5752, 5490)
            >>> mask.dtype
            dtype('bool')
        """
        if self._mask is None:
            if self.header is None:
                logger.warning("No header loaded, cannot compute mask")
                return None
            # Compute mask from imarray (this will load imarray if needed)
            imarray = self.imarray
            if imarray is None:
                return None
            self._mask = self._compute_mask(imarray)
        return self._mask

    def _compute_mask(self, imarray: np.ndarray) -> np.ndarray:
        """Compute the valid pixel mask from imarray.
        
        Supports three mechanisms for reading mask:
        1. GDAL internal mask (per-dataset mask)
        2. Alpha channel (RGBA/MSA images)
        3. Nodata value (DSM/multispectral)
        
        Parameters
        ----------
        imarray : np.ndarray
            The image array with shape (height, width) or (height, width, bands)
        
        Returns
        -------
        np.ndarray
            Boolean mask with shape (height, width), True for valid pixels
        """
        # 1. Try reading GDAL internal mask
        # Only read from file if imarray matches the file dimensions
        if self.file_path is not None and Path(self.file_path).exists():
            try:
                with rio.open(self.file_path) as src:
                    # Check if imarray size matches file size (for full image only)
                    file_shape = (src.height, src.width)
                    imarray_squeezed = np.squeeze(imarray)
                    imarray_shape = imarray_squeezed.shape[:2]  # (height, width)
                    
                    if imarray_shape == file_shape:
                        mask_flags = src.mask_flag_enums
                        # Check if has per-dataset mask (not just nodata/all_valid)
                        has_internal = any('per_dataset' in str(f).lower() for f in mask_flags[0])
                        if has_internal:
                            internal_mask = src.read_masks(1)
                            return internal_mask > 0
            except Exception:
                pass
            
        # 2. Fallback: compute from data (nodata value / alpha channel)
        imarray = np.squeeze(imarray)
        ndim = len(imarray.shape)
        nodata = self.header.get("nodata", None)
        
        if ndim == 2:
            # Single band (DSM): valid if != nodata
            if nodata is not None:
                # Handle NaN comparison
                if np.isnan(nodata) if isinstance(nodata, float) else False:
                    return ~np.isnan(imarray)
                return imarray != nodata
            else:
                # If no nodata defined, all pixels are valid
                return np.ones(imarray.shape, dtype=bool)
        
        elif ndim == 3:
            height, width, bands = imarray.shape
            data_type = self._get_data_type()
            
            # For types with alpha channel
            if data_type in ('rgba', 'msa'):
                # Alpha > 0 means valid
                return imarray[:, :, -1] > 0
            elif nodata is not None:
                # For other multi-band (e.g., RGB, MS), check any band != nodata
                return np.any(imarray != nodata, axis=2)
            else:
                # No alpha, no nodata: all pixels are valid
                return np.ones((height, width), dtype=bool)
        else:
            raise ValueError(f"Unsupported imarray shape: {imarray.shape}")

    def _get_data_type(self) -> str:
        """Detect the data type of this GeoTiff.
        
        Returns
        -------
        str
            One of 'dsm', 'rgb', 'rgba', 'ms', 'msa'
            
            - dsm: single band elevation data
            - rgb: 3-band uint8 visual imagery
            - rgba: 4-band uint8 visual imagery with alpha
            - ms: multi-spectral imagery (>4 bands or non-uint8)
            - msa: multi-spectral imagery with alpha channel
        
        Notes
        -----
        This method reads the colorinterp (color interpretation) from the 
        GeoTiff header to determine if an alpha mask is present. The colorinterp 
        field is extracted from the TIFF metadata by rasterio, which maps to
        the GDAL/TIFF PHOTOMETRIC and EXTRASAMPLES tags.
        
        Example
        -------
        .. code-block:: python
        
            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)
            >>> dsm._get_data_type()
            'dsm'
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> dom._get_data_type()
            'rgba'
        """
        if self.header is None:
            return 'dsm'  # default
            
        dim = self.header.get('dim', 1)
        dtype = self.header.get('dtype', np.dtype('float32'))
        has_alpha = self.header.get('has_alpha', False)
        
        if dim == 1:
            return 'dsm'
        elif dim == 3 and dtype == np.dtype('uint8'):
            # RGB without alpha (has_alpha should be False for 3-band)
            return 'rgb'
        elif dim == 4 and dtype == np.dtype('uint8'):
            # Standard RGBA image  
            return 'rgba'
        else:
            # Multi-spectral: use has_alpha flag from colorinterp to determine
            # if an alpha band is present, instead of relying on heuristics
            if has_alpha:
                return 'msa'
            else:
                return 'ms'


    def _check_data(func):
        """A warp to check if has data"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.header is None:
                raise FileNotFoundError(
                    "Could not operate if not specify correct geotiff file"
                )
            return func(self, *args, **kwargs)

        return wrapper

    def has_data(self) -> bool:
        """Check if the geotiff has data"""
        return self.header is not None

    def open(self, tif_path: str | Path):
        """Open and get the meta information (header) from geotiff

        Parameters
        ----------
        tif_path : str | pathlib.Path
            the path to geotiff file

        Example
        -------
        Though this function can be used by:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> dom = idp.GeoTiff()
            >>> dom.read_geotiff(test_data.pix4d.lotus_dom)

        It is highly recommended to specify the geotiff path when initializing the geotiff object:

        .. code-block:: python

            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

        """
        tif_path = Path(tif_path)
        if tif_path.exists():
            self.file_path = tif_path
            self.header = get_header(self.file_path)
            self._imarray = None
        else:
            logger.warning(f"Can not find file [{tif_path}], skip loading")

    @_check_data
    def save(self, save_path: str | Path, overwrite: bool = False, apply_mask: bool = True) -> bool:
        """Save GeoTiff as tiff file with proper nodata/mask handling.

        Mask is only applied during save. Previous crop operations preserve 
        full rectangular data, allowing further calculations on edge pixels.

        The save strategy depends on data type:
        - DSM: uses nodata value (-32767.0)
        - RGB/RGBA: uses alpha channel
        - MS/MSA (multispectral): adds alpha channel to protect original data

        Parameters
        ----------
        save_path : str | Path
            The file path to save the geotiff
        overwrite : bool, optional
            If True, overwrite existing file without prompting, by default False
        apply_mask : bool, optional
            If True and mask exists, apply mask appropriately, by default True
            
        Returns
        -------
        bool
            True if save succeeded, False if cancelled
            
        Example
        -------
        .. code-block:: python
        
            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()
            >>> dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)
            >>> dsm.save('output_dsm.tif', overwrite=True)
            True
        """
        save_path = Path(save_path).absolute()
        if save_path.suffix.lower() not in ['.tif', '.tiff']:
            save_path = save_path.with_suffix('.tif')

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if save_path.exists() and not overwrite:
            user_input = input(f"File [{save_path}] already exists. Overwrite? (y/n): ")
            if user_input.lower() != 'y':
                logger.info("File save cancelled by user.")
                return False

        # Prepare data and profile
        data_type = self._get_data_type()
        imarray = self._imarray.copy()
        mask = self._mask
        profile = self.header['profile'].copy()

        if data_type == 'dsm':
            # DSM: use nodata value -32767.0
            if apply_mask and mask is not None:
                imarray = imarray.astype(np.float32)  # Ensure float for -32767.0
                imarray[~mask] = -32767.0
                profile['nodata'] = -32767.0
                profile['dtype'] = 'float32'
        
        elif data_type in ('rgb', 'rgba', 'ms', 'msa'):
            # RGB/Multispectral: use alpha band to protect original data
            if apply_mask and mask is not None:
                alpha = (mask * 255).astype('uint8')
                if data_type in ('rgb', 'ms'):
                    # Add new alpha band
                    imarray = np.dstack([imarray, alpha])
                    profile['count'] = imarray.shape[2]
                else:  # rgba / msa - merge with existing alpha
                    imarray[:, :, -1] = np.where(mask, imarray[:, :, -1], 0)
            # Remove nodata for images with alpha
            profile.pop('nodata', None)

        # Write to file
        # rasterio requires (bands, height, width), self._imarray is (height, width, bands)
        imarray_rio = np.moveaxis(imarray, -1, 0)
        
        with rio.open(save_path, 'w', **profile) as dst:
            dst.write(imarray_rio)

        logger.success(f"GeoTiff successfully saved to: {save_path}")
        return True


    @_check_data
    def geo2pixel(self, polygon_hv: np.ndarray, return_index=False) -> np.ndarray:
        """Convert geo coordinate (lon, lat) to geotiff pixel coordinate (horizontal, vertical). 
        A warpper of `rasterio.io.DatasetReader.transform() <https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html#rasterio.io.DatasetReader.transform>`_ 
        and `rasterio.io.DatasetReader.index() <https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html#rasterio.io.DatasetReader.index>`_

        Parameters
        ----------
        points_hv : numpy nx2 array
            [horizontal, vertical] points
        return_index : bool, default false
            if false: will get float coordinates -> (23.5, 27.8)
            if true: will get int pixel index -> (23, 27)

        Returns
        -------
        The ndarray pixel position of these points (horizontal, vertical)

        Example
        -------
        Prepare data:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare the roi data
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> roi.change_crs(dom.crs)
            >>> roi_test = roi[111]
            array([[ 368051.75902187, 3955484.68169527],
                   [ 368053.70441367, 3955485.09879908],
                   [ 368054.11515079, 3955483.14704415],
                   [ 368052.16020711, 3955482.73630818],
                   [ 368051.75902187, 3955484.68169527]])

        Use this function:

        .. code-block:: python

            >>> roi_test_pixel = dom.geo2pixel(roi_test)
            array([[5043.01515811, 4551.90714551],
                   [5306.6183839 , 4495.38901391],
                   [5362.27381938, 4759.85445164],
                   [5097.37630191, 4815.50973136],
                   [5043.01515811, 4551.90714551]])
        """
        pixel_coords = []
        with rio.open(self.file_path) as src:
            # judge x, y order in crs:
            crs_xy_order = idp.geotools._get_crs_xy_order(self.crs)

            for geo_h, geo_v in polygon_hv:
                if crs_xy_order == 'xy':
                    # This CRS expects (x, y) order, our input is (h, v), nothing to do
                    input_x = geo_h
                    input_y = geo_v
                else:   # 'yx' order
                    # This CRS expects (y, x) order, our input is (h, v), need reverse
                    input_x = geo_v
                    input_y = geo_h

                if return_index:
                    # src.index(x, y) returns (row, col)
                    row, col = src.index(input_x, input_y)
                    pixel_coords.append((col, row))  # ensure is the (horizontal, vertical) order
                else:
                    # ~src.transform * (x, y) returns (col_float, row_float)
                    col_float, row_float = ~src.transform * (input_x, input_y)
                    pixel_coords.append((col_float, row_float)) # ensure is the (horizontal, vertical) order
                       
        return np.asarray(pixel_coords)
        
    
    @_check_data
    def pixel2geo(self, polygon_hv):
        """Convert geotiff pixel coordinate or index (horizontal, vertical) to geo coordinate (x, y). 
        A warpper of `rasterio.io.DatasetReader.xy() <https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html#rasterio.io.DatasetReader.xy>`_
        and `rasterio.io.DatasetReader.transform() <https://rasterio.readthedocs.io/en/stable/api/rasterio.io.html#rasterio.io.DatasetReader.transform>`_

        Parameters
        ----------
        points_hv : numpy nx2 array
            [horizontal, vertical] points
            if dtype is np.floating, view as pixel coordinate
            if dtype is np.integer, view as pixel index (return left upper corner of pixel)

        Returns
        -------
        The ndarray pixel position of these points (horizontal, vertical)

        Example
        -------
        Prepare data:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare the roi data
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> roi.change_crs(dom.crs)
            >>> roi_test = roi[111]
            >>> roi_test_pixel = dom.geo2pixel(roi_test)
            array([[5043.01515811, 4551.90714551],
                   [5306.6183839 , 4495.38901391],
                   [5362.27381938, 4759.85445164],
                   [5097.37630191, 4815.50973136],
                   [5043.01515811, 4551.90714551]])

        Use this function:

        .. code-block:: python

            >>> roi_test_back = dom.pixel2geo(roi_test_pixel)
            array([[ 368051.75902187, 3955484.68169527],
                   [ 368053.70441367, 3955485.09879908],
                   [ 368054.11515079, 3955483.14704415],
                   [ 368052.16020711, 3955482.73630818],
                   [ 368051.75902187, 3955484.68169527]])

        """
        geo_coords = []

        # 判断输入像素坐标的类型
        if np.issubdtype(polygon_hv.dtype, np.integer):
            logger.info(f"The input dtype is {polygon_hv.dtype}, viewed as pixel INDEX (horizontal, vertical) rather than pixel coordinate")
            is_integer_pixels = True
        elif np.issubdtype(polygon_hv.dtype, np.floating):
            logger.info(f"The input dtype is {polygon_hv.dtype}, viewed as pixel COORDINATE (horizontal, vertical) rather than pixel index")
            is_integer_pixels = False
        else:
            err_info = f"The `points_hv` only accept numpy ndarray integer and float types, but got [{polygon_hv.dtype}] instead"
            logger.error(err_info)
            raise TypeError(err_info)
        
        with rio.open(self.file_path) as src:
            # judge x, y order in crs:
            crs_xy_order = idp.geotools._get_crs_xy_order(self.crs)

            # src.xy(row, col) 返回该像素中心的地理坐标
            # src.transform * (col, row) 返回的是该像素左上角的地理坐标
            for col, row in polygon_hv:
                if is_integer_pixels:
                    # index coordinate, use src.xy to get the geo-coord
                    # src.xy(row, col) returns (x_geo, y_geo)
                    x_geo, y_geo = src.xy(row=row, col=col)
                else:
                    # pixel coordinate, use src.transform to get the geo-coord
                    # src.transform * (col, row) returns (x_geo, y_geo)
                    x_geo, y_geo = src.transform * (col, row)

                # change order according to crs, ensure outputs order is (horzontal, vertical)
                if crs_xy_order == 'xy':
                    geo_coords.append((x_geo, y_geo))
                else:  # == 'yx'
                     # CRS 是 (y, x) 顺序，但我们想输出 (h, v)，
                    # 此时 x_geo 实际上是 CRS 的 y 轴值，y_geo 实际上是 CRS 的 x 轴值。
                    # 所以我们期望的 (h, v) 应该是 (y_geo, x_geo)
                    geo_coords.append((y_geo, x_geo))
                
        return np.asarray(geo_coords)

    @_check_data
    def point_query(self, points_hv, is_geo=True):
        """Get the pixel value of given point(s)

        Parameters
        ----------
        points_hv : tuple | list | nx2 ndarray
            | The coordinates of qurey points, in order (horizontal, vertical)
        is_geo : bool, optional
            | The given polygon is geo coords ( ``True`` , default) or pixel coords ( ``False`` ) on imarray.

        Returns
        -------
        ndarray
            the obtained pixel value (RGB or height) 

        Example
        -------
        Prequirements

        .. code-block:: python

            >>> import easyidp as idp
            >>> dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)

        Query one point by tuple

        .. code-block:: python
        
            >>> # one point tuple
            >>> pts = (368023.004, 3955500.669)
            >>> dsm.point_query(pts, is_geo=True)
            array([97.45558])

        Query one point by list

        .. code-block:: python

            >>> # one point list
            >>> pts = [368023.004, 3955500.669]
            >>> dsm.point_query(pts, is_geo=True)
            array([97.45558])
        
        
        Query several points by list

        .. code-block:: python

            >>> pts = [
            ...    [368022.581, 3955501.054], 
            ...    [368024.032, 3955500.465]
            ... ]
            >>> dsm.point_query(pts, is_geo=True)
            array([97.624344, 97.59617])

        Query several points by numpy

        .. code-block:: python

            >>> pts = np.array([
            ...    [368022.581, 3955501.054], 
            ...    [368024.032, 3955500.465]
            ... ])
            >>> dsm.point_query(pts, is_geo=True)
            array([97.624344, 97.59617])

        See also
        --------
        easyidp.geotiff.point_query
        """

        # processing the input points
        if isinstance(points_hv, (tuple, list, np.ndarray)):
            temp = np.array(points_hv)

            dim = len(temp.shape)
            if dim == 1 and temp.shape[0] == 2:
                # fit the one point
                points_hv = np.array([temp])
            elif dim == 2 and temp.shape[1] == 2:
                # fit the points
                points_hv = temp
            else:
                raise IndexError("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")
        else:
            raise TypeError(f"Only tuple, list, ndarray are supported, not {type(points_hv)}")
        
        # convert to geo coordinate if input is pixel
        if not is_geo:
            points_hv_geo = self.pixel2geo(points_hv)
        else:
            points_hv_geo = points_hv

        with rio.open(self.file_path) as src:
            crs_xy_order = idp.geotools._get_crs_xy_order(self.crs)
            
            if crs_xy_order == 'xy':
                adjusted_geo_points = points_hv_geo
            else:
                adjusted_geo_points = points_hv_geo[:, [1, 0]]

            sample_gen = np.array(list(src.sample(adjusted_geo_points)))

        # For single-band images (DSM), flatten to 1D array for convenience
        # rasterio returns (n_points, n_bands), e.g., [[val1], [val2]] for DSM
        # We want [val1, val2] for single band
        if sample_gen.shape[1] == 1:
            sample_gen = sample_gen.flatten()

        return sample_gen
    
    @_check_data
    def crop_rois(self, roi, is_geo=True, save_folder=None, return_geotiff:bool=False):
        """Crop several ROIs from the geotiff by given <ROI> object with several polygons and polygon names

        Parameters
        ----------
        roi : easyidp.ROI | dict
            the <ROI> object created by easyidp.ROI(), or dictionary with multiple polygons.
            If you just need crop single polygon with ndarray coordinates, please use GeoTiff.crop_polygon() instead.
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_folder : str, optional
            the folder to save cropped images, use ROI indices as file_names, by default "", means not save.

        Returns
        -------
        dict,
            The dictionary with key=id and value=ndarray data

        Example
        -------
        Prepare data:

        .. code-block:: python
        
            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare dom geotiff
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

            # prepare several ROIs
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> roi = roi[0:3]    # only use 3 for quick example
            >>> roi.change_crs(obj.crs)   # transform to the same CRS like DOM
            {0: array([[ 368017.7565143 , 3955511.08102276],
                       [ 368019.70190232, 3955511.49811902],
                       [ 368020.11263046, 3955509.54636219],
                       [ 368018.15769062, 3955509.13563382],
                       [ 368017.7565143 , 3955511.08102276]]), 
             1: array([[ 368018.20042946, 3955508.96051697],
                       [ 368020.14581791, 3955509.37761334],
                       [ 368020.55654627, 3955507.42585654],
                       [ 368018.601606  , 3955507.01512806],
                       [ 368018.20042946, 3955508.96051697]]), 
             2: array([[ 368018.64801755, 3955506.84956301],
                       [ 368020.59340644, 3955507.26665948],
                       [ 368021.00413502, 3955505.31490271],
                       [ 368019.04919431, 3955504.90417413],
                       [ 368018.64801755, 3955506.84956301]])}

        Use this function:

        .. code-block:: python

            >>> out_dict = obj.crop_rois(roi)
            {"N1W1": array[...], "N1W3": array[...], ...}

            >>> out_dict["N1W1"].shape
            (320, 319, 4)

        If you want automatically save geotiff results to specific folder:

        .. code-block:: python

            >>> tif_out_folder = "./cropped_geotiff"
            >>> os.mkdir(tif_out_folder)
            >>> out_dict = obj.crop_rois(roi, save_folder=tif_out_folder)

        """
        if not isinstance(roi, (dict, idp.ROI)):
            raise TypeError(f"Only <dict> and <easyidp.ROI> with multiple polygons are accepted, not {type(roi)}. If it is 2D ndarray coordiante for just one polygon, please use `GeoTiff.crop_polygon()` instead.")

        pbar = tqdm(roi.items(), desc=f"Crop roi from geotiff [{os.path.basename(self.file_path)}]")
        out_dict = {}
        for k, polygon_hv in pbar:
            if save_folder is not None and Path(save_folder).exists():
                save_path = Path(save_folder) / (k + ".tif")
            else:
                save_path = None

            if polygon_hv.shape[1] == 3:
                # probably xyz coordinates
                polygon_hv = polygon_hv[:, :2]
                logger.info(f"Polygon coordinates are in xyz format {polygon_hv.shape}, only horizontal and vertical coordinates are used for cropping roi.")

            imarray = self.crop_polygon(polygon_hv, is_geo, save_path, return_geotiff)

            out_dict[k] = imarray

        return out_dict
    
    @_check_data
    def crop_shapely_polygon(self, shapely_polygon: Polygon, save_path:str|Path|None=None, return_geotiff:bool=False):
        """Crop a given polygon from geotiff, the base function of cropping geotiff
        
        Parameters
        ----------
        shapely_polygon : shapely.geometry.Polygon
            The polygon to crop, in geo coordinate
        save_path : str, optional
            if given, will save the cropped as \*.tif file to path
        return_geotiff : bool, optional
            if specify to True, will return idp.GeoTiff object instead of ndarray

        Returns
        -------
        idp.GeoTiff object
        """
        with rio.open(self.file_path) as src:
            # 从地理边界计算窗口 (使用 from_bounds 创建一个新的 transform)
            # mask 函数会处理 CRS 轴序，我们只需要提供正确的 GeoJSON 形状
            shapes = [mapping(shapely_polygon)]
            
            out_image, out_transform = riomask(src, shapes, crop=True, nodata=src.nodata)

            # 更新 profile
            out_profile = src.profile.copy()
            out_profile.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            if src.nodata is not None:
                out_profile['nodata'] = src.nodata

            # out_profile all keys and values:
            # {
            #     'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 320, 'height': 321, 
            #     'count': 4, 'crs': CRS.from_wkt('PROJCS["WGS 84 / UTM zone 54N", ... ,AUTHORITY["EPSG","32654"]]'), 
            #     'transform': Affine(0.00738, 0.0, 368017.74449, 0.0, -0.00738, 3955511.4999300004), 
            #     'blockxsize': 5490, 'blockysize': 1, 'tiled': False, 'compress': 'lzw', 'interleave': 'pixel'
            # }

        # create header from profile
        header = {}
        # keys: 'width', 'height', 'dim', 'scale', 'tie_point',
        #       'nodata', 'crs', 'dtype', 'band_num'
        header["height"] = out_profile['height']
        header["width"] = out_profile['width']
        header["dim"] = out_profile['count']
        header["nodata"] = out_profile['nodata']
        header["dtype"] = np.dtype(out_profile['dtype'])

        transform = out_profile['transform']
        header["transform"] = transform
        header["scale"] = [transform.a, abs(transform.e)]
        header["tie_point"] = [transform.c, transform.f]

        header['crs'] = self.crs
        header['profile'] = out_profile.copy()

        # rasterio 读取为 (bands, height, width)
        # 需要转换为 (height, width, bands) 以保持与旧版本 tifffile 的兼容性
        out_imarray = np.moveaxis(out_image, 0, -1)

        out_geotiff = GeoTiff(imarray=out_imarray, header=header)
        
        # Compute mask for cropped region (记录有效区域，不应用到数据)
        # This preserves full rectangular data for further calculations
        out_geotiff._mask = out_geotiff._compute_mask(out_imarray)

        if save_path is not None:
            out_geotiff.file_path = Path(save_path)
            out_geotiff.save(save_path)

        if return_geotiff:
            return out_geotiff
        else:
            return out_geotiff.imarray


    @_check_data
    def crop_polygon(self, polygon_hv, is_geo=True, save_path:str|Path|None=None, return_geotiff:bool=False):
        """Crop a given polygon from geotiff

        Parameters
        ----------
        polygon_hv : numpy nx2 array
            (horizontal, vertical) points
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_path : str | pathlib.Path, optional
            if given, will save the cropped as \*.tif file to path, by default None
        return_geotiff : bool, optional
            if specify to True, will return idp.GeoTiff object instead of ndarray

        Returns
        -------
        imarray_out
            The cropped numpy pixels imarray

        Example
        -------
        Prepare data:

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare geotiff
            >>> dom = idp.GeoTiff(test_data.pix4d.lotus_dom)

            # prepare polygon
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> roi = roi[0]
            >>> roi.change_crs(dom.crs)
            >>> roi
            array([[ 368017.7565143 , 3955511.08102276],
                   [ 368019.70190232, 3955511.49811902],
                   [ 368020.11263046, 3955509.54636219],
                   [ 368018.15769062, 3955509.13563382],
                   [ 368017.7565143 , 3955511.08102276]])

        Use this function:

        .. code-block:: python
            
            >>> imarray = dom.crop_polygon(roi, is_geo=True)
            >>> imarray.shape
            (320, 319, 4)

        If you want to save the previous as new GeoTiff:

        .. code-block:: python

            >>> save_tiff = "path/to/save/cropped.tif"
            >>> imarray = obj.crop_polygon(polygon_hv, is_geo=True, save_path=save_tiff)
            
        """
        if not isinstance(polygon_hv, np.ndarray) or polygon_hv.ndim != 2 or polygon_hv.shape[1] != 2:
            actual_info = polygon_hv.shape if hasattr(polygon_hv, "shape") else type(polygon_hv)
            error_info = f"Polygon_hv must be a 2D numpy array of shape (N, 2), not current input {actual_info}."
            logger.error(error_info)
            raise ValueError(error_info)
        
        crs_xy_order = idp.geotools._get_crs_xy_order(self.crs)

        if is_geo:
            # 调整坐标顺序以匹配 CRS 期望的 (x, y) 或 (y, x)
            adjusted_coords = []
            for h_coord, v_coord in polygon_hv:
                if crs_xy_order == 'xy':
                    adjusted_coords.append((h_coord, v_coord)) # (x, y)
                else: # 'yx'
                    adjusted_coords.append((v_coord, h_coord)) # (y, x)
        else:
            adjusted_coords = self.pixel2geo(polygon_hv)
        
        return self.crop_shapely_polygon( Polygon(adjusted_coords), save_path=save_path, return_geotiff=return_geotiff)


    @_check_data
    def crop_rectangle(self, left:int, top:int, w:int, h:int, is_geo:bool=True, save_path:str|Path|None=None, return_geotiff:bool=False):
        """Extract a rectangle regeion crop from a GeoTIFF image file.

        .. code-block:: text

            (0,0)
            o--------------------------
            |           ^
            |           | top
            |           v
            | <-------> o=============o  ^
            |   left    |<---- w ---->|  |
            |           |             |  h
            |           |             |  |
            |           o=============o  v

        Parameters
        ----------
        top: int 
            Coordinates of the top left corner of the desired crop.
        left: int
            Coordinates of the top left corner of the desired crop.
        h: int
            Desired crop height.
        w: int
            Desired crop width.
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        save_path : str | pathlib.Path, optional
            if given, will save the cropped as \*.tif file to path
        return_geotiff : bool, optional
            if specify to True, will return idp.GeoTiff object instead of ndarray
            
        Returns
        -------
        ndarray
            Extracted crop.

        Example
        -------

        .. code-block:: python

            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            >>> obj = idp.GeoTiff(test_data.pix4d.lotus_dom)
            >>> out = obj.crop_rectangle(left=434, top=918, w=320, h=321, is_geo=False)
            >>> out.shape
            (321, 320, 4)


        .. note::
            It is not recommended to use without specifying parameters like this:
            
            ``crop_rectiange(434, 918, 320, 321)``

            It is hard to know the exactly order

        """
        crs_xy_order = idp.geotools._get_crs_xy_order(self.crs)

        if is_geo:
            # 输入是地理坐标和地理宽度/高度
            # left, top 是 (horizontal, vertical)
            # w, h 是地理宽度和高度
            
            # 计算裁剪区域的地理边界 (minx, miny, maxx, maxy)
            # 注意：rasterio 的 transform 通常是 north-up，y 轴向下（行号增加方向），
            # 所以 top 是北边界，top-h 是南边界。
            # left 是西边界，left+w 是东边界。

            '''
            the geotiff coordiate y axis is upward, so need to reverse top-h
            otherwise will get an negative value.

            Not using abs() to fit the same logic with the geo2pixel() and
            to avoid potential logic error.

            Y    easyidp coord
            ^    o--------------------------> X
            |    |           ^
            |    |           | top
            |    |           v
            |    | <-------> o=============o  ^
            |    |   left    |<---- w ---->|  |
            |    |           |             |  h
            |    |           |             |  |
            |    |           o=============o  v
            |    v Y
            |
            o--------------------------------------------------> X
            Geotiff coordinate
            '''
            
            # 根据 CRS 轴序调整地理坐标
            if crs_xy_order == 'xy':
                # CRS 也是 (x, y) 顺序，所以 left 是 x，top 是 y
                minx_geo = left
                maxx_geo = left + w
                maxy_geo = top # top 是最高的 y 值
                miny_geo = top - h # top - h 是最低的 y 值
            else: # crs_xy_order == 'yx'
                # CRS 是 (y, x) 顺序，所以 left 是 y，top 是 x
                # 在这种情况下，我们假设用户输入的 (left, top) 仍然是 (horizontal, vertical)
                # 那么 left 对应 CRS 的 y 轴，top 对应 CRS 的 x 轴
                # 所以实际的 x 范围是 (top, top+h)
                # 实际的 y 范围是 (left-w, left)
                minx_geo = top # top 是 horizontal
                maxx_geo = top + h # h 是 horizontal 宽度
                maxy_geo = left # left 是 vertical
                miny_geo = left - w # w 是 vertical 宽度
                
            # 从地理边界计算窗口 (使用 from_bounds 创建一个新的 transform)
            # mask 函数会处理 CRS 轴序，我们只需要提供正确的 GeoJSON 形状
            bbox_polygon = Polygon.from_bounds(minx_geo, miny_geo, maxx_geo, maxy_geo)
            

        else:
            # 输入是像素坐标和像素宽度/高度
            # left 是 col, top 是 row
            # w 是像素宽度, h 是像素高度
            # 1. 将像素坐标矩形转换为地理坐标多边形
            # (col, row) -> (x, y)

            polygon = np.array([
                [left,   top], 
                [left+w, top], 
                [left+w, top+h], 
                [left,   top+h], 
                [left,   top]]
            )

            logger.debug(f"polygon (pixel coords): {polygon}")

            polygon_geo = self.pixel2geo(polygon)
            logger.debug(f"polygon_geo (geo coords): {polygon_geo}")
            
            bbox_polygon = Polygon(polygon_geo)

        return self.crop_shapely_polygon(bbox_polygon, save_path=save_path, return_geotiff=return_geotiff)
        
    @_check_data
    def polygon_math(self, polygon_hv: np.ndarray | None = None, is_geo=True, kernel="mean"):
        """Calculate the valus inside given polygon

        Parameters
        ----------
        polygon_hv : numpy nx2 array | None, optional
            (horizontal, vertical) points. 
            If None, the calculation will be performed on the entire image. Defaults to None.
        is_geo : bool, optional
            whether the given polygon is pixel coords on imarray or geo coords (default)
        kernel : str, optional
            The method to calculate polygon summary, options are: ["mean", "min", "max", "pmin5", "pmin10", "pmax5", "pmax10"], please check notes section for more details.
        
        Notes
        -----
        Option details for ``kernel`` parameter:

        - "mean": the mean value inside polygon
        - "min": the minimum value inside polygon
        - "max": the maximum value inside polygon
        - "pmin5": 5th [percentile mean]_ inside polygon
        - "pmin10": 10th [percentile mean]_ inside polygon
        - "pmax5": 95th [percentile mean]_ inside polygon
        - "pmax10": 90th [percentile mean]_ inside polygon

        .. [percentile mean] the mean value of all pixels over/under xth percentile threshold
        
        Example
        -------
        Prepare data:

        .. code-block:: python
        
            >>> import easyidp as idp
            >>> test_data = idp.data.TestData()

            # prepare the roi data
            >>> roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
            >>> dsm = idp.GeoTiff(test_data.pix4d.lotus_dsm)
            >>> roi.change_crs(dsm.crs)
            >>> roi_test = roi[111]
            array([[ 368051.75902187, 3955484.68169527],
                   [ 368053.70441367, 3955485.09879908],
                   [ 368054.11515079, 3955483.14704415],
                   [ 368052.16020711, 3955482.73630818],
                   [ 368051.75902187, 3955484.68169527]])

        Use this function:

        .. code-block:: python

            >>> dsm.polygon_math(roi_test, is_geo=True, kernel="mean")
            97.20491

            >>> dsm.polygon_math(roi_test, is_geo=True, kernel="pmax10")
            97.311844

        .. caution::

            This function is initially designed for doing some simple calculations one band (layer) geotiff.

            If you applying this function on RGB color geotiff, it will return the calculated results of each layer
            
            .. code-block:: python

                >>> dom.polygon_math(roi_test, is_geo=True, kernel="pmax10")
                array([139.97428808, 161.36439038, 122.30964888, 255.        ])

            The four values are RGBA four color channels.

        """

        if polygon_hv is None:  # == full_map
            imarray = self.imarray.copy()
        else:
            imarray = self.crop_polygon(polygon_hv, is_geo, return_geotiff=False)

        # Squeeze to remove single dimensions (e.g., (h, w, 1) -> (h, w))
        imarray = np.squeeze(imarray)
        
        # Compute mask for valid pixels using the unified mask method
        # Create a temporary header-like dict for _compute_mask
        mask = self._compute_mask(imarray)
        
        # Extract valid values based on mask
        if len(imarray.shape) == 2:
            # Single band (DSM)
            inside_value = imarray[mask]
            # Handle case where all pixels are nodata (fix bug #69)
            if len(inside_value) == 0:
                nodata_val = self.header.get("nodata", np.nan)
                inside_value = np.array([nodata_val])
        elif len(imarray.shape) == 3:
            # Multi-band image
            inside_value = imarray[mask, :]  # shape: (valid_pixels, bands)
            # Handle case where all pixels are masked out
            if len(inside_value) == 0:
                nodata_val = self.header.get("nodata", 0)
                inside_value = np.array([[nodata_val] * imarray.shape[2]])
        else:
            raise IndexError(f"Unsupported imarray shape: {imarray.shape}")

        return idp.roi.calculate_kernel_stats(inside_value, kernel)


##############
# Func tools #
##############

def get_header(tif_path: str | Path) -> dict:
    """Read the necessary meta infomation from TIFF file

    Parameters
    ----------
    tif_path : str
        the path to the geotiff file

    Returns
    -------
    header: dict
        the container of acquired meta info

    .. caution::

        Since v2.0.2, this function backend has been switched from `tifffile` to `rasterio` to improve the performance,
        Some of the key tags in `header` like 'tags', 'photometric', 'planarconfig', 'compress' has been deprecated.

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> lotus_full = idp.geotiff.get_header(test_data.pix4d.lotus_dom)
        >>> lotus_full
        {'height': 5752, 'width': 5490, 'dim': 4, 'nodata': 0, 'dtype': dtype('uint8'), 
            'scale': [0.00738, 0.00738], 'tie_point': [368014.54157, 3955518.2747700005], 
            'crs': <Derived Projected CRS: EPSG:32654>
                Name: WGS 84 / UTM zone 54N
                Axis Info [cartesian]:
                - E[east]: Easting (metre)
                - N[north]: Northing (metre)
                Area of Use:
                - name: Between 138°E and 144°E, northern hemisphere between equator and 84°N, 
                        onshore and offshore. Japan. Russian Federation.
                - bounds: (138.0, 0.0, 144.0, 84.0)
                Coordinate Operation:
                - name: UTM zone 54N
                - method: Transverse Mercator
                Datum: World Geodetic System 1984 ensemble
                - Ellipsoid: WGS 84
                - Prime Meridian: Greenwich
        }

            
    """
    if isinstance(tif_path, str):
        file_path = Path(tif_path)
    elif isinstance(tif_path, Path):
        file_path = tif_path
    else:
        logger.error(f"Input should be either [str] or [pathlib.Path], but got [{type(tif_path)}]")

    with rio.open(file_path) as src:
        header = {}
        # keys: 'width', 'height', 'dim', 'scale', 'tie_point',
        #       'nodata', 'crs', 'dtype', 'band_num'

        header["height"] = src.height
        header["width"] = src.width
        header["dim"] = src.count
        header["nodata"] = src.nodata
        header["dtype"] = np.dtype(src.dtypes[0])

        # for save geotiff (deprecated since v2.0.2, rasterio handles this)
        # header["tags"] = None
        # header["photometric"] = None
        # header["planarconfig"] = None
        # header["compress"] = None

        header["transform"] = src.transform
        header["scale"] = [src.transform.a, abs(src.transform.e)]
        header["tie_point"] = [src.transform.c, src.transform.f]
        
        if src.crs:
            header['crs'] = pyproj.CRS.from_wkt(src.crs.to_wkt())
        else:
            header['crs'] = None
            logger.warning(f"[io][geotiff][get_header] Could not find Coordinate Reference System (CRS) for [{tif_path}]\n"
                            f"but you can still manual specify it by \n"
                            f">>> import pyproj \n"
                            f">>> proj = pyproj.CRS.from_epsg() # or from_string() or refer official documents:\n"
                            f"https://pyproj4.github.io/pyproj/dev/api/crs/coordinate_operation.html")
            
        header['profile'] = src.profile.copy()

        # Read colorinterp to detect alpha band
        # colorinterp is a tuple of ColorInterp enums for each band
        header['colorinterp'] = tuple(src.colorinterp)
        
        # Check if any band is marked as alpha
        # Using rasterio.enums.ColorInterp to check for alpha
        header['has_alpha'] = ColorInterp.alpha in src.colorinterp

    
    return header

def get_imarray(tif_path: str | Path) -> np.ndarray:
    """Read full map data as numpy array (time and RAM costy, not recommended, often requires ``4 x file_size`` of RAM)

    Parameters
    ----------
    tif_path : str | Path
        the path to geotiff file

    Returns
    -------
    data: ndarray
        the obtained image data, in shape of ()

    Example
    -------

    .. code-block:: python

        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> maize_dom = idp.GeoTiff(test_data.pix4d.maize_dom)
        >>> maize_part_np = maize_dom.get_imarray()
        >>> maize_part_np.shape
        (722, 836, 4)

    """
    if isinstance(tif_path, str):
        file_path = Path(tif_path)
    elif isinstance(tif_path, Path):
        file_path = tif_path
    else:
        logger.error(f"Input should be either [str] or [pathlib.Path], but got [{type(tif_path)}]")
        return None
        
    with rio.open(file_path) as src:
        # Estimate the required memeory
        height = src.height
        width = src.width
        count = src.count
        dtype = np.dtype(src.dtypes[0])
        required_memory_gb = (height * width * count * dtype.itemsize) / (1024**3)

        # Check available RAM
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        logger.debug(f"Available RAM: {available_memory_gb:.2f} GB | Required RAM: {required_memory_gb:.2f} GB")

        if required_memory_gb > available_memory_gb * 0.99:
            logger.warning(
                f"No enough memory to load this geotiff ({file_path.name})."
                f"{required_memory_gb:.2f} GB RAM, but only {available_memory_gb:.2f} GB available."
            )
            return None
    
        if required_memory_gb > available_memory_gb * 0.8:
            logger.warning(
                f"Fully loal this geotiff ({file_path.name}) requires "
                f"{required_memory_gb:.2f} GB RAM, but only {available_memory_gb:.2f} GB available."
            )

        # 读取数据
        imarray = src.read()
        # rasterio 读取为 (bands, height, width)dom
        # 需要转换为 (height, width, bands) 以保持与旧版本 tifffile 的兼容性
        return np.moveaxis(imarray, 0, -1)
    

def geo2pixel(points_hv, header, return_index=False):
    """[Deprecated] Convert geo coordinate (lon, lat) to geotiff pixel coordinate (horizontal, vertical)

    ..caution::

        Since v2.0.2, this function is deprecated, the conversion method is old and may not accurate.
        Please use `:func:`easyidp.geotiff.GeoTiff.geo2pixel <easyidp.geotiff.GeoTiff.geo2pixel>` instead,
        a warpper for `rasterio.io.DatasetReader.index()` function

    Parameters
    ----------
    points_hv : numpy nx2 array
        [horizontal, vertical] points
    header : dict
        the geotiff head dictionary from get_header()
    return_index : bool, default false
        if false: will get float coordinates -> (23.5, 27.8)
        if true: will get int pixel index -> (23, 27)
    Returns
    -------
    ndarray 
        pixel position of these points (horizontal, vertical)

    Notes
    -----
    Please note: gis UTM coordinate, horizontal is x axis, vertical is y axis, origin at left upper.

    To crop image ndarray:

    - the first columns is vertical pixel (along height),
    - the second columns is horizontal pixel number (along width),
    - the third columns is 3 or 4 bands (RGB, alpha),
    - the x and y is reversed compared with gis coordinates.
        
    This function has already do this reverse, so that you can use the output directly.

    Example
    -------
    .. code-block:: python

        # manual specify header just as example (no need to open geotiff)
        >>> header = {'width': 19436, 'height': 31255, 'dim':4, 
                      'scale': [0.001, 0.001], 'nodata': None,
                      'tie_point': [484576.70205, 3862285.5109300003], 
                      'proj': pyproj.CRS.from_string("WGS 84 / UTM zone 53N")}
        # prepare coord data (no need to read)
        >>> gis_coord = np.asarray([
                [ 484593.67474654, 3862259.42413431],
                [ 484593.41064743, 3862259.92582402],
                [ 484593.64841806, 3862260.06515117],
                [ 484593.93077419, 3862259.55455913],
                [ 484593.67474654, 3862259.42413431]])
        # get the results
        >>> idp.geotiff.geo2pixel(gis_coord, header, return_index=True)
        array([[16972, 26086],
               [16708, 25585],
               [16946, 25445],
               [17228, 25956],
               [16972, 26086]])

    See also
    --------
    :func:`easyidp.GeoTiff.geo2pixel <easyidp.geotiff.GeoTiff.geo2pixel>`

    """
    logger.warning(
        "Since v2.0.2, function `idp.geotiff.geo2pixel()` is deprecated, "
        "the conversion method is old and may not accurate. "
        "Please use `easyidp.GeoTiff.geo2pixel()` instead, "
        "a wrapper for `rasterio.io.DatasetReader.index()` function.",
        DeprecationWarning,
        stacklevel=2
    )

    gis_ph = points_hv[:, 0]
    gis_pv = points_hv[:, 1]

    gis_xmin = header['tie_point'][0]
    gis_ymax = header['tie_point'][1]

    scale_x = header['scale'][0]
    scale_y = header['scale'][1]

    # get float coordinate on pixels
    # - numpy_axis1 = x
    np_ax_h = (gis_ph - gis_xmin) / scale_x
    # - numpy_axis0 = y
    np_ax_v = (gis_ymax - gis_pv) / scale_y

    # get the pixel index (int)
    if return_index:  
        np_ax_h = np.floor(np_ax_h).astype(int)
        np_ax_v = np.floor(np_ax_v).astype(int)

    pixel = np.vstack([np_ax_h, np_ax_v]).T

    return pixel


def pixel2geo(points_hv, header):
    """[Deprecated] Convert geotiff pixel coordinate (horizontal, vertical) to geo coordinate (x, y)

    ..caution::W

        Since v2.0.2, this function is deprecated,  the conversion method is old and may not accurate.
        Please use `:func:`easyidp.geotiff.GeoTiff.pixel2geo <easyidp.geotiff.GeoTiff.pixel2geo>` instead,
        a warpper for `rasterio.io.DatasetReader.xy()` function

    Parameters
    ----------
    points_hv : numpy nx2 array
        [horizontal, vertical] points
    header : dict
        the geotiff head dictionary from get_header()

    Returns
    -------
    The ndarray pixel position of these points (horizontal, vertical)

    Example
    -------
    .. code-block:: python

        >>> header = {'width': 19436, 'height': 31255, 'dim':4, 
                      'scale': [0.001, 0.001], 'nodata': None,
                      'tie_point': [484576.70205, 3862285.5109300003], 
                      'crs': pyproj.CRS.from_string("WGS 84 / UTM zone 53N")}
        >>> pixel_coord = np.asarray([
                [16972, 26086],
                [16708, 25585],
                [16946, 25445],
                [17228, 25956],
                [16972, 26086]])
        >>> idp.geotiff.pixel2geo(pix_coord, header)
        array([[16972.69654   , 26086.79569047],
               [16708.59742997, 25585.10598028],
               [16946.36805996, 25445.77883044],
               [17228.72418998, 25956.37087012],
               [16972.69654   , 26086.79569047]])
    
    See also
    --------
    :func:`easyidp.GeoTiff.pixel2geo <easyidp.geotiff.GeoTiff.pixel2geo>`
    """
    logger.warning(
        "Since v2.0.2, function `idp.geotiff.pixel2geo()` is deprecated, "
        "the conversion method is old and may not accurate. "
        "Please use `easyidp.GeoTiff.pixel2geo()` instead, "
        "a wrapper for `rasterio.io.DatasetReader.xy()` function.",
        DeprecationWarning,
        stacklevel=2
    )

    if not np.issubdtype(points_hv.dtype, np.number):
        raise TypeError(f"The `points_hv` only accept numpy ndarray float and int types")

    gis_xmin = header['tie_point'][0]
    gis_ymax = header['tie_point'][1]

    scale_x = header['scale'][0]
    scale_y = header['scale'][1]

    # the px is numpy axis0 (vertical, h)
    #     py is numpy axis1 (horizontal, w)
    pix_ph = points_hv[:, 0]
    pix_pv = points_hv[:, 1]

    gis_px = gis_xmin + pix_ph * scale_x
    gis_py = gis_ymax - pix_pv * scale_y

    gis_geo = np.vstack([gis_px, gis_py]).T

    return gis_geo


def one_raw_roi2geotiff(
    roi_crs: pyproj.CRS,
    roi_geo_coords: np.ndarray,
    raw_img_path: str | Path,
    roi_raw_px: np.ndarray,
    nodata: float | int = 0,
    has_alpha: bool = True,
) -> GeoTiff:
    """Transform a single ROI's raw image region to a GeoTiff with the same CRS.

    This function takes a ROI defined by both geo coordinates and raw image
    pixel coordinates, crops the raw image, and warps it to create a
    geo-referenced GeoTiff.

    Parameters
    ----------
    roi_crs : pyproj.CRS
        The coordinate reference system of the ROI.
    roi_geo_coords : np.ndarray
        GIS geo coordinates of ROI polygon, shape (n, 2) or (n, 3).
        Only first n-1 points are used (last point is duplicate for closure).
        Z values are ignored if present.
    raw_img_path : str | Path
        Path to the raw image file.
    roi_raw_px : np.ndarray
        ROI pixel coordinates on the raw image, shape (n, 2).
    nodata : float | int, optional
        Value to use for pixels outside the ROI, by default 0.
    has_alpha : bool, optional
        If True, use alpha layer for mask storage.
        If False, apply nodata to mask regions.
        GeoTiff class always stores mask and imarray separately.
        By default True.

    Returns
    -------
    GeoTiff
        A GeoTiff object containing the warped image with proper geo-referencing.
        Call `.save()` to write to file.

    Example
    -------
    .. code-block:: python

        >>> import easyidp as idp
        >>> # After running roi.back2raw(recons)
        >>> roi_geo = roi['N1W1'][:, :2]  # Get 2D geo coords
        >>> roi_px = back2raw_result['N1W1']['IMG_0001']
        >>> gtiff = idp.geotiff.one_raw_roi2geotiff(
        ...     roi_crs=roi.crs,
        ...     roi_geo_coords=roi_geo,
        ...     raw_img_path='path/to/IMG_0001.JPG',
        ...     roi_raw_px=roi_px
        ... )
        >>> gtiff.save('output.tif')

    Notes
    -----
    This function uses skimage.transform.ProjectiveTransform and assumes the
    ROI region is relatively flat. For terrain with significant elevation
    variations, the transformation may produce distortions.
    """
    raw_img_path = Path(raw_img_path)
    if not raw_img_path.exists():
        raise FileNotFoundError(f"Raw image not found: {raw_img_path}")

    # Prepare coordinates: use only first n-1 points (remove closure point)
    roi_geo_2d = roi_geo_coords[:, :2].copy()
    if np.allclose(roi_geo_2d[0], roi_geo_2d[-1]):
        roi_geo_2d = roi_geo_2d[:-1]
        roi_raw_px = roi_raw_px[:-1].copy()

    # Step 1: Read raw image
    raw_img = imread(raw_img_path)

    # Step 2: Crop raw image by ROI pixel coordinates
    roi_px_closed = np.vstack([roi_raw_px, roi_raw_px[0]])  # Re-close for crop
    cropped_img, offset, crop_mask = idp.cvtools.imarray_crop(
        raw_img, roi_px_closed, nodata=None
    )

    # Adjust ROI pixel coords to local crop coordinates
    roi_local_px = roi_raw_px - offset

    # Step 3: Calculate GeoTiff scale from correspondence
    geo_width = roi_geo_2d[:, 0].max() - roi_geo_2d[:, 0].min()
    geo_height = roi_geo_2d[:, 1].max() - roi_geo_2d[:, 1].min()
    px_width = roi_raw_px[:, 0].max() - roi_raw_px[:, 0].min()
    px_height = roi_raw_px[:, 1].max() - roi_raw_px[:, 1].min()

    scale_x = geo_width / px_width if px_width > 0 else 1.0
    scale_y = geo_height / px_height if px_height > 0 else 1.0
    scale = [scale_x, scale_y]

    # Step 4: Calculate GeoTiff dimensions
    tie_point = [roi_geo_2d[:, 0].min(), roi_geo_2d[:, 1].max()]
    out_width = int(np.ceil(geo_width / scale_x))
    out_height = int(np.ceil(geo_height / scale_y))

    # Step 5: Compute projective transform (raw local px -> geo pixel)
    # Target geo pixel coordinates for each ROI vertex
    geo_px = np.zeros_like(roi_geo_2d)
    geo_px[:, 0] = (roi_geo_2d[:, 0] - tie_point[0]) / scale_x
    geo_px[:, 1] = (tie_point[1] - roi_geo_2d[:, 1]) / scale_y

    # Create projective transform from source (local px) to destination (geo px)
    pt = ProjectiveTransform()
    pt.estimate(src=roi_local_px, dst=geo_px)

    # Step 6: Warp cropped image to geo-referenced space
    output_shape = (out_height, out_width)
    if len(cropped_img.shape) == 3:
        output_shape = (out_height, out_width, cropped_img.shape[2])

    warped_img = warp(
        cropped_img,
        pt.inverse,
        output_shape=output_shape,
        preserve_range=True,
        cval=nodata,
    ).astype(cropped_img.dtype)

    # Step 7: Generate mask from ROI geo coords polygon
    geo_px_closed = np.vstack([geo_px, geo_px[0]])
    roi_mask = idp.cvtools.poly2mask((out_width, out_height), geo_px_closed)

    # Step 8: Create GeoTiff header
    n_bands = warped_img.shape[2] if len(warped_img.shape) == 3 else 1
    header = {
        'height': out_height,
        'width': out_width,
        'dim': n_bands,
        'dtype': warped_img.dtype,
        'nodata': nodata if not has_alpha else None,
        'scale': scale,
        'tie_point': tie_point,
        'crs': roi_crs,
        'has_alpha': False,
        'profile': {
            'driver': 'GTiff',
            'height': out_height,
            'width': out_width,
            'count': n_bands,
            'dtype': str(warped_img.dtype),
            'crs': roi_crs,
            'transform': rio.transform.from_bounds(
                tie_point[0],
                tie_point[1] - out_height * scale_y,
                tie_point[0] + out_width * scale_x,
                tie_point[1],
                out_width,
                out_height,
            ),
        },
    }

    # Create GeoTiff object
    gtiff = GeoTiff(imarray=warped_img, header=header, mask=roi_mask)
    return gtiff


def back2raw2geotiff(
    recons: idp.reconstruct.Recons,
    back2raw_result: dict,
    roi,
    output_folder: str | Path | None = None,
    nodata: float | int = 0,
    has_alpha: bool = True,
    img_suffix: str = '.JPG',
) -> dict:
    """Convert back2raw results to GeoTiff objects.

    A higher-level wrapper that processes the output of roi.back2raw() or
    sort_img_by_distance(), transforming each ROI's raw image regions
    into geo-referenced GeoTiff files.

    Parameters
    ----------
    recons: easyidp.reconstruct.Recons
        the reconstruction object like <easyidp.Metashape> or <easyidp.Pix4D> object (support both) 
    back2raw_result : dict
        Output from `roi.back2raw()` or `sort_img_by_distance()`.
        Structure: {roi_id: {img_id: roi_pixel_coords, ...}, ...}
    roi : easyidp.ROI
        The ROI object with geo coordinates and CRS.
    output_folder : str | Path, optional
        Folder to save GeoTiff files. If specified, files are saved as
        'output_folder/roi_id/img_id.tif'. By default None (no saving).
    nodata : float | int, optional
        Value for pixels outside ROI, by default 0.
    has_alpha : bool, optional
        If True, use alpha layer for mask. By default True.
    img_suffix : str, optional
        File suffix for raw images, by default '.JPG'.

    Returns
    -------
    dict
        Dictionary with same structure as input:
        {roi_id: {img_id: GeoTiff, ...}, ...}

    Example
    -------
    .. code-block:: python

        >>> import easyidp as idp
        >>> roi = idp.ROI('plots.shp')
        >>> roi.get_z_from_dsm('dsm.tif')
        >>> ms = idp.Metashape('project.psx')
        >>> back2raw_out = roi.back2raw(ms)
        >>>
        >>> geotiffs = idp.geotiff.back2raw2geotiff(
        ...     back2raw_result=back2raw_out,
        ...     roi=roi,
        ...     raw_img_folder='./photos',
        ...     output_folder='./geotiff_output'
        ... )
        >>> # Access specific GeoTiff
        >>> geotiffs['N1W1']['IMG_0001'].save('custom_path.tif')

    See Also
    --------
    easyidp.ROI.back2raw : Generate back2raw results
    one_raw_roi2geotiff : Process single ROI-image pair
    """
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    result = {}
    total_items = sum(len(imgs) for imgs in back2raw_result.values())

    with tqdm(total=total_items, desc="Converting to GeoTiff") as pbar:
        for roi_id, img_dict in back2raw_result.items():
            result[roi_id] = {}

            # Get ROI geo coordinates (use only xy, remove z if present)
            roi_geo_coords = roi[roi_id][:, :2]

            for img_id, roi_raw_px in img_dict.items():
                pbar.set_postfix_str(f"{roi_id}/{img_id}")

                # Construct raw image path
                try:
                    raw_img = recons.photos[img_id]
                    img_path = raw_img.path

                    if not Path(img_path).exists():
                        logger.warning(f"Image file not found at {img_path}, skipping")
                        pbar.update(1)
                        continue
                except:
                    logger.warning(f"Image not found: {img_id}, skipping")
                    pbar.update(1)
                    continue

                # Convert to GeoTiff
                gtiff = one_raw_roi2geotiff(
                    roi_crs=roi.crs,
                    roi_geo_coords=roi_geo_coords,
                    raw_img_path=img_path,
                    roi_raw_px=roi_raw_px,
                    nodata=nodata,
                    has_alpha=has_alpha,
                )

                result[roi_id][img_id] = gtiff

                # Save if output folder specified
                if output_folder is not None:
                    roi_folder = output_folder / str(roi_id)
                    roi_folder.mkdir(parents=True, exist_ok=True)
                    save_path = roi_folder / f"{img_id}.tif"
                    gtiff.save(save_path, overwrite=True)

                pbar.update(1)

    return result