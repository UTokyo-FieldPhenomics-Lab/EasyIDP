import pyproj
import numpy as np

############################
# pyproj transformer tools #
############################

def convert_proj(shp_dict, crs_origin, crs_target):
    """ 
    Provide the geo coordinate transfrom based on pyproj package

    Parameters
    ----------
    shp_dict : dict
        the output of read_shp() function
    crs_origin : pyproj object
        the hidden output of read_shp(..., return_proj=True)
    crs_target : str | pyproj object
        | Examples:
        | ``crs_target = pyproj.CRS.from_epsg(4326)``
        | ``crs_target = r'path/to/{shp_name}.prj'``

    Example
    -------
    Data prepare
    
    .. code-block:: python

        >>> 
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

        >>> plot = {'N1W1': np.array([[139.54052962,  35.73475194], [139.54055106,  35.73475596]])}

        >>> proj = pyproj.CRS.from_epsg(4326)
        >>> proj_to = pyproj.CRS.from_epsg(32654)

    Then do the transformation from lon-lat coordainte to WGS 84 / UTM zone 54N (CRS: EPSG:32654)

    .. code-block:: python

        >>> idp.geotools.convert_proj(plot, proj, proj_to)
        {'N1W1': array([[ 368017.75637046, 3955511.0806603 ],
                        [ 368019.70199342, 3955511.49771163]])}

    """
    transformer = pyproj.Transformer.from_proj(crs_origin, crs_target)
    trans_dict = {}
    for k, coord_np in shp_dict.items():
        origin_xy_order = _get_crs_xy_order(crs_origin)
        target_xy_order = _get_crs_xy_order(crs_target)
        if len(coord_np.shape) == 1:
            if origin_xy_order == 'xy':
                # by default, the coord_np is (lon, lat), but transform needs (lat, lon)
                transformed = transformer.transform(coord_np[0], coord_np[1])
            else:
                transformed = transformer.transform(coord_np[1], coord_np[0])
        elif len(coord_np.shape) == 2:
            if origin_xy_order == 'xy':
                transformed = transformer.transform(coord_np[:, 0], coord_np[:, 1])
            else:
                transformed = transformer.transform(coord_np[:, 1], coord_np[:, 0])
        else:
            raise IndexError(
                f"The input coord should be either [x, y] -> shape=(2,) "
                f"or [[x,y], [x,y], ...] -> shape=(n, 2)"
                f"not current {coord_np.shape}")

        if target_xy_order == 'xy':
            coord_np = np.asarray(transformed).T
        else:
            coord_np = np.flip(np.asarray(transformed).T, axis=1)

        # judge if has inf value, means convert fail
        if True in np.isinf(coord_np):
            raise ValueError(
                f'Fail to convert points from "{crs_origin.name}" to '
                f'"{crs_target.name}", '
                f'this may caused by the uncertainty of .prj file strings, '
                f'please check the coordinate manually via QGIS Layer Infomation, '
                f'get the EPGS code, and specify the function argument'
                f'read_shp2d(..., given_proj=pyproj.CRS.from_epsg(xxxx))')
        trans_dict[k] = coord_np

    return trans_dict


def convert_proj3d(points_np, crs_origin, crs_target, is_xyz=True):
    """Transform a point or points from one CRS to another CRS, by pyproj.CRS.Transformer function

    Parameters
    ----------
    points_np : np.ndarray
        the nx3 3D coordinate points
    crs_origin : pyproj.CRS object
        the CRS of points_np
    crs_target : pyproj.CRS object
        the CRS of target
    is_xyz: bool, default false
        The format of points_np; 
        True: x, y, z; False: lon, lat, alt

    Returns
    -------
    np.ndarray

    Notes
    -----
    ``point_np`` and ``fmt`` parameters

    .. tab:: is_xyz = True

        points_np in this format:

        .. code-block:: text

               x  y  z
            0  1  2  3

    .. tab:: is_xyz = False

        points_np in this format:

        .. code-block:: text

                lon  lat  alt
            0    1    2    3
            1    4    5    6

    .. caution::

        pyproj.CRS order: (lat, lon, alt)
        points order in EasyIDP are commonly (lon, lat, alt)

        But if is xyz format, no need to change order

    Example
    -------
    Data prepare
    
    .. code-block:: python

        >>> import pyproj
        >>> import numpy as np
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()

    The geodetic 3D coordinate

    .. code-block:: python

        >>> geocentric = np.array([-3943658.7087006606, 3363404.124223561, 3704651.3067566575])
        >>> geo_c = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})


    And the same point in 3D geocentric coordaintes, order in columns=['lon', 'lat', 'alt']

    .. code-block:: python

        >>> geodetic = np.array([139.54033578028609, 35.73756358928734, 96.87827569602781])
        >>> geo_d = pyproj.CRS.from_epsg(4326)

    Then do the transformation:

    .. code-block:: python

        >>> out_c = idp.geotools.convert_proj3d(geodetic, geo_d, geo_c, is_xyz=True)
        array([-3943658.71530418,  3363404.13219933,  3704651.34270485])

        >>> out_d = idp.geotools.convert_proj3d(geocentric, geo_c, geo_d, is_xyz=False)
        array([139.5403358 ,  35.73756338,  96.849     ])

    """
    ts = pyproj.Transformer.from_crs(crs_origin, crs_target)

    points_np, is_single = is_single_point(points_np)

    # check unit to know if is (lon, lat, lat) -> degrees or (x, y, z) -> meters
    if crs_origin.coordinate_system is not None:
        # suitable for pyproj > 3.4.0 < 3.6.0
        x_unit = crs_origin.coordinate_system.axis_list[0].unit_name
        y_unit = crs_origin.coordinate_system.axis_list[1].unit_name
    elif crs_origin.axis_info is not None:
        # suitable for pyproj > 3.6.1
        x_unit = crs_origin.axis_info[0].unit_name
        y_unit = crs_origin.axis_info[1].unit_name
    else:
        raise AttributeError(
            f'The API of pyproj to get axis unit may changed at current {pyproj.__version__}.'
            f'Unable to find at both "crs.coordinate_system.axis_list" (pyproj < 3.6.0) '
            f'and "crs.axis_info" (pyproj > 3.6.0), please report this issue or downgrade your pyproj version to 3.6.1'
        )

    if x_unit == "degree" and y_unit == "degree": 
        is_xyz = False
    else:
        is_xyz = True

    if is_xyz:
        if crs_target.is_geocentric:
            x, y, z = ts.transform(*points_np.T)
            out =  np.vstack([x, y, z]).T
        elif crs_target.is_geographic:
            lon, lat, alt = ts.transform(*points_np.T)
            # the pyproj output order is reversed
            out = np.vstack([lat, lon, alt]).T
        elif crs_target.is_projected:
            lat_m, lon_m, alt_m = ts.transform(*points_np.T)
            out = np.vstack([lat_m, lon_m, alt_m]).T
        else:
            raise TypeError(f"Given crs is neither `crs.is_geocentric=True` nor `crs.is_geographic` nor `crs.is_projected`")
    else:   
        lon, lat, alt = points_np[:,0], points_np[:,1], points_np[:,2]
        
        if crs_target.is_geocentric:
            x, y, z = ts.transform(lat, lon, alt)
            out = np.vstack([x, y, z]).T
        elif crs_target.is_geographic:
            lat, lon, alt = ts.transform(lat, lon, alt)
            out = np.vstack([lon, lat, alt]).T
        elif crs_target.is_projected and crs_target.is_derived:
            lat_m, lon_m, alt_m = ts.transform(lat, lon, alt)
            out = np.vstack([lon_m, lat_m, alt_m]).T
        else:
            raise TypeError(f"Given crs is neither `crs.is_geocentric=True` nor `crs.is_geographic` nor `crs.is_projected`")
    
    if is_single:
        return out[0, :]
    else:
        return out

def is_single_point(points_np):
    """format one point coordinate ``[x,y,z]`` to ``[[x, y, z]]``

    Parameters
    ----------
    points_np : np.ndarray
        the ndarray point coordiantes

    Returns
    -------
    ndarray, bool
        the converted coordinate, whether is single point

    Example
    --------

    .. code-block:: python

        >>> import easyidp as idp
        >>> import numpy as np

        >>> a = np.array([2,3,4])

        >>> o, b = idp.geotools.is_single_point(a)
        (array([[2, 3, 4]]), True)

    """
    # check if only contains one point
    if points_np.shape == (3,):
        # with only single point
        return np.array([points_np]), True
    else:
        return points_np, False


def _get_crs_xy_order(crs):
    """get the axis order of pyproj CRS coordinates

    Parameters
    ----------
    crs : pyproj object
        _description_
    """
    if crs.axis_info[0].abbrev in ['E', 'X', 'east', 'Lon']:
        return 'xy'
    elif crs.axis_info[0].abbrev in ['N', 'Y', 'north', 'Lat']:
        return 'yx'
    else:
        raise ValueError(f'Unable to parse the crs axis info\n- {crs.axis_info[0]}\n- {crs.axis_info[1]}')