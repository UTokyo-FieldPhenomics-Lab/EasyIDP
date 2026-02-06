import pyproj
import numpy as np
from shapely.geometry import Polygon
from loguru import logger
import easyidp as idp

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
    if crs.axis_info[0].direction == 'east':
        return 'xy'
    elif crs.axis_info[0].direction == 'north':
        return 'yx'
    else:
        raise ValueError(f'Unable to parse the crs axis info\n- {crs.axis_info[0]}\n- {crs.axis_info[1]}')


##################
# Subplot Tools  #
##################

def generate_subplots(
    boundary,
    row_num=None,
    col_num=None,
    width=None,
    height=None,
    x_interval=0.0,
    y_interval=0.0,
    keep="all",
):
    """Generate subplots within a boundary polygon.

    Supports two modes (mutually exclusive):
    - **By grid**: Specify `row_num` and `col_num`
    - **By size**: Specify `width` and `height`

    Parameters
    ----------
    boundary : idp.ROI
        ROI object containing exactly one polygon as the boundary.
    row_num : int, optional
        Number of rows (vertical divisions). Used with `col_num`.
    col_num : int, optional
        Number of columns (horizontal divisions). Used with `row_num`.
    width : float, optional
        Subplot width in CRS units (typically meters). Used with `height`.
    height : float, optional
        Subplot height in CRS units (typically meters). Used with `width`.
    x_interval : float, optional
        Horizontal spacing between subplots, by default 0.0
    y_interval : float, optional
        Vertical spacing between subplots, by default 0.0
    keep : str, optional
        Filter mode for subplots based on boundary relationship, by default "all"

        - ``"all"``: Keep all subplots within MAR (including outside boundary)
        - ``"touch"``: Keep subplots that intersect with boundary
        - ``"inside"``: Keep only subplots fully contained within boundary

    Returns
    -------
    idp.ROI
        ROI object containing generated subplot polygons with attributes:
        - `row`: Row index (1-based)
        - `col`: Column index (1-based)
        - `status`: "inside", "touch", or "outside"

    Raises
    ------
    ValueError
        If boundary has zero or more than one polygon,
        or if parameter combination is invalid.

    Examples
    --------
    Generate 4x6 grid of subplots:

    >>> import easyidp as idp
    >>> boundary = idp.ROI("field_boundary.shp")
    >>> subplots = idp.geotools.generate_subplots(
    ...     boundary, row_num=4, col_num=6,
    ...     x_interval=0.5, y_interval=0.5
    ... )
    >>> subplots.save_shp("output_subplots.shp")

    Generate subplots by size (2m x 3m):

    >>> subplots = idp.geotools.generate_subplots(
    ...     boundary, width=2.0, height=3.0,
    ...     x_interval=0.3, y_interval=0.3,
    ...     keep="inside"
    ... )
    """
    # Input validation
    boundary_poly = _validate_boundary(boundary)
    _validate_parameters(row_num, col_num, width, height, keep)

    # Get MAR and orientation vectors
    mar_info = _compute_mar_info(boundary_poly)

    # Calculate grid dimensions
    rows, cols, cell_w, cell_h = _compute_grid_dimensions(
        mar_info, row_num, col_num, width, height, x_interval, y_interval
    )

    # Generate subplot polygons
    subplots_data = _generate_subplot_grid(
        mar_info, rows, cols, cell_w, cell_h, x_interval, y_interval
    )

    # Classify subplots by boundary relationship
    subplots_data = _classify_subplots(subplots_data, boundary_poly)

    # Filter by keep mode
    subplots_data = _filter_by_keep_mode(subplots_data, keep)

    # Convert to ROI object
    return _create_roi_from_subplots(subplots_data, boundary.crs)


def _validate_boundary(boundary):
    """Validate boundary ROI and extract single polygon.

    Parameters
    ----------
    boundary : idp.ROI
        ROI object to validate.

    Returns
    -------
    shapely.geometry.Polygon
        The single boundary polygon.

    Raises
    ------
    ValueError
        If boundary is empty or contains more than one polygon.
    TypeError
        If boundary is not an idp.ROI object.
    """
    if not isinstance(boundary, idp.ROI):
        raise TypeError(
            f"Expected idp.ROI object, got {type(boundary).__name__}"
        )

    if len(boundary) == 0:
        raise ValueError("Boundary ROI is empty, must contain exactly one polygon")

    if len(boundary) != 1:
        raise ValueError(
            f"Boundary must contain exactly one polygon, got {len(boundary)}"
        )

    # Extract the polygon coordinates
    coords = list(boundary.values())[0]
    return Polygon(coords[:, :2])


def _validate_parameters(row_num, col_num, width, height, keep):
    """Validate parameter combinations.

    Parameters
    ----------
    row_num : int or None
        Number of rows.
    col_num : int or None
        Number of columns.
    width : float or None
        Subplot width.
    height : float or None
        Subplot height.
    keep : str
        Keep mode.

    Raises
    ------
    ValueError
        If parameter combination is invalid.
    """
    # Check mode exclusivity
    grid_mode = row_num is not None or col_num is not None
    size_mode = width is not None or height is not None

    if grid_mode and size_mode:
        raise ValueError(
            "Cannot specify both grid mode (row_num/col_num) and "
            "size mode (width/height) simultaneously"
        )

    if not grid_mode and not size_mode:
        raise ValueError(
            "Must specify either grid mode (row_num, col_num) or "
            "size mode (width, height)"
        )

    # Grid mode validation
    if grid_mode:
        if row_num is None or col_num is None:
            raise ValueError(
                "Grid mode requires both row_num and col_num"
            )
        if row_num < 1 or col_num < 1:
            raise ValueError(
                f"row_num and col_num must be >= 1, got row_num={row_num}, col_num={col_num}"
            )

    # Size mode validation
    if size_mode:
        if width is None or height is None:
            raise ValueError("Size mode requires both width and height")
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be > 0, got width={width}, height={height}"
            )

    # Keep mode validation
    valid_keep = ("all", "touch", "inside")
    if keep not in valid_keep:
        raise ValueError(f"keep must be one of {valid_keep}, got '{keep}'")


def _compute_mar_info(polygon):
    """Compute Minimum Area Rectangle info for polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The boundary polygon.

    Returns
    -------
    dict
        Dictionary containing:
        - start_p: Origin point (numpy array)
        - width_dir: Width direction unit vector
        - height_dir: Height direction unit vector
        - width_len: Total width length
        - height_len: Total height length
    """
    # Get MAR from shapely
    mar = polygon.minimum_rotated_rectangle
    coords = list(mar.exterior.coords)

    if len(coords) < 4:
        raise ValueError("Invalid MAR geometry")

    # Extract corner points
    p0 = np.array(coords[0])
    p1 = np.array(coords[1])
    p2 = np.array(coords[2])

    # Calculate edge vectors
    edge1_vec = p1 - p0
    edge2_vec = p2 - p1

    len1 = np.linalg.norm(edge1_vec)
    len2 = np.linalg.norm(edge2_vec)

    # Longest edge is width direction (columns distributed along width)
    # Shortest edge is height direction (rows distributed along height)
    if len1 >= len2:
        width_vec = edge1_vec
        height_vec = edge2_vec
        width_len = len1
        height_len = len2
        start_p = p0
    else:
        width_vec = edge2_vec
        height_vec = p0 - p1
        width_len = len2
        height_len = len1
        start_p = p1

    # Normalize vectors
    width_dir = width_vec / width_len
    height_dir = height_vec / height_len

    return {
        "start_p": start_p,
        "width_dir": width_dir,
        "height_dir": height_dir,
        "width_len": width_len,
        "height_len": height_len,
    }


def _compute_grid_dimensions(
    mar_info, row_num, col_num, width, height, x_interval, y_interval
):
    """Compute grid dimensions based on mode.

    Parameters
    ----------
    mar_info : dict
        MAR information from _compute_mar_info.
    row_num : int or None
        Number of rows (grid mode).
    col_num : int or None
        Number of columns (grid mode).
    width : float or None
        Subplot width (size mode).
    height : float or None
        Subplot height (size mode).
    x_interval : float
        Horizontal spacing.
    y_interval : float
        Vertical spacing.

    Returns
    -------
    tuple
        (rows, cols, cell_width, cell_height)
    """
    total_width = mar_info["width_len"]
    total_height = mar_info["height_len"]

    if row_num is not None and col_num is not None:
        # Grid mode: calculate cell size from total dimensions
        # Formula: cell_w = (TotalW - (cols-1)*x_interval) / cols
        cell_w = (total_width - (col_num - 1) * x_interval) / col_num
        cell_h = (total_height - (row_num - 1) * y_interval) / row_num

        if cell_w <= 0 or cell_h <= 0:
            raise ValueError(
                f"Interval too large: resulting cell size is negative. "
                f"cell_w={cell_w:.2f}, cell_h={cell_h:.2f}"
            )

        return row_num, col_num, cell_w, cell_h
    else:
        # Size mode: calculate grid size from cell dimensions
        # cols = floor((TotalW + x_interval) / (width + x_interval))
        cols = int((total_width + x_interval) / (width + x_interval))
        rows = int((total_height + y_interval) / (height + y_interval))

        if cols < 1:
            cols = 1
        if rows < 1:
            rows = 1

        logger.debug(
            f"Size mode: calculated {rows} rows x {cols} cols "
            f"for {width}x{height} subplots"
        )

        return rows, cols, width, height


def _generate_subplot_grid(
    mar_info, rows, cols, cell_w, cell_h, x_interval, y_interval
):
    """Generate grid of subplot polygons.

    Parameters
    ----------
    mar_info : dict
        MAR information.
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    cell_w : float
        Cell width.
    cell_h : float
        Cell height.
    x_interval : float
        Horizontal spacing.
    y_interval : float
        Vertical spacing.

    Returns
    -------
    list
        List of dicts with 'polygon', 'row', 'col' keys.
    """
    start_p = mar_info["start_p"]
    width_dir = mar_info["width_dir"]
    height_dir = mar_info["height_dir"]

    # Step vectors (cell size + interval)
    step_x = width_dir * (cell_w + x_interval)
    step_y = height_dir * (cell_h + y_interval)

    # Cell size vectors
    vec_cw = width_dir * cell_w
    vec_ch = height_dir * cell_h

    subplots = []
    for r in range(rows):
        for c in range(cols):
            # Origin of current cell
            origin = start_p + (c * step_x) + (r * step_y)

            # Four corners (closed polygon)
            corners = np.array([
                origin,
                origin + vec_cw,
                origin + vec_cw + vec_ch,
                origin + vec_ch,
                origin,  # Close polygon
            ])

            subplots.append({
                "polygon": corners,
                "row": r + 1,
                "col": c + 1,
                "status": None,  # Will be set in _classify_subplots
            })

    return subplots


def _classify_subplots(subplots_data, boundary_poly):
    """Classify subplots by spatial relationship with boundary.

    Parameters
    ----------
    subplots_data : list
        List of subplot dicts.
    boundary_poly : shapely.geometry.Polygon
        The boundary polygon.

    Returns
    -------
    list
        Updated subplots_data with 'status' field set.
    """
    for subplot in subplots_data:
        subplot_poly = Polygon(subplot["polygon"])

        if boundary_poly.contains(subplot_poly):
            subplot["status"] = "inside"
        elif boundary_poly.intersects(subplot_poly):
            subplot["status"] = "touch"
        else:
            subplot["status"] = "outside"

    return subplots_data


def _filter_by_keep_mode(subplots_data, keep):
    """Filter subplots based on keep mode.

    Parameters
    ----------
    subplots_data : list
        List of subplot dicts.
    keep : str
        Keep mode: "all", "touch", or "inside".

    Returns
    -------
    list
        Filtered subplots_data.
    """
    if keep == "all":
        return subplots_data
    elif keep == "touch":
        return [s for s in subplots_data if s["status"] in ("inside", "touch")]
    elif keep == "inside":
        return [s for s in subplots_data if s["status"] == "inside"]
    else:
        return subplots_data


def _create_roi_from_subplots(subplots_data, crs):
    """Create ROI object from subplots data.

    Parameters
    ----------
    subplots_data : list
        List of subplot dicts.
    crs : pyproj.CRS or None
        Coordinate reference system.

    Returns
    -------
    idp.ROI
        ROI object with subplot polygons.
    """
    roi = idp.ROI()
    roi.crs = crs

    # Store additional metadata for each subplot
    roi._subplot_meta = {}

    for i, subplot in enumerate(subplots_data):
        # Generate name: R{row}C{col}
        name = f"R{subplot['row']}C{subplot['col']}"

        # Store polygon
        roi[name] = subplot["polygon"]

        # Store metadata
        roi._subplot_meta[name] = {
            "row": subplot["row"],
            "col": subplot["col"],
            "status": subplot["status"],
        }

    logger.info(
        f"Generated {len(roi)} subplots "
        f"(inside: {sum(1 for s in subplots_data if s['status'] == 'inside')}, "
        f"touch: {sum(1 for s in subplots_data if s['status'] == 'touch')}, "
        f"outside: {sum(1 for s in subplots_data if s['status'] == 'outside')})"
    )

    return roi