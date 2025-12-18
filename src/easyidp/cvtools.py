import numpy as np
from skimage.draw import polygon2mask
from shapely.geometry import MultiPoint, Polygon

from loguru import logger

# ignore the warning of shapely convert coordiante
# import warnings
# warnings.filterwarnings("ignore", message="The array interface is deprecated and will no longer work in Shapely 2.0")


def imarray_crop(
    imarray: np.ndarray, 
    polygon_hv: np.ndarray, 
    nodata_value: float | int = 0, 
    input_mask: np.ndarray | None = None,
    return_mask: bool = True
) -> tuple:
    """Crop a given ndarray image by given polygon pixel positions.
    
    This is a basic numpy processing tool for cropping images with polygon regions.
    
    Parameters
    ----------
    imarray : np.ndarray
        The image data in numpy ndarray.
        Shape can be (height, width) for DSM or (height, width, bands) for DOM.
        
    polygon_hv : np.ndarray
        2D ndarray of shape (n, 2) containing pixel positions of polygon boundary.
        Coordinates are in (horizontal, vertical) order.
        
        .. caution::
            Coordinate order is reversed from numpy indexing.
            horizontal = numpy axis 1, vertical = numpy axis 0.
            
    nodata_value : float | int, optional
        Value to use for pixels outside the polygon, by default 0.
        For DSM geotiffs, this is typically -10000.0.
        
    input_mask : np.ndarray | None, optional
        Optional input mask with shape matching imarray's first two dimensions.
        If provided, the polygon mask will be combined (AND) with this mask.
        
    return_mask : bool, optional
        If True, return the polygon mask along with cropped data, by default True.
        
    Returns
    -------
    imarray_out : np.ndarray
        The cropped image array with pixels outside polygon set to nodata_value.
    roi_offset : np.ndarray
        The (horizontal, vertical) pixel offset of the crop region's top-left corner.
    mask : np.ndarray, optional
        The (height, width) boolean mask if return_mask=True.
        True values indicate pixels inside the polygon.
        
    Example
    -------
    .. code-block:: python
    
        >>> import numpy as np
        >>> import easyidp as idp
        >>> 
        >>> # Create sample image and polygon
        >>> imarray = np.random.rand(100, 100)
        >>> polygon = np.array([[20, 20], [80, 20], [80, 80], [20, 80], [20, 20]])
        >>> 
        >>> # Crop with mask output
        >>> cropped, offset, mask = idp.cvtools.imarray_crop(
        ...     imarray, polygon, nodata_value=-1, return_mask=True
        ... )
    """
    # Input validation
    if not isinstance(imarray, np.ndarray):
        raise TypeError(f"The `imarray` must be numpy ndarray, not {type(imarray)}")
    
    if not (np.issubdtype(imarray.dtype, np.integer) or np.issubdtype(imarray.dtype, np.floating)):
        raise TypeError(f"The `imarray` only accept numpy ndarray integer and float types, not {imarray.dtype}")
    
    if not isinstance(polygon_hv, np.ndarray):
        raise TypeError(f"Only numpy 2d array is accepted for polygon_hv, not {type(polygon_hv)}")
    
    if len(polygon_hv.shape) != 2 or polygon_hv.shape[1] != 2:
        raise AttributeError(f"polygon_hv must have shape (n, 2), not {polygon_hv.shape}")
    
    # Convert polygon to integer if float
    if np.issubdtype(polygon_hv.dtype, np.floating):
        polygon_hv = polygon_hv.astype(np.int32)
    elif not np.issubdtype(polygon_hv.dtype, np.integer):
        raise TypeError(f"polygon_hv must have integer or float dtype, not {polygon_hv.dtype}")

    # Calculate bounding box
    roi_min = polygon_hv.min(axis=0)  # (horizontal_min, vertical_min)
    roi_max = polygon_hv.max(axis=0)  # (horizontal_max, vertical_max)
    roi_size = roi_max - roi_min
    
    # Offset polygon to local coordinates
    polygon_local = polygon_hv - roi_min
    
    # Handle edge case: polygon points on maximum boundary need to be inside
    polygon_local[polygon_local[:, 0] == roi_size[0], 0] -= 1
    polygon_local[polygon_local[:, 1] == roi_size[1], 1] -= 1
    
    # Squeeze to handle (h, w, 1) -> (h, w)
    imarray = np.squeeze(imarray)
    ndim = len(imarray.shape)
    
    if ndim not in [2, 3]:
        raise ValueError(f"imarray must be 2D or 3D, got shape {imarray.shape}")

    # Crop to bounding box
    # Note: numpy uses (row, col) = (vertical, horizontal)
    roi_cropped = imarray[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]]
    
    # Generate polygon mask
    crop_height, crop_width = roi_cropped.shape[:2]
    polygon_mask = poly2mask((crop_width, crop_height), polygon_local)
    
    # Combine with input mask if provided
    if input_mask is not None:
        # Crop input mask to match
        input_mask_cropped = input_mask[roi_min[1]:roi_max[1], roi_min[0]:roi_max[0]]
        polygon_mask = polygon_mask & input_mask_cropped
    
    # Apply mask: set outside pixels to nodata
    imarray_out = roi_cropped.copy()
    imarray_out[~polygon_mask] = nodata_value
    
    if return_mask:
        return imarray_out, roi_min, polygon_mask
    else:
        return imarray_out, roi_min


def poly2mask(image_shape, poly_coord, engine="skimage"):
    """convert vector polygon to raster masks

    Parameters
    ----------
    image_shape : tuple with 2 element
        .. caution::
            it is reversed with numpy index order 

        (horizontal, vertical) = (width, height)

    poly_coord : (n, 2) np.ndarray -> dtype = int or float
        .. caution::
            The xy is reversed with numpy index order

            (horizontal, vertical) = (width, height)
            
    engine : str, default "skimage"
        | "skimage" only; the "pillow" and "shapely" has been deprecated;
        | skimage - ``skimage.draw.polygon2mask``, the default method;
        | pillow is slight different than "skimage", deprecated;
        | shapely is almost the same with "skiamge", but effiency is very slow, deprecated.

    Returns
    -------
    mask : numpy.ndarray
        the generated binary mask
        
    Notes
    -----
    This code is inspired from [1]_ .

    And for the poly_coord, if using **shapely** engine, it will following this logic for int and float:

    If dtype is int -> view coord as pixel index number
        Will + 0.5 to coords (pixel center) as judge point
    if dtype is float -> view coords as real coord
        (0,0) will be the left upper corner of pixel square

    References
    ----------
    .. [1] https://stackoverflow.com/questions/62280398/checking-if-a-point-is-contained-in-a-polygon-multipolygon-for-many-points

    """

    # check the type of input
    # is ndarray -> is int or float ndarray
    if not isinstance(poly_coord, np.ndarray) or \
        not (
            np.issubdtype(poly_coord.dtype, np.integer) \
                or \
            np.issubdtype(poly_coord.dtype, np.floating)
            ):
        raise TypeError(f"The `poly_coord` only accept numpy ndarray integer and float types")

    if len(poly_coord.shape) != 2 or poly_coord.shape[1] != 2:
        raise AttributeError(f"Only nx2 ndarray are accepted, not {poly_coord.shape}")

    w, h = image_shape

    # check whether the poly_coords out of mask boundary
    xmin, ymin = poly_coord.min(axis=0)
    xmax, ymax = poly_coord.max(axis=0)

    if xmin < 0 or ymin < 0 or xmax >= w or ymax >= h:
        raise ValueError(f"The polygon coords ({xmin}, {ymin}, {xmax}, {ymax}) is out of mask boundary [0, 0, {w}, {h}]")

    if engine != "skimage":
        logger.warning("The `shapely` and `pillow` engine has been deprecated, using only skimage as engine since easyidp 2.0.1")
    mask = polygon2mask((w, h), poly_coord).T

    return mask


def rgb2gray(rgb):
    """Transform the RGB image to gray image

    Parameters
    ----------
    rgb : mxnx3 ndarray
        The RGB ndarray image need to be converted

    Returns
    -------
    gray : mxn ndarray
        The output 2D ndarray after convension

    Notes
    -----
    Using the same formular that matplotlib did [1]_ for the transformation.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])