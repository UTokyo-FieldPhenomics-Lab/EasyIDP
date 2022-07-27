import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Polygon

# ignore the warning of shapely convert coordiante
import warnings
warnings.filterwarnings("ignore", message="The array interface is deprecated and will no longer work in Shapely 2.0")


def imarray_crop(imarray, polygon_hv, outside_value=0):
    """crop a given ndarray image by given polygon pixel positions

    Parameters
    ----------
    imarray : ndarray
        the image data, shape = (height,width)
    polygon_hv : ndarray
        pixel position of boundary point, (horizontal, vertical) which reverted the imarray axis 0 to 1
    outside_value: int | float
        specify exact value outside the polgyon, default 0
        But for some DSM geotiff, it can be -10000.0

    returns
    -------
    imarray_out : ndarray
        the (m,n,d) ndrray to store pixel info
    roi_top_left_offset : ndarray
        the (h, v) pixel index that represent the polygon bbox left top corner

    """
    # (horizontal, vertical) remember to revert in all the following codes
    roi_top_left_offset = polygon_hv.min(axis=0)
    roi_max = polygon_hv.max(axis=0)
    roi_length = roi_max - roi_top_left_offset

    roi_rm_offset = polygon_hv - roi_top_left_offset
    # the polygon will generate index outside the image
    # this will cause out of index error in the `poly2mask`
    # so need to find out the point locates on the maximum edge and minus 1 
    # >>> a = np.array([217, 468])  # roi_max
    # >>> b  # polygon
    # array([[217, 456],
    #        [ 30, 468],
    #        [  0,  12],
    #        [187,   0],
    #        [217, 456]])
    # >>> b[:,0] == a[0]
    # array([ True, False, False, False,  True])
    # >>> b[b[:,0] == a[0], 0] -= 1
    # >>> b
    # array([[216, 456],
    #        [ 30, 468],
    #        [  0,  12],
    #        [187,   0],
    #        [216, 456]])
    roi_rm_offset[roi_rm_offset[:,0] == roi_length[0], 0] -= 1
    roi_rm_offset[roi_rm_offset[:,1] == roi_length[1], 1] -= 1

    # remove (160, 160, 1) such fake 3 dimention
    imarray = np.squeeze(imarray)
    
    dim = len(imarray.shape)
    if dim == 2: 
        # only has 2 dimensions
        # e.g. DSM 1 band only, other value outside polygon = empty value
        
        # here need to reverse 
        # imarray.shape -> (h, w), but poly2mask need <- (w, h)
        roi_clipped = imarray[roi_top_left_offset[1]:roi_max[1], 
                              roi_top_left_offset[0]:roi_max[0]]
        rh = roi_clipped.shape[0]
        rw = roi_clipped.shape[1]
        mask = poly2mask((rw, rh), roi_rm_offset)

        roi_clipped[~mask] = outside_value
        imarray_out = roi_clipped

    elif dim == 3: 
        # has 3 dimensions
        # e.g. DOM with RGB or RGBA band, other value outside changed alpha layer to 0
        roi_clipped = imarray[roi_top_left_offset[1]:roi_max[1], roi_top_left_offset[0]:roi_max[0], :]

        rh = roi_clipped.shape[0]
        rw = roi_clipped.shape[1]
        layer_num = roi_clipped.shape[2]

        # here need to reverse 
        # imarray.shape -> (h, w), but poly2mask need <- (w, h)
        mask = poly2mask((rw, rh), roi_rm_offset)

        if layer_num == 3:  
            # DOM without alpha layer
            # but output add mask as alpha layer directly
            mask = mask.astype(np.uint8) * 255

            imarray_out = np.concatenate([roi_clipped, mask[:, :, None]], axis=2).astype(np.uint8)

        elif layer_num == 4:  
            # DOM with alpha layer
            mask = mask.astype(int)

            original_mask = roi_clipped[:, :, 3].copy()
            merged_mask = original_mask * mask

            imarray_out = np.dstack([roi_clipped[:,:, 0:3], merged_mask]).astype(np.uint8)
        else:
            raise TypeError(f'Unable to solve the layer number {layer_num}')

    return imarray_out, roi_top_left_offset


def poly2mask(image_shape, poly_coord, engine="pillow"):
    """convert vector polygon to raster masks, aim to avoid using skimage package

    Parameters
    ----------
    image_shape : tuple with 2 element
        (horizontal, vertical) = (width, height)
        !!! it is reversed with numpy index order !!!
    poly_coord : np.ndarray -> dtype = int or float
        (horizontal, vertical) = (width, height)
        !!! it is reversed with numpy index order !!!

        If dtype is int -> view coord as pixel index number
            Will + 0.5 to coords (pixel center) as judge point
        if dtype is float -> view coords as real coord
            (0,0) will be the left upper corner of pixel square
    engine : str, default "pillow"
        "pillow" or "shapely"
        pillow is slight different than skimage.polygon2mask
        shapely is almost the same with skiamge.polygon2mask, but effiency is very slow, not recommended

    Returns
    -------
    mask : numpy.ndarray
        the generated binary mask
        
    Notes
    -----
    This code is inspired from here:
    https://stackoverflow.com/questions/62280398/checking-if-a-point-is-contained-in-a-polygon-multipolygon-for-many-points

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

    if len(poly_coord.shape)!=2 or poly_coord.shape[1] != 2:
        raise AttributeError(f"Only nx2 ndarray are accepted, not {poly_coord.shape}")

    w, h = image_shape

    # check whether the poly_coords out of mask boundary
    xmin, ymin = poly_coord.min(axis=0)
    xmax, ymax = poly_coord.max(axis=0)

    if max(xmax-xmin, ymax-ymin) > 100:
        warnings.warn("Shaply Engine can not handle size over 100 efficiently, convert using pillow engine")
        engine = "pillow"

    if xmin < 0 or ymin < 0 or xmax >= w or ymax >= h:
        raise ValueError(f"The polygon coords ({xmin}, {ymin}, {xmax}, {ymax}) is out of mask boundary [0, 0, {w}, {h}]")

    if engine == "shapely":
        mask = _shapely_poly2mask(h, w, poly_coord)
    else:   # using pillow
        mask = _pillow_poly2mask(h, w, poly_coord)

    return mask


def _shapely_poly2mask(h, w, poly_coord):
    mask = np.zeros((h, w), dtype=bool)

    # use the pixel center as judgement points
    x = np.arange(0, w) + 0.5
    y = np.arange(0, h) + 0.5

    xx, yy = np.meshgrid(x, y)

    # get the coordinates of all pixel points
    # it is reversed with numpy index order -> [vertical, horizontal]
    pts = np.array([yy.ravel(), xx.ravel()]).T
    points = MultiPoint(pts)

    # judge the type of polygon coordinates
    if np.issubdtype(poly_coord.dtype, np.integer):
        # is int type, mainly means it represent
        # the id of int rather than coords xy values
        # -> shift 0.5 as the pixel center
        poly = Polygon(poly_coord + 0.5)
    elif np.issubdtype(poly_coord.dtype, np.floating):
        poly = Polygon(poly_coord)

    points_in = points.intersection(poly)

    # here will raise warning when obtain coords from shapely multipoints
    # -0.5 turns points center coords to point id
    # here are point index of "masked" pixels
    idx = (np.array(points_in) - 0.5).astype(int)

    # turn to masks
    # idx -> (pixel horizontal, pixel vertical)
    # it is reversed with numpy index order -> [vertical, horizontal]
    mask[idx[:,1], idx[:,0]] = True

    return mask

def _pillow_poly2mask(h, w, poly_coord):
    mask = Image.new('1', (w, h), color=0)
    draw = ImageDraw.Draw(mask)

    xy_pil = [tuple(i) for i in poly_coord]
    
    draw.polygon(xy_pil, fill=1, outline=1)

    mask = np.array(mask, dtype=bool)

    return mask


def rgb2gray(rgb):
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    # how matplotlib did the transform
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])