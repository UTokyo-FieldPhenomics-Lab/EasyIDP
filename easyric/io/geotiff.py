import numpy as np
import pyproj
from skimage.external import tifffile
from skimage.draw import polygon
from pyproj.exceptions import CRSError

from easyric import caas_lite
import tifffile as tf


def _get_header_string(geotiff_path):
    with tifffile.TiffFile(geotiff_path) as tif:
        # >>> print(tif.info())
        '''
        TIFF file: broccoli_tanashi_5_20191008_mavicRGB_15m_M_dsm.tif, 274 MiB, little endian

        Series 0: 19866x13503, float32, YX, 1 pages, not mem-mappable

        Page 0: 19866x13503, float32, 32 bit, minisblack, lzw
        * 256 image_width (1H) 13503
        * 257 image_length (1H) 19866
        * 258 bits_per_sample (1H) 32
        * 259 compression (1H) 5
        * 262 photometric (1H) 1
        * 273 strip_offsets (19866I) (159344, 159734, 160140, 160546, 160952, 161359, 1
        * 277 samples_per_pixel (1H) 1
        * 278 rows_per_strip (1H) 1
        * 279 strip_byte_counts (19866I) (390, 406, 406, 406, 407, 410, 412, 414, 416,
        * 284 planar_configuration (1H) 1
        * 305 software (12s) b'pix4dmapper'
        * 317 predictor (1H) 3
        * 339 sample_format (1H) 3
        * 33550 model_pixel_scale (3d) (0.0029700000000000004, 0.0029700000000000004, 0
        * 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823,
        ******** some case ********* (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823, 0
        * 34735 geo_key_directory (32H) (1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 1026
        * 34737 geo_ascii_params (30s) b'WGS 84 / UTM zone 54N|WGS 84|'
        * 42113 gdal_nodata (7s) b'-10000'

        ====================
        but in some case, it will has some series like:

        Series 1: 3107x3989, float32, YX, 1 pages, not mem-mappable

        Page 1: 3107x3989, float32, 32 bit, minisblack, lzw, reduced|tiled
        * 254 new_subfile_type (1I) 1
        * 256 image_width (1H) 3989
        * 257 image_length (1H) 3107
        * 258 bits_per_sample (1H) 32
        * 259 compression (1H) 5
        * 262 photometric (1H) 1
        ...
        '''
        return tif.info()


def _prase_header_string(geotiff_string):
    header = {'width': None, 'length': None, 'scale': None, 'tie_point': None, 'nodata': None, 'proj': None}

    for line in geotiff_string.split('\n'):
        # just load page 0 contents
        if "Series 1" in line:
            break

        if '*' in line:
            line_sp = line.split(' ')
            code = line_sp[1]
            if code == '256':
                # * 256 image_width (1H) 13503
                header['width'] = int(line_sp[-1])
            elif code == '257':
                # * 257 image_length (1H) 19866
                header['length'] = int(line_sp[-1])
            elif code == '33550':
                # * 33550 model_pixel_scale (3d) (0.0029700000000000004, 0.0029700000000000004, 0
                x = float(line_sp[-3][1:-1])
                y = float(line_sp[-2][:-1])
                header['scale'] = (x, y)
            elif code == '33922':
                # * 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823,
                x = float(line_sp[7][:-1])
                y = float(line_sp[8][:-1])
                header['tie_point'] = (x, y)
            elif code == '34737':
                # * 34737 geo_ascii_params (30s) b'WGS 84 / UTM zone 54N|WGS 84|'
                # * 34737 geo_ascii_params (7s) b'WGS-84'
                # * 34737 geo_ascii_params (30s) b'WGS 84|WGS 84 / UTM zone 54N|'
                bstring = max(line.split('34737')[-1].split(") b'")[-1].split('|'), key=len)
                #bstring = line.split('34737')[-1].split(") b'")[-1].split('|')[0]
                # in WGS 84|WGS 84 / UTM zone 54N| -> WGS 84, will give a wrong result, so pick the longest string instead
                # url: https://www.kite.com/python/answers/how-to-find-the-longest-string-in-a-list-in-python
                print(f'[io][geotiff][GeoCorrd] Comprehense [{line}] to geotiff coordinate tag [{bstring}]')
                try:
                    proj = pyproj.CRS.from_string(bstring)
                    header['proj'] = proj
                except CRSError as e:
                    print(f'[io][geotiff][GeoCorrd] Generation failed, because [{e}], but you can manual specify it later by \n'
                          '>>> import pyproj \n'
                          '>>> proj = pyproj.CRS.from_epsg() # or from_string() or refer official documents:\n'
                          'https://pyproj4.github.io/pyproj/dev/api/crs/coordinate_operation.html')
                    pass
            elif code == '42113':
                header['nodata'] = int(line_sp[-1][2:-1])
            else:
                pass
        else:
            continue

    return header


def get_header(geotiff_path):
    geotiff_string = _get_header_string(geotiff_path)
    header = _prase_header_string(geotiff_string)

    return header


def get_imarray(geotiff_path, geo_head=None):
    """

    Parameters
    ----------
    geotiff_path
    geo_head

    Returns
    -------

    """
    with tifffile.TiffFile(geotiff_path) as tif:
        if geo_head is None:
            geo_head = _prase_header_string(tif.info())
        data = tif.asarray().astype(float)
        data[data == geo_head['nodata']] = np.nan

    return data


def point_query(geotiff, point_hv, geo_head=None, mode="full"):
    '''
    :param geotiff:
        string, the path of geotiff(dsm only) file
        ndarray, the ndarray of readed geotiff file (avoid read every time in for loops)
    :param point_hv: can be following three types
        1. one point tuple (not recommended)
            :input:  (34.57, 45.62)
            :return: float, value
        2. 2d numpy array
            :input:  np.asarray([[34.57, 45.62],[35.57, 46.62]])
            :return: np.array, 1d
        3. list of 2d numpy arrays
            a = np.asarray([[34.57, 45.62],[35.57, 46.62]])
            b = np.asarray([[36.57, 47.62],[38.57, 48.62]])
            :input:  p_list = [a, b]
            :return: list, contains np.1darray of each
    :param geo_head: the geotiff head of reading geotiff, default is None
        if geotiff is string:
            geo_head = None -> read header from geotiff_path
            geo_head = Given -> do nothing
        if geotiff is ndarray:
            geo_head = None -> point_hv is pixel_coordinate
            geo_head = Given -> use geo2pixel to convert point_hv from geo_coordinate to pixel_coordinate
    :param mode: full -> load full dsm into memeory (suite query large amount of points)
                 part -> partly load dsm into memeory (suit query a few points)
    '''
    is_geo = True
    if not isinstance(geotiff, str):
        if mode == "full":
            with tifffile.TiffFile(geotiff) as tif:
                # equal to get_header()
                if geo_head is None:
                    geo_head = _prase_header_string(tif.info())
                # equal to get_imarray()
                data = tif.asarray().astype(float)
                data[data == geo_head['nodata']] = np.nan
        else:
            ts = caas_lite.TiffSpliter(geotiff, 2000, 2000)
            tif = tf.TiffFile(ts.tif_path)
    elif isinstance(geotiff, np.ndarray):
        data = geotiff
        if geo_head is None:
            is_geo = False
        else:
            is_geo = True
    else:
        raise TypeError(f'The geotiff should be either "str" or "np.ndarray", not {type(geotiff)}')

    if isinstance(point_hv, tuple):
        point_hv = np.asarray([[point_hv[0], point_hv[1]]])
        if is_geo:
            px = geo2pixel(point_hv, geo_head)  # px = (horizontal, vertical)
        else:
            px = point_hv
        # imarray axis0 = vertical, axis1 = horizontal
        if mode == "full":
            height_values = data[px[:, 1], px[:, 0]]
        else:
            cropped = ts.get_crop(tif.pages[0], px[:, 1], px[:, 0], h=1, w=1)
            height_values = cropped[0]
    elif isinstance(point_hv, np.ndarray):
        if is_geo:
            px = geo2pixel(point_hv, geo_head)  # px = (horizontal, vertical)
        else:
            px = point_hv
        # imarray axis0 = vertical, axis1 = horizontal
        if mode == "full":
            height_values = data[px[:, 1], px[:, 0]]
        else:
            cropped = ts.get_crop(tif.pages[0], px[:, 1], px[:, 0], h=1, w=1)
            height_values = cropped[0]
    elif isinstance(point_hv, list):
        height_values = []
        for p in point_hv:
            if not isinstance(p, np.ndarray):
                raise TypeError('Only numpy.ndarray in list are supported')
            else:
                if is_geo:
                    px = geo2pixel(p, geo_head)  # px = (horizontal, vertical)
                else:
                    px = p
                # imarray axis0 = vertical, axis1 = horizontal
                if mode == "full":
                    height_values.append(data[px[:, 1], px[:, 0]])
                else:
                    cropped = ts.get_crop(tif.pages[0], px[:, 1], px[:, 0], h=1, w=1)
                    height_values.append(cropped[0])
    else:
        raise TypeError('Only one point tuple, numpy.ndarray, and list contains numpy.ndarray are supported')

    return height_values


def mean_values(geotiff_path, polygon='all', geo_head=None):
    """
    :param geotiff_path:
    :param polygon:
    :param geo_head:
    :return:
    """
    with tifffile.TiffFile(geotiff_path) as tif:
        # equal to get_header()
        if geo_head is None:
            geo_head = _prase_header_string(tif.info())
        # equal to get_imarray()
        #data = tif.asarray().astype(float)
        #data[data == geo_head['nodata']] = np.nan
            
        # !!! a temporary modify for large DOM !!!
        ts = caas_lite.TiffSpliter(geotiff_path, 2000, 2000)
        tif = tf.TiffFile(ts.tif_path)

        if polygon == 'all':
            data = tif.asarray().astype(float)
            data[data == geo_head['nodata']] = np.nan
            z_mean = np.nanmean(data)
        else:
            if isinstance(polygon, np.ndarray):
                roi = geo2pixel(polygon, geo_head)   # roi = (horizontal, vertical)
                # [TODO] only dsm supported
                #imarray, offsets = imarray_clip(data, roi)
                imarray, _, _ = crop_by_coord(geotiff_path, roi, buffer=0, ts=ts, tif=tif)
                z_mean = np.nanmean(imarray)
            elif isinstance(polygon, list):
                z_mean = []
                total_num = len(polygon)
                for i, poly in enumerate(polygon):
                    if isinstance(poly, np.ndarray):
                        roi = geo2pixel(poly, geo_head)
                        #imarray, offsets = imarray_clip(data, roi)
                        imarray, _, _ = crop_by_coord(geotiff_path, roi, buffer=0, ts=ts, tif=tif)
                        z_mean.append(np.nanmean(imarray))
                    else:
                        raise TypeError('Only numpy.ndarray points itmes in the list are supported')

                    print(f"[io][geotiff][mean] Reading DSM clippers | {i+1}/{total_num}", end="\r")
            else:
                raise TypeError('Only numpy.ndarray points list are supported')

    return z_mean


def percentile_values(data, percentile=5):
    if percentile < 50:
        v = np.nanmean(data[data < np.nanpercentile(data, percentile)])
    else:
        v = np.nanmean(data[data > np.nanpercentile(data, percentile)])

    return v


def min_values(geotiff_path, polygon='all', geo_head=None, pctl=5):
    """
    :param geotiff_path:
    :param polygon:
    :param geo_head:
    :return:
    """
    with tifffile.TiffFile(geotiff_path) as tif:
        # equal to get_header()
        if geo_head is None:
            geo_head = _prase_header_string(tif.info())
        # equal to get_imarray()
        #data = tif.asarray().astype(float)
        #data[data == geo_head['nodata']] = np.nan

        # !!! a temporary modify for large DOM !!!
        ts = caas_lite.TiffSpliter(geotiff_path, 2000, 2000)
        tif = tf.TiffFile(ts.tif_path)

        if polygon == 'all':
            data = tif.asarray().astype(float)
            data[data == geo_head['nodata']] = np.nan
            z_min = percentile_values(data, pctl)
        else:
            if isinstance(polygon, np.ndarray):
                roi = geo2pixel(polygon, geo_head)   # roi = (horizontal, vertical)
                # [TODO] only dsm supported
                #imarray, offsets = imarray_clip(data, roi)
                imarray, _, _ = crop_by_coord(geotiff_path, roi, buffer=0, ts=ts, tif=tif)
                z_min = percentile_values(imarray, pctl)
            elif isinstance(polygon, list):
                z_min = []
                total_num = len(polygon)
                for i, poly in enumerate(polygon):
                    if isinstance(poly, np.ndarray):
                        roi = geo2pixel(poly, geo_head)
                        #imarray, offsets = imarray_clip(data, roi)
                        imarray, _, _ = crop_by_coord(geotiff_path, roi, buffer=0, ts=ts, tif=tif)
                        z_min.append(percentile_values(imarray, pctl))
                    else:
                        raise TypeError('Only numpy.ndarray points itmes in the list are supported')

                    print(f"[io][geotiff][mean] Reading DSM clippers | {i+1}/{total_num}", end="\r")
            else:
                raise TypeError('Only numpy.ndarray points list are supported')

    return z_min


def geo2pixel(points_hv, geo_head):
    '''
    convert point cloud xyz coordinate to geotiff pixel coordinate (horizontal, vertical)

    :param points_hv: numpy nx3 array, [x, y, z] points or nx2 array [x, y]
    :param geo_head: the geotiff head dictionary from io.geotiff.get_header() function

    :return: The ndarray pixel position of these points (horizontal, vertical)
        Please note: gis coordinate, horizontal is x axis, vertical is y axis, origin at left upper
        To clip image ndarray, the first columns is vertical pixel (along height),
            then second columns is horizontal pixel number (along width),
            the third columns is 3 or 4 bands (RGB, alpha),
            the x and y is reversed compared with gis coordinates.
            This function has already do this reverse, so that you can use the output directly.

        >>> geo_head = easyric.io.geotiff.get_header('dom_path.tiff')
        >>> gis_coord = np.asarray([(x1, y1), ..., (xn, yn)])  # x is horizonal, y is vertical
        >>> photo_ndarray = skimage.io.imread('img_path.jpg')
        (h, w, 4) ndarray  # please note the axes differences
        >>> pixel_coord = geo2pixel(gis_coord, geo_head)
        (horizontal, vertical) ndarray
        # then you can used the outputs with reverse 0 and 1 axis
        >>> region_of_interest = photo_ndarray[pixel_coord[:,1], pixel_coord[:,0], 0:3]
    '''

    gis_xmin = geo_head['tie_point'][0]
    #gis_xmax = geo_head['tie_point'][0] + geo_head['width'] * geo_head['scale'][0]
    #gis_ymin = geo_head['tie_point'][1] - geo_head['length'] * geo_head['scale'][1]
    gis_ymax = geo_head['tie_point'][1]

    gis_ph = points_hv[:, 0]
    gis_pv = points_hv[:, 1]

    # numpy_axis1 = x
    np_ax_h = (gis_ph - gis_xmin) // geo_head['scale'][0]
    # numpy_axis0 = y
    np_ax_v = (gis_ymax - gis_pv) // geo_head['scale'][1]

    pixel = np.concatenate([np_ax_h[:, None], np_ax_v[:, None]], axis=1)

    return pixel.astype(int)


def pixel2geo(points_hv, geo_head):
    '''
    convert  geotiff pixel coordinate (horizontal, vertical) to point cloud xyz coordinate (x, y, z)

    :param points_hv: numpy nx2 array, [horizontal, vertical] points
    :param geo_head: the geotiff head dictionary from io.geotiff.get_header() function

    :return: The ndarray pixel position of these points (horizontal, vertical)
    '''
    gis_xmin = geo_head['tie_point'][0]
    #gis_xmax = geo_head['tie_point'][0] + geo_head['width'] * geo_head['scale'][0]
    #gis_ymin = geo_head['tie_point'][1] - geo_head['length'] * geo_head['scale'][1]
    gis_ymax = geo_head['tie_point'][1]

    # remember the px is numpy axis0 (vertical, h), py is numpy axis1 (horizontal, w)
    pix_ph = points_hv[:, 0] + 0.5  # get the pixel center rather than edge
    pix_pv = points_hv[:, 1] + 0.5

    gis_px = gis_xmin + pix_ph * geo_head['scale'][0]
    gis_py = gis_ymax - pix_pv * geo_head['scale'][1]

    gis_geo = np.concatenate([gis_px[:, None], gis_py[:, None]], axis=1)

    return gis_geo


def _is_roi_type(roi_polygon2d):
    container = []
    if isinstance(roi_polygon2d, np.ndarray):
        container.append(roi_polygon2d)
    elif isinstance(roi_polygon2d, list):
        for poly in roi_polygon2d:
            if isinstance(poly, np.ndarray):
                container.append(poly)
            else:
                raise TypeError('Only list contains numpy.ndarray points are supported')
    else:
        raise TypeError('Only numpy.ndarray points and list contains numpy.ndarray points are supported')

    return container


def imarray_clip(imarray, polygon_hv):
    """
    clip a given ndarray image by given polygon pixel positions
    :param imarray: ndarray
    :param polygon: pixel position of boundary point, (horizontal, vertical) which reverted the imarray axis 0 to 1
    :return:
    """
    imarray_out = None

    # (horizontal, vertical) remember to revert in all the following codes
    roi_offset = polygon_hv.min(axis=0)
    roi_max = polygon_hv.max(axis=0)
    roi_length = roi_max - roi_offset

    roi_rm_offset = polygon_hv - roi_offset

    dim = len(imarray.shape)

    if dim == 2: # only has 2 dimensions, DSM 1 band only, other value outside polygon = np.nan
        roi_clipped = imarray[roi_offset[1]:roi_max[1], roi_offset[0]:roi_max[0]]

        mask = np.full(roi_clipped.shape, np.nan, dtype=np.float)
        rr, cc = polygon(roi_rm_offset[:, 1], roi_rm_offset[:, 0])
        mask[rr, cc] = 1.0

        imarray_out = roi_clipped * mask

    elif dim == 3: # has 3 dimensions, DOM with RGB or RGBA band, other value outside changed alpha layer to 0
        roi_clipped = imarray[roi_offset[1]:roi_max[1], roi_offset[0]:roi_max[0], :]
        layer_num = roi_clipped.shape[2]

        if layer_num == 3:  # DOM without alpha layer
            mask = np.zeros(roi_clipped.shape[0:2], dtype=np.uint8)
            rr, cc = polygon(roi_rm_offset[:, 1], roi_rm_offset[:, 0])
            mask[rr, cc] = 255

            # [Todo] Debug here
            imarray_out = np.concatenate([roi_clipped, mask[:, :, None]], axis=2)

        elif layer_num == 4:  # DOM with alpha layer
            mask = np.zeros(roi_clipped.shape[0:2], dtype=np.uint8)
            rr, cc = polygon(roi_rm_offset[:, 1], roi_rm_offset[:, 0])
            mask[rr, cc] = 1

            original_mask = roi_clipped[:, :, 3].copy()
            merged_mask = original_mask * mask
            #roi_clipped[:, :, 3] = mask

            imarray_out = np.dstack([roi_clipped[:,:, 0:3], merged_mask])
        else:
            raise TypeError(f'Unable to solve the layer number {layer_num}')

    return imarray_out, roi_offset


def clip_roi(roi_polygon_hv, geotiff, is_geo=False, geo_head=None):
    """
    :param roi_polygon_hv: ndarray, or polygon list,
        please do not use "for loops" outside to iterate a list of polygon, for example:
        >>> polygon_list= [poly1, poly2, ...]
        # the not properate usage:
        >>> for poly in polgon_list:
        >>> ... clip_roi(poly, dsm_path, ...)
        # the recommended usage:
        >>> clip_roi(polygon_list, dsm_path, ...)
    :param geotiff: string of geotiff file
    :param is_geo: the unit of polygon numpy, default is pixel coordinate of DOM/DSM, change to True to use as geo coordinate
    :return:
    """
    roi_list = _is_roi_type(roi_polygon_hv)

    if geo_head is None:
        geo_head = get_header(geotiff)

    if not isinstance(geotiff, str):   # give a file_path of geo_tiff
    #    dxm = get_imarray(geotiff, geo_head)
    #elif isinstance(geotiff, np.ndarray):
    #    dxm = geotiff
    #else:
        raise TypeError('Invalid geotiff type, either geotiff_file_path or read ndarray by geotiff.get_imarray() function')

    offsets = []
    imarrays = []

    # !!! a temporary modify for large DOM !!!
    ts = caas_lite.TiffSpliter(geotiff, 2000, 2000)
    tif = tf.TiffFile(ts.tif_path)

    for roi in roi_list:
        if is_geo:
            roi_pix = geo2pixel(roi, geo_head)   # (horizontal, vertical)
        else:
            roi_pix = roi

        #imarray_out, offset_out = imarray_clip(dxm, roi_geo)
        imarray_tmp, coord_np_off, offset_out = crop_by_coord(geotiff, roi_pix, buffer=0, ts=ts, tif=tif)

        imarray_out, _ = imarray_clip(imarray_tmp, coord_np_off)

        imarrays.append(imarray_out)
        offsets.append(offset_out)

    return imarrays, offsets


def crop_by_coord(dom, coord_np, buffer=0, ts=None, tif=None):
    """[summary]

    Parameters
    ----------
    dom : str
        The path to dom file
    coord_np : np.ndarray
        the coordinate of ROI, unit is image pixel, NOT geo position
    buffer : int, optional
        The boundary of ROI, by default 0
    ts : caas_lite.TiffSpliter class, optional
        by giving this, it will not generate the following variable for each iteration
        e.g. ts = caas_lite.TiffSpliter(geotiff, 2000, 2000)
    tif : tifffile.TiffFile, optional
        by giving this, it will not generate the following variable for each iteration
        tif = tf.TiffFile(ts.tif_path)

    Returns
    -------
    cropped: np.ndarray
        The cropped image data
    coord_np_off: np.ndarray
        The ROI coordinate in the cropped image data
    offset: np.ndarray
        The offset of cropped image data (up-left corner) in original DOM image
    """
    xmin, ymin = coord_np.min(axis=0)
    xmax, ymax = coord_np.max(axis=0)
    
    j0 = xmin-buffer
    i0 = ymin-buffer
    w = xmax-xmin+buffer*2
    h = ymax-ymin+buffer*2
    
    if ts is None:
        ts = caas_lite.TiffSpliter(dom, 2000, 2000)
        tif = tf.TiffFile(ts.tif_path)

    # deal with out of bound cases
    page = tif.pages[0]
    im_width = page.imagewidth
    im_height = page.imagelength
    
    i1, j1 = i0 + h, j0 + w
    if (i0 < 0) or (j0 < 0) or (i1 >= im_height) or (j1 >= im_width):
        i_st = max(i0, 0)
        j_st = max(j0, 0)
        i_ed = min(i1, im_height)
        j_ed = min(j1, im_width)

        if (i_st >= im_height) or (j_st >= im_width) or (i_ed < 0) or (j_ed < 0):
            # out of geotiff boundary, return an empty part
            print(f"[warning] The crop (i0={i0}, j0={j0}, w={w}, h={h}) is out of geotiff boundary (0, 0, {im_height}, {im_width}), cropped an empty image")
            cropped = ts._make_empty_container(h, w, layer_num=ts.img_band_num)
        else:
            # some part is in the geotiff
            h_new = i_ed-i_st
            w_new = j_ed-j_st
            img_clip = ts.get_crop(page, i_st, j_st, h_new, w_new)

            if i0 < 0:
                ioff = - i0
            else:
                ioff = 0

            if j0 < 0:
                joff = - j0
            else:
                joff = 0

            cropped = ts._make_empty_container(h, w, layer_num=ts.img_band_num)
            cropped[ioff:h_new+ioff, joff:w_new+joff,:] = img_clip
    else:
        cropped = ts.get_crop(page, i0, j0, h, w)
    
    offset = np.asarray([j0, i0])
    
    coord_np_off = coord_np - offset
    
    return cropped, coord_np_off, offset 

def save_geotiff(imarray, offset, save_path, ts, tif):
    """

    Parameters
    ----------
    imarray : [type]
        [description]
    offset : [type]
        [description]
    save_path : [type]
        [description]
    ts:
        ts = caas_lite.TiffSpliter(dom_path, 2000, 2000)
    tif : 
        tif = tf.TiffFile(ts.tif_path)
    """
    geo_corner = ts.pixel2geo(np.asarray([offset]))
    geo_x = geo_corner[0, 0]
    geo_y = geo_corner[0, 1] 

    page = tif.pages[0]

    container = []
    for k in page.tags.keys():
        if k < 30000:
            continue

        t = page.tags[k]
        if tf.__version__ < "2020.11.26" and t.dtype[0] == '1':
            dtype = t.dtype[-1]
        else:
            dtype = t.dtype

        if k == 33922:
            value = (0, 0, 0, geo_x, geo_y, 0)
        else:
            value = t.value

        container.append((t.code, dtype, t.count, value, True))

    form = save_path.split('.')[-1]

    if form == 'tif':
        # write to file
        with tf.TiffWriter(save_path) as wtif:
            wtif.save(data=imarray, software='easyidp', 
                    photometric=page.photometric, 
                    planarconfig=page.planarconfig, 
                    compress=page.compression, 
                    resolution=page.tags[33550].value[0:2], 
                    extratags=container)
    else:
        raise TypeError("only *.tif file name is supported")