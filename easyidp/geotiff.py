import os
import pyproj
import numpy as np
import tifffile as tf
import warnings

from pyproj.exceptions import CRSError
from .cvtools import poly2mask
from . import __version__


class GeoTiff(object):

    def __init__(self, tif_path="") -> None:
        self.file_path = os.path.abspath(tif_path)
        self.header = None

        if len(tif_path) > 0:
            self.read_geotiff(tif_path)
            
    def read_geotiff(self, tif_path):
        """Open and get the meta information (header) from geotiff

        Parameters
        ----------
        tif_path : str
            the path to geotiff file
        """
        if os.path.exists(tif_path):
            self.file_path = os.path.abspath(tif_path)
            self.header = get_header(tif_path)
        else:
            warnings.warn(f"Can not find file [{tif_path}], skip loading")

    def not_empty(func):
        # the decorator to check before doing functions
        def wrapper(self, *args, **kwargs):
            if self.header is None or not os.path.exists(self.file_path):
                raise FileNotFoundError("Could not operate if not specify correct geotiff file")
            return func(*args, **kwargs)

        return wrapper

    @not_empty
    def crop(self, roi, save_folder=None):  # clip_roi
        # roi: ROI class, with several roi lists
        pass

    @not_empty
    def _crop_one_polygon(self, polygon_hv, save_path=None):
        # roi = single polygon

        # if save_path is not None and is file path:
        #     _save_geotiff()
        pass

    @not_empty
    def save_geotiff(self, imarray, pix_offset_hv, save_path):
        geo_corner = pixel2geo(np.asarray([pix_offset_hv]), self.header)
        geo_h = geo_corner[0, 0]
        geo_v = geo_corner[0, 1]

        model_tie_point = (0, 0, 0, geo_h, geo_v, 0)

        extratags = []
        for k, t in self.header["tags"].items():
            '''
            TiffTag 256 ImageWidth @10 SHORT @18 = 5490
            TiffTag 257 ImageLength @22 SHORT @30 = 5752
            TiffTag 258 BitsPerSample @34 SHORT[4] @230 = (8, 8, 8, 8)
            TiffTag 259 Compression @46 SHORT @54 = LZW
            TiffTag 262 PhotometricInterpretation @58 SHORT @66 = RGB
            TiffTag 273 StripOffsets @70 LONG[5752] @23246 = (46439, 46678, 46934, 47207, 4
            TiffTag 277 SamplesPerPixel @82 SHORT @90 = 4
            TiffTag 278 RowsPerStrip @94 SHORT @102 = 1
            TiffTag 279 StripByteCounts @106 LONG[5752] @238 = (239, 256, 273, 278, 296, 30
            TiffTag 284 PlanarConfiguration @118 SHORT @126 = CONTIG
            TiffTag 305 Software @130 ASCII[12] @46262 = pix4dmapper
            TiffTag 317 Predictor @142 SHORT @150 = HORIZONTAL
            TiffTag 338 ExtraSamples @154 SHORT @162 = (<EXTRASAMPLE.UNASSALPHA: 2>,)
            TiffTag 339 SampleFormat @166 SHORT[4] @46254 = ('UINT', 'UINT', 'UINT', 'UINT'
            TiffTag 33550 ModelPixelScaleTag @178 DOUBLE[3] @46274 = (0.00738, 0.00738, 0.0
            TiffTag 33922 ModelTiepointTag @190 DOUBLE[6] @46298 = (0.0, 0.0, 0.0, 368014.5
            TiffTag 34735 GeoKeyDirectoryTag @202 SHORT[32] @46346 = (1, 1, 0, 7, 1024, 0,
            TiffTag 34737 GeoAsciiParamsTag @214 ASCII[29] @46410 = WGS84 / UTM zone 54N|WG
            '''
            if k < 30000:
                # this will be automatically added by wtif.save(data=imarray)
                # the key of this step is extract "hidden" tags
                continue

            if tf.__version__ < "2020.11.26" and t.dtype[0] == '1':
                dtype = t.dtype[-1]
            else:
                dtype = t.dtype

            # <tifffile.TiffTag 33922 ModelTiepointTag @190>
            if k == 33922:
                # replace the value for this tag
                value = (0, 0, 0, geo_h, geo_v, 0)
            else:
                # other just using parent value.
                value = t.value

            extratags.append((t.code, dtype, t.count, value, True))

        if os.path.splitext(save_path) == ".tif":
            # write geotiff
            with tf.TiffWriter(save_path) as wtif:
                wtif.save(data=imarray, 
                          software=f"EasyIDP {__version__}", 
                          photometric=self.header["photometric"], 
                          planarconfig=self.header["planarconfig"], 
                          compress=self.header["compression"], 
                          resolution=self.header["scale"], 
                          extratags=extratags)
        else:
            raise TypeError("only *.tif file name is supported")


def get_header(tif_path):
    """Read the necessary meta infomation from TIFF file

    Parameters
    ----------
    tif_path : str
        the path to the geotiff file

    Returns
    -------
    header: dict
        the container of acquired meta info
    """
    with tf.TiffFile(tif_path) as tif:
        header = {}
        # keys: 'width', 'height', 'dim', 'scale', 'tie_point',
        #       'nodata', 'proj', 'dtype', 'band_num', 
        # for export:
        #       'tags', 'photometric', 'planarconfig', 'compression'
        page = tif.pages[0]

        header["height"] = page.shape[0]
        header["width"] = page.shape[1]
        if len(page.shape) > 2:
            # header["dim"] = page.shape[2] 
            # `band_num` used in other functions in the old version
            header["dim"] = page.samplesperpixel
        else:
            header["dim"] = 1
        header["nodata"] = page.nodata
        header["dtype"] = page.dtype

        # for save geotiff
        header["tags"] = page.tags
        header["photometric"] = page.photometric
        header["planarconfig"] = page.planarconfig
        header["compress"] = page.compression
        
        # page.geotiff_tags
        # -> 'ModelPixelScale': [0.0034900000000000005, 0.0034900000000000005, 0.0]
        header["scale"] = page.geotiff_tags["ModelPixelScale"][0:2]
        
        # page.geotiff_tags
        # -> 'ModelTiepoint': [0.0, 0.0, 0.0, 419509.89816000004, 3987344.8286, 0.0]
        header["tie_point"] = page.geotiff_tags["ModelTiepoint"][3:5]
        
        # pix4d:
        #    page.geotiff_tags
        #    -> 'GTCitationGeoKey': 'WGS 84 / UTM zone 54N'
        if "GTCitationGeoKey" in page.geotiff_tags.keys():
            proj_str = page.geotiff_tags["GTCitationGeoKey"]
        # metashape:
        #     page.geotiff_tags
        #     -> 'PCSCitationGeoKey': 'WGS 84 / UTM zone 54N'
        elif "PCSCitationGeoKey" in page.geotiff_tags.keys():
            proj_str = page.geotiff_tags["PCSCitationGeoKey"]
        else:
            raise KeyError("Can not find key 'GTCitationGeoKey' or 'PCSCitationGeoKey' in Geotiff tages")
        
        try:
            proj = pyproj.CRS.from_string(proj_str)
            header['proj'] = proj
        except CRSError as e:
            print(f'[io][geotiff][GeoCorrd] Generation failed, because [{e}], but you can manual specify it later by \n'
                    '>>> import pyproj \n'
                    '>>> proj = pyproj.CRS.from_epsg() # or from_string() or refer official documents:\n'
                    'https://pyproj4.github.io/pyproj/dev/api/crs/coordinate_operation.html')
            pass

    return header


def get_imarray(tif_path):
    """Read full map data as numpy array (time and RAM costy, not recommended)

    Parameters
    ----------
    tif_path : str
        the path to geotiff file

    Returns
    -------
    data: ndarray
        the obtained image data
    """
    with tf.TiffFile(tif_path) as tif:
        data = tif.pages[0].asarray()

    return data


def geo2pixel(points_hv, header, return_index=False):
    """convert point cloud xyz coordinate to geotiff pixel coordinate (horizontal, vertical)

    Parameters
    ----------
    points_hv : numpy nx3 array
        [horizontal, vertical] points
    header : dict
        the geotiff head dictionary from get_header()

    Returns
    -------
    The ndarray pixel position of these points (horizontal, vertical)
        Please note: gis coordinate, horizontal is x axis, vertical is y axis, origin at left upper
        To crop image ndarray, the first columns is vertical pixel (along height),
            then second columns is horizontal pixel number (along width),
            the third columns is 3 or 4 bands (RGB, alpha),
            the x and y is reversed compared with gis coordinates.
            This function has already do this reverse, so that you can use the output directly.

    Examples
    --------
    >>> geo_head = easyric.io.geotiff.get_header('dom_path.tiff')
    >>> gis_coord = np.asarray([(x1, y1), ..., (xn, yn)])  # x is horizonal, y is vertical
    >>> photo_ndarray = skimage.io.imread('img_path.jpg')
    (h, w, 4) ndarray  # please note the axes differences
    >>> pixel_coord = geo2pixel(gis_coord, geo_head)
    (horizontal, vertical) ndarray
    # then you can used the outputs with reverse 0 and 1 axis
    >>> region_of_interest = photo_ndarray[pixel_coord[:,1], pixel_coord[:,0], 0:3]
    """
    gis_xmin = header['tie_point'][0]
    gis_ymax = header['tie_point'][1]

    gis_ph = points_hv[:, 0]
    gis_pv = points_hv[:, 1]


    # get float coordinate on pixels
    # - numpy_axis1 = x
    np_ax_h = (gis_ph - gis_xmin) / header['scale'][0]
    # - numpy_axis0 = y
    np_ax_v = (gis_ymax - gis_pv) / header['scale'][1]

    # get the pixel index (int)
    if return_index:  
        np_ax_h = np.floor(np_ax_h).astype(int)
        np_ax_v = np.floor(np_ax_v).astype(int)

    pixel = np.vstack([np_ax_h, np_ax_v]).T

    return pixel


def pixel2geo(points_hv, header):
    """convert geotiff pixel coordinate (horizontal, vertical) to point cloud xyz coordinate (x, y, z)

    Parameters
    ----------
    points_hv : numpy nx2 array
        [horizontal, vertical] points
    geo_head : dict
        the geotiff head dictionary from get_header()

    Returns
    -------
    The ndarray pixel position of these points (horizontal, vertical)
    """
    gis_xmin = header['tie_point'][0]
    gis_ymax = header['tie_point'][1]

    # the px is numpy axis0 (vertical, h)
    # py is numpy axis1 (horizontal, w)
    if np.issubdtype(points_hv.dtype, np.integer):
        # all integer possible means the pixel index 
        #    rather than specific coordinates
        # +0.5 to get the pixel center rather than edge
        pix_ph = points_hv[:, 0] + 0.5  
        pix_pv = points_hv[:, 1] + 0.5
    elif np.issubdtype(points_hv.dtype, np.floating):
        # all floats possible means it is the pixel coordinates
        #    rather than pixel index
        # no need to +0.5 as image center
        pix_ph = points_hv[:, 0]
        pix_pv = points_hv[:, 1]
    else:
        raise TypeError(f"The `points_hv` only accept numpy ndarray integer and float types")

    gis_px = gis_xmin + pix_ph * header['scale'][0]
    gis_py = gis_ymax - pix_pv * header['scale'][1]

    gis_geo = np.vstack([gis_px, gis_py]).T

    return gis_geo


def tifffile_crop(page, top, left, h, w):  
    """
    Extract a crop from a TIFF image file directory (IFD).

    Only the tiles englobing the crop area are loaded and not the whole page.
    This is usefull for large Whole slide images that can't fit int RAM.

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

    Modified from: 
    https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743#file-get_crop-py-L60

    Previous version: 
    caas_lite.get_crop(page, i0, j0, h, w)
    
    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    top, left: int
        Coordinates of the top left corner of the desired crop.
        top = i0 = height_st
        left = j0 = w_st
    h: int
        Desired crop height.
    w: int
        Desired crop width.
        
    Returns
    -------
    out : ndarray of shape (h, w, sampleperpixel)
        Extracted crop.""
    """
    if page.is_tiled:
        out = _get_tiled_crop(page, top, left, h, w)
    else:
        out = _get_untiled_crop(page, top, left, h, w)

    return out


def _get_tiled_crop(page, i0, j0, h, w):
    """
    The submodule of self.get_crop() for those tiled geotiff

    Copied from: 
    https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743#file-get_crop-py-L60
    """
    if not page.is_tiled:
        raise ValueError("Input page must be tiled")

    im_width = page.imagewidth
    im_height = page.imagelength
    im_pyramid = page.imagedepth
    im_dim = page.samplesperpixel

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")
        
    i1, j1 = i0 + h, j0 + w
    if i0 < 0 or j0 < 0 or i1 >= im_height or j1 >= im_width:
        raise ValueError(f"Requested crop area [({i0}, {i1}), ({j0}, {j1})] is out of image bounds ({im_height}, {im_width})")

    tile_width, tile_height = page.tilewidth, page.tilelength

    tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
    tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(im_width / tile_width))

    # older version: (img_depth, h, w, dim)
    out = np.empty((im_pyramid,
                    (tile_i1 - tile_i0) * tile_height,
                    (tile_j1 - tile_j0) * tile_width,
                    im_dim), dtype=page.dtype)

    fh = page.parent.filehandle

    for i in range(tile_i0, tile_i1):
        for j in range(tile_j0, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index)

            im_i = (i - tile_i0) * tile_height
            im_j = (j - tile_j0) * tile_width
            out[:, im_i: im_i + tile_height, im_j: im_j + tile_width, :] = tile

    im_i0 = i0 - tile_i0 * tile_height
    im_j0 = j0 - tile_j0 * tile_width

    # old version: out[:, im_i0: im_i0 + h, im_j0: im_j0 + w, :]
    return out[0, im_i0: im_i0 + h, im_j0: im_j0 + w, :]

def _get_untiled_crop(page, i0, j0, h, w):
    """
    The submodule of self.get_crop(), for those untiled geotiff

    Copied from: 
    https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743#file-get_crop-py-L60
    """
    if page.is_tiled:
        raise ValueError("Input page must not be tiled")

    im_width = page.imagewidth
    im_height = page.imagelength
    im_pyramid = page.imagedepth
    im_dim = page.samplesperpixel

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    i1, j1 = i0 + h, j0 + w
    if i0 < 0 or j0 < 0 or i1 >= im_height or j1 >= im_width:
        raise ValueError(f"Requested crop area [({i0}, {i1}), ({j0}, {j1})] is out of image bounds ({im_height}, {im_width})")
    
    fh = page.parent.filehandle

    # for data/pix4d/maize_tanashi dom:
    # dom shape: (722, 836)
    # >>> page.dataoffsets
    # (2554, 9242, 15930, 22618,  ...)  # 6688 interval
    # >>> page.databytecounts
    # (6688, 6688, 6688, 6688,  ...)
    # >>> len(page.dataoffsets)
    # 361  # = dom.h / 2 -> read 2 row once
    row_step = im_height / len(page.dataoffsets)

    # judge whether can be perfect divided
    if row_step % 1 == 0.0:
        row_step = int(row_step)
    else:
        raise InterruptedError(f"img.ht={im_height} with offset number {len(page.dataoffsets)}, {row_step} row per read, not a integer")

    # commonly, it read 1 row once,
    # no need to do extra things
    if row_step == 1: 
        read_tile_idx = np.arange(i0, i1)

    # sometime it not read once per row, it reads two rows once, etc...
    # e.g. tile.shape = (1, 2, full_width, channel_num)
    #      expected ->  (1, 1, full_width, channel_num)
    # if still just picking the first row, will
    # result in a horzontal zoom 50% result
    else:
        # recalculate the row id to read
        # >>> idx = array([10, 11, 12, 13, 14])   # orignal line index
        # >>> idx_dv = np.floor(idx / row_step).astype(int)
        # array([5, 5, 6, 6, 7])
        # >>> np.unique(idx_dv)
        # array([5, 6, 7])   -> known which index to read
        read_row_id = np.arange(i0, i1)
        read_tile_idx = np.unique(np.floor(read_row_id / row_step).astype(int))

        # Then decide the start line and end line
        # >>> idx % 2
        # array([0, 1, 0, 1, 0], dtype=int32)
        # # get start id
        st_line_id = read_row_id[0] % row_step
        # # get reversed id
        # line id
        #  3  |  |  |  |...
        #  -------------------  last read
        #  0  |  |  |  |
        #  1  |  |  |  |...
        #  2  |  |  |  |
        #  -> get the [:, 0], [:, -1] or [:, -2] index
        #     as the `ed_line_id`
        ed_line_id = read_row_id[-1] % row_step - (row_step-1)

    # crop them vertically first, then crop
    temp_out = np.empty((im_pyramid, 0, w, im_dim), dtype=page.dtype)
    for index in read_tile_idx:
        offset = page.dataoffsets[index]
        bytecount = page.databytecounts[index]

        fh.seek(offset)
        data = fh.read(bytecount)

        tile, indices, shape = page.decode(data, index)

        # double check if is row_step per read
        # shape -> (1, 2, full_width, channel_num)
        if shape[1] != row_step:
            raise LookupError(f"the calculated {row_step} row per read does not match {shape[1]} of tifffile decoded.")

        temp_out = np.concatenate([temp_out, tile[:, :,j0:j1,:]], axis=1)

    # return cropped result
    if row_step == 1:
        if im_dim == 1:
            # is dsm -> shape (w, h)
            return temp_out[0,:,:,0]
        else:  
            # is dom -> shape (w, h, d)         
            return temp_out[0,:,:,0:im_dim] 
    else:
        if ed_line_id == 0:
            if im_dim == 1:  
                # is dsm -> shape (w, h)
                return temp_out[0, st_line_id:, :, 0]
            else:
                # is dom -> shape (w, h, d)
                return temp_out[0, st_line_id:, :, 0:im_dim]
        else:
            if im_dim == 1:
                # is dsm -> shape (w, h)
                return temp_out[0, st_line_id:ed_line_id, :, 0]
            else:
                # is dom -> shape (w, h, d)
                return temp_out[0, st_line_id:ed_line_id, :, 0:im_dim]


def point_query(page, points_hv, header=None):
    """get the pixel value of given point(s)

    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    points_hv : tuple | list | nx2 ndarray
        1. one point tuple
            e.g. (34.57, 45.62)
        2. one point list
            e.g. [34.57, 45.62]
        3. points lists
            e.g. [[34.57, 45.62],[35.57, 46.62]]
        4. 2d numpy array
            e.g. np.array([[34.57, 45.62],[35.57, 46.62]])
    header : dict, optional
        the geotiff head dictionary from get_header()
        if specified, will view the `points_hv` as geo position
            e.g. [longtitude, latitude]
        if not specified, will view as pixel index
            e.g. [1038, 567] -> pixel id

    Returns
    -------
    values: ndarray
        the obtained pixel value (RGB or height) 
    """

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

    # convert to pixel index
    if header is None:   # means point hv is pixel id
        # check if is integer
        if np.issubdtype(points_hv.dtype, np.integer):
            px = points_hv
        # if float, converted to int by floor()
        else:
            px = np.floor(points_hv).astype(int)
            warnings.warn("The given pixel coordinates is not integer and is converted, if it is geo_coordinate, please specfiy `header=get_header()`")
    else:
        px = geo2pixel(points_hv, header, return_index=True)

    # get values
    # - prepare container
    values_list = []
    for p in px:
        '''
        if dom:
            cropped.shape = (1, 1, 4)
            cropped -> array([[[0, 0, 0, rgba]]], dtype=uint8)
            wanted = cropped[0,0,:]
        if dsm:
            cropped.shape = (1, 1, 1)
            cropped -> array([[-10000.]], dtype=float32)
            wanted = cropped[0,0]
        '''
        cropped = tifffile_crop(page, top=p[1], left=p[0], h=1, w=1)

        if page.samplesperpixel == 1:
            values_list.append(cropped[0,0])
        else:
            values_list.append(cropped[0,0,:])

    return np.array(values_list)

def imarray_crop(imarray, polygon_hv, empty_value=0):
    """crop a given ndarray image by given polygon pixel positions

    Parameters
    ----------
    imarray : ndarray
        the image data, shape = (height,width)
    polygon_hv : ndarray
        pixel position of boundary point, (horizontal, vertical) which reverted the imarray axis 0 to 1
    empty_value: int | float
        for 

    returns
    -------
    imarray_out : ndarray
    roi_offset : ndarray
    """
    # (horizontal, vertical) remember to revert in all the following codes
    roi_offset = polygon_hv.min(axis=0)
    roi_max = polygon_hv.max(axis=0)
    roi_length = roi_max - roi_offset

    roi_rm_offset = polygon_hv - roi_offset
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
    
    dim = len(imarray.shape)
    if dim == 2: 
        # only has 2 dimensions
        # e.g. DSM 1 band only, other value outside polygon = empty value
        
        # here need to reverse 
        # imarray.shape -> (h, w), but poly2mask need <- (w, h)
        roi_clipped = imarray[roi_offset[1]:roi_max[1], 
                              roi_offset[0]:roi_max[0]]
        rh = roi_clipped.shape[0]
        rw = roi_clipped.shape[1]
        mask = poly2mask((rw, rh), roi_rm_offset)

        roi_clipped[~mask] = empty_value
        imarray_out = roi_clipped

    elif dim == 3: 
        # has 3 dimensions
        # e.g. DOM with RGB or RGBA band, other value outside changed alpha layer to 0
        roi_clipped = imarray[roi_offset[1]:roi_max[1], roi_offset[0]:roi_max[0], :]

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

    return imarray_out, roi_offset

def _make_empty_imarray(header, h, w, layer_num=None):
    """
    Produce a empty image, suit the requirement for nodata
    """
    # possible dsm with only one band
    if header["dim"] == 1:
        
        # old version: np.ones((self.img_depth, h, w, 1))
        empty_template = np.ones((h, w)) * header["nodata"]
        
    # possible RGB band
    elif header["dim"] == 3 and header["dtype"] == np.uint8:
        if layer_num == 4:
            # old version: np.ones((self.img_depth, h, w, 1))
            empty_template = np.ones((h, w, 4)).astype(np.uint8) * 255
            empty_template[:,:,3] = empty_template[:,:,3] * 0
        else:
            # old version: np.ones((self.img_depth, h, w, 1))
            empty_template = np.ones((h, w, 3)).astype(np.uint8) * 255
        
    # possible RGBA band, empty defined by alpha = 0
    elif header["dim"] == 4 and header["dtype"] == np.uint8:
        # old version: np.ones((h, w, 1))
        empty_template = np.ones((h, w, 4)).astype(np.uint8) * 255
        empty_template[:,:,3] = empty_template[:,:,3] * 0
    else:
        raise ValueError(f"Current version only support DSM, RGB and RGBA images (band expect: 1,3,4; get [{header['dim']}], dtype=np.uint8; get [{header['dtype']}])")
        
    return empty_template

def _is_empty_imarray(header, imarray):
    """
    Judge if current img_array is empty grids
        e.g. dsm=-10000, # (page.nodata)
             rgb=[255,255,255], [0,0,0] # pure white and black
             or RGBA with full empty alpha layer  # alpha all=0
    
    Parameters
    ----------
    img_array: np.ndarray
        the outputs of self.get_crop()
    
    Returns
    -------
    bool: True is empty image.
    
    """
    is_empty = False
    if len(imarray.shape) == 2:
        imarray_dim = 1
    elif len(imarray.shape) == 3:
        _, _, imarray_dim = imarray.shape

    if imarray_dim != header["dim"]:
        raise IndexError(f"The imarray dimention [{imarray_dim}] does not match with header dimention [{header['dim']}]")
    
    if header["dim"] == 1:
        if np.all(imarray==header["nodata"]):
            is_empty = True
    elif header["dim"] == 3:
        # for the case that use (255, 255, 255) white as background
        if np.all(imarray==255):
            is_empty = True
        # in case some use (0, 0, 0) black as background
        if np.all(imarray==0):
            is_empty = True
        # in case some header specify nodata rather than 255 and 0
        if np.all(imarray==header["nodata"]):
            is_empty = True
    # for those RGBA with alpha layers, assume alpha=0 as empty
    elif header["dim"] == 4:
        if np.all(imarray[:,:,3]==0):
            is_empty = True
    else:
        raise ValueError(f"Current version only support DSM, RGB and RGBA images (band expect: 1,3,4; get [{header['dim']}], dtype=np.uint8; get [{header['dtype']}])")

    return is_empty