import numpy as np
from skimage.external import tifffile

def point_query(geotiff_path, points):
    '''
    :param geotiff_path:
    :param points: gis points, nx3 ndarray, often x~10^5, y~10^6
    :return:
    '''
    head = get_header(geotiff_path)
    px = gis2pixel(points, head)
    with tifffile.TiffFile(geotiff_path) as tif:
        data = tif.asarray()
        height_values = data[px[:,0], px[:,1]]

    return height_values

def mean_values(geotiff_path):
    header = get_header(geotiff_path)
    with tifffile.TiffFile(geotiff_path) as tif:
        data = tif.asarray()
        data[data == header['nodata']] = np.nan
        z_mean = np.nanmean(data)

    return z_mean

def get_header(geotiff_path):
    '''
    Read geotiff header
    :param geotiff_path:
    :return:
    '''
    header = {'width': None, 'length': None, 'scale': None, 'tie_point': None, 'nodata': None}

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
        * 34735 geo_key_directory (32H) (1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 1026
        * 34737 geo_ascii_params (30s) b'WGS 84 / UTM zone 54N|WGS 84|'
        * 42113 gdal_nodata (7s) b'-10000'
        '''
        for line in tif.info().split('\n'):
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
                    x = float(line_sp[-2][:-1])
                    y = float(line_sp[-1][:-1])
                    header['tie_point'] = (x, y)
                elif code == '42113':
                    header['nodata'] = int(line_sp[-1][2:-1])
                else:
                    pass
            else:
                continue

    return header

def gis2pixel(points, geo_head):
    '''
    convert point cloud xyz coordinate to geotiff pixel coordinate

    :param points: numpy nx3 array, [x, y, z] points
    :param geo_head: the geotiff head dictionary from io.geotiff.get_header() function

    === optional ===
    the GIS coordinate often around x~10^5, y~10^6,
    if points around 0 ± plot size (m), means the point coordinate is not GIS location,
    which is shifted by offset file (pix4d software)
    need use this offset to correct back

    :param offset_x: float number, can be obtained from io.pix4d.read_xyz() function
    :param offset_y: float number, can be obtained from io.pix4d.read_xyz() function
    :return:
    '''

    gis_xmin = geo_head['tie_point'][0]
    gis_xmax = geo_head['tie_point'][0] + geo_head['width'] * geo_head['scale'][0]
    gis_ymin = geo_head['tie_point'][1] - geo_head['length'] * geo_head['scale'][1]
    gis_ymax = geo_head['tie_point'][1]

    gis_px = points[:, 0]
    gis_py = points[:, 1]

    # numpy_axis1 = x
    np_ax1 = (gis_px - gis_xmin) // geo_head['scale'][0]
    # numpy_axis0 = y
    np_ax0 = (gis_ymax - gis_py) // geo_head['scale'][1]

    pixel = np.concatenate([np_ax0[:, None], np_ax1[:, None]], axis=1)

    return pixel.astype(int)

def pixel2gis(points, geo_head):
    gis_xmin = geo_head['tie_point'][0]
    gis_xmax = geo_head['tie_point'][0] + geo_head['width'] * geo_head['scale'][0]
    gis_ymin = geo_head['tie_point'][1] - geo_head['length'] * geo_head['scale'][1]
    gis_ymax = geo_head['tie_point'][1]

    return "still under construction"