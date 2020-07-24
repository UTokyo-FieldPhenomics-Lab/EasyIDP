import pyproj
import pytest
import numpy as np
from easyric.io import geotiff, shp
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def test_prase_header_string_width():
    out_dict = geotiff._prase_header_string("* 256 image_width (1H) 13503")
    assert out_dict['width'] == 13503


def test_prase_header_string_length():
    out_dict = geotiff._prase_header_string("* 257 image_length (1H) 19866")
    assert out_dict['length'] == 19866


def test_prase_header_string_scale():
    in_str = "* 33550 model_pixel_scale (3d) (0.0029700000000000004, 0.0029700000000000004, 0"
    out_dict = geotiff._prase_header_string(in_str)
    assert out_dict['scale'] == (0.0029700000000000004, 0.0029700000000000004)


def test_prase_header_string_tie_point():
    in_str = "* 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823,"
    out_dict = geotiff._prase_header_string(in_str)
    assert out_dict['tie_point'] == (368090.77975000005, 3956071.13823)

    in_str = "* 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 368090.77975000005, 3956071.13823, 0"
    out_dict = geotiff._prase_header_string(in_str)
    assert out_dict['tie_point'] == (368090.77975000005, 3956071.13823)


def test_prase_header_string_nodata():
    out_dict = geotiff._prase_header_string("* 42113 gdal_nodata (7s) b'-10000'")
    assert out_dict['nodata'] == -10000


def test_prase_header_string_proj_normal(capsys):
    in_str = "* 34737 geo_ascii_params (30s) b'WGS 84 / UTM zone 54N|WGS 84|'"
    out_dict = geotiff._prase_header_string(in_str)
    captured = capsys.readouterr()

    assert f"[io][geotiff][GeoCorrd] Comprehense [{in_str}]" in captured.out
    assert out_dict['proj'] == pyproj.CRS.from_epsg(32654)


def test_prase_header_string_proj_error(capsys):
    # should raise error because WGS 84 / UTM ... should be full
    out_dict = geotiff._prase_header_string("* 34737 geo_ascii_params (30s) b'UTM zone 54N|WGS 84|'")
    captured = capsys.readouterr()
    assert '[io][geotiff][GeoCorrd] Generation failed, because [Input is not a CRS: UTM zone 54N]' in  captured.out
    assert out_dict['proj'] == None


def test_get_imarray_without_header(capsys):
    pass


def test_get_imarray_with_header(capsys):
    pass


def test_point_query_one_point():
    point = (368023.004, 3955500.669)
    out = geotiff.point_query(r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif', point)
    np.testing.assert_almost_equal(out, np.float32(97.45558), decimal=3)


def test_point_query_numpy_points():
    points = np.asarray([[368022.581, 3955501.054], [368024.032, 3955500.465]])
    out = geotiff.point_query(r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif', points)
    expected = np.asarray([97.624344, 97.59617])

    np.testing.assert_almost_equal(out, expected, decimal=3)


def test_point_query_list_numpy_points():
    points = np.asarray([[368022.581, 3955501.054], [368024.032, 3955500.465]])
    point = np.asarray([[368023.004, 3955500.669]])
    p_list = [point, points]

    expected = [np.asarray([97.45558]), np.asarray([97.624344, 97.59617])]
    out = geotiff.point_query(r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif', p_list)

    assert type(expected) == type(out)
    np.testing.assert_almost_equal(expected[0], out[0], decimal=3)
    np.testing.assert_almost_equal(expected[1], out[1], decimal=3)

def test_point_query_wrong_types():
    # [TODO]
    pass

def test_mean_values(capsys):
    mean_ht = geotiff.mean_values(r'file/pix4d.diy/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif')
    captured = capsys.readouterr()
    # When not convert to float, mean_values = 97.562584
    # assert mean_ht == np.float32(97.562584)
    np.testing.assert_almost_equal(mean_ht, np.float32(97.562584), decimal=3)

    # another case that not working in previous version:
    # Cannot convert np.nan to int, fixed by astype(float)
    mean_ht = geotiff.mean_values(r'file/tiff_test/2_12.tif')
    captured = capsys.readouterr()
    np.testing.assert_almost_equal(mean_ht, np.float(72.31657466298653), decimal=3)

def test_gis2pixel2gis():
    geo_head_txt = """
TIFF file: 200423_G_M600pro_transparent_mosaic_group1.tif, 411 MiB, little endian, bigtiff

Series 0: 31255x19436x4, uint8, YXS, 1 pages, not mem-mappable

Page 0: 31255x19436x4, uint8, 8 bit, rgb, lzw
* 256 image_width (1H) 19436
* 257 image_length (1H) 31255
* 258 bits_per_sample (4H) (8, 8, 8, 8)
* 259 compression (1H) 5
* 262 photometric (1H) 2
* 273 strip_offsets (31255Q) (500650, 501114, 501578, 502042, 502506, 502970, 5
* 277 samples_per_pixel (1H) 4
* 278 rows_per_strip (1H) 1
* 279 strip_byte_counts (31255Q) (464, 464, 464, 464, 464, 464, 464, 464, 464, 
* 284 planar_configuration (1H) 1
* 305 software (12s) b'pix4dmapper'
* 317 predictor (1H) 2
* 338 extra_samples (1H) 2
* 339 sample_format (4H) (1, 1, 1, 1)
* 33550 model_pixel_scale (3d) (0.001, 0.001, 0.0)
* 33922 model_tie_point (6d) (0.0, 0.0, 0.0, 484576.70205, 3862285.5109300003, 
* 34735 geo_key_directory (32H) (1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 1026
* 34737 geo_ascii_params (30s) b'WGS 84 / UTM zone 53N|WGS 84|'
"""
    gis_coord = np.asarray([[ 484593.67474654, 3862259.42413431],
                            [ 484593.41064743, 3862259.92582402],
                            [ 484593.64841806, 3862260.06515117],
                            [ 484593.93077419, 3862259.55455913],
                            [ 484593.67474654, 3862259.42413431]])

    header = geotiff._prase_header_string(geo_head_txt)

    expected_pixel = np.asarray([[16972, 26086],
                                 [16708, 25585],
                                 [16946, 25445],
                                 [17228, 25956],
                                 [16972, 26086]])

    pixel_coord = geotiff.geo2pixel(gis_coord, header)

    np.testing.assert_almost_equal(pixel_coord, expected_pixel)

    gis_revert = geotiff.pixel2geo(pixel_coord, header)

    np.testing.assert_almost_equal(gis_revert, gis_coord, decimal=3)


def test_is_roi_type():
    roi1 = np.asarray([[123, 456], [456, 789]])
    roi2 = [roi1, roi1]

    roi_wrong_1 = (123, 345)
    roi_wrong_2 = [123, 456]
    roi_wrong_3 = [[123, 345], [456, 789]]

    roi1_out = geotiff._is_roi_type(roi1)
    assert roi1_out == [roi1]

    roi2_out = geotiff._is_roi_type(roi2)
    assert roi2_out == roi2

    with pytest.raises(TypeError) as errinfo:
        roi_w1_out = geotiff._is_roi_type(roi_wrong_1)
        assert 'Only numpy.ndarray points and list contains numpy.ndarray points are supported' in str(errinfo.value)

    with pytest.raises(TypeError) as errinfo:
        roi_w2_out = geotiff._is_roi_type(roi_wrong_2)
        assert 'Only list contains numpy.ndarray points are supported' in str(errinfo.value)

    with pytest.raises(TypeError) as errinfo:
        roi_w3_out = geotiff._is_roi_type(roi_wrong_3)
        assert 'Only list contains numpy.ndarray points are supported' in str(errinfo.value)


def test_imarray_clip_2d_rgb_rgba():
    photo_path = 'file/pix4d.diy/photos/DJI_0174.JPG'
    roi = np.asarray([[2251, 1223], [2270, 1270], [2227, 1263], [2251, 1223]])

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    # -----------------------------------------------
    imarray_rgb = imread(photo_path)
    assert imarray_rgb.shape == (3456, 4608, 3)

    im_out_rgb, offsets_rgb = geotiff.imarray_clip(imarray_rgb, roi)

    ax[1].imshow(im_out_rgb / 255)
    ax[1].set_title('rgb')

    # -----------------------------------------------
    imarray_2d = rgb2gray(imarray_rgb)
    assert imarray_2d.shape == (3456, 4608)

    im_out_2d, offsets_2d = geotiff.imarray_clip(imarray_2d, roi)

    ax[0].imshow(im_out_2d, cmap='gray')
    ax[0].set_title('gray')

    # -----------------------------------------------
    imarray_rgba = np.dstack((imarray_rgb, np.ones((3456, 4608)) * 255))
    assert imarray_rgba.shape == (3456, 4608, 4)

    im_out_rgba, offsets = geotiff.imarray_clip(imarray_rgba, roi)
    ax[2].imshow(im_out_rgba/255)
    ax[2].set_title('rgba')

    plt.show()


def test_clip_roi_pixel():
    poly = shp.read_shp2d('file/shp_test/test.shp')
    poly_pixel = geotiff.geo2pixel(poly['0'], geotiff.get_header('file/tiff_test/2_12.tif'))
    imarray, offset = geotiff.clip_roi(poly_pixel, 'file/tiff_test/2_12.tif', is_geo=False)
    assert len(imarray) == 1


def test_clip_roi_geo():
    poly = shp.read_shp2d('file/shp_test/test.shp')
    imarray, offset = geotiff.clip_roi(poly['0'], 'file/tiff_test/2_12.tif', is_geo=True)
    assert len(imarray) == 1