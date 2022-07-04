import os
import pytest
import pyproj
import re
import tifffile as tf
import numpy as np
import easyidp as idp
import warnings
import matplotlib.pyplot as plt


data_path =  "./tests/data/tiff_test"

lotus_full_dom = "./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"
lotus_full_dsm = "./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"

lotus_part_dom = "./tests/data/pix4d/lotus_tanashi/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"
lotus_part_dsm = "./tests/data/pix4d/lotus_tanashi/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"

maize_part_dom = "./tests/data/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/3_dsm_ortho/2_mosaic/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_transparent_mosaic_group1.tif"
maize_part_dsm = "./tests/data/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/3_dsm_ortho/1_dsm/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_dsm.tif"


def test_get_header():
    lotus_full = idp.geotiff.get_header(lotus_full_dom)
    assert lotus_full["width"] == 5490
    assert lotus_full["height"] == 5752
    assert lotus_full["dim"] == 4
    assert lotus_full["nodata"] == 0
    assert lotus_full["proj"].name == "WGS 84 / UTM zone 54N"
    assert lotus_full["scale"][0] == 0.00738
    assert lotus_full["scale"][1] == 0.00738
    assert lotus_full["tie_point"][0] == 368014.54157
    assert lotus_full["tie_point"][1] == 3955518.2747700005

    lotus_full = idp.geotiff.get_header(lotus_full_dsm)
    assert lotus_full["width"] == 5490
    assert lotus_full["height"] == 5752
    assert lotus_full["dim"] == 1
    assert lotus_full["nodata"] == -10000.0
    assert lotus_full["proj"].name == "WGS 84 / UTM zone 54N"
    assert lotus_full["scale"][0] == 0.00738
    assert lotus_full["scale"][1] == 0.00738
    assert lotus_full["tie_point"][0] == 368014.54157
    assert lotus_full["tie_point"][1] == 3955518.2747700005

    lotus_part = idp.geotiff.get_header(lotus_part_dom)
    assert lotus_part["width"] == 437
    assert lotus_part["height"] == 444
    assert lotus_part["proj"].name == "WGS 84 / UTM zone 54N"
    assert lotus_part["tie_point"][0] == 368024.0839
    assert lotus_part["tie_point"][1] == 3955479.7512


def test_get_imarray():
    maize_part_np = idp.geotiff.get_imarray(maize_part_dom)
    assert maize_part_np.shape == (722, 836, 4)

    lh = idp.geotiff.get_header(lotus_part_dom)
    lotus_part_np = idp.geotiff.get_imarray(lotus_part_dom)
    assert lotus_part_np.shape == (lh["height"], lh["width"], lh["dim"])


def test_geo2pixel2geo():
    gis_coord = np.asarray([
        [ 484593.67474654, 3862259.42413431],
        [ 484593.41064743, 3862259.92582402],
        [ 484593.64841806, 3862260.06515117],
        [ 484593.93077419, 3862259.55455913],
        [ 484593.67474654, 3862259.42413431]])

    # example file
    # TIFF file: 200423_G_M600pro_transparent_mosaic_group1.tif, 411 MiB, little endian, bigtiff
    # please check v1.0 easyric.tests.test_io_geotiff.py line 114
    # to get the full string of this header
    # here we just use the extracted results
    header = {'width': 19436, 'height': 31255, 'dim':4, 
              'scale': [0.001, 0.001], 'nodata': None,
              'tie_point': [484576.70205, 3862285.5109300003], 
              'proj': pyproj.CRS.from_string("WGS 84 / UTM zone 53N")}

    expected_pixel_idx = np.array([
        [16972, 26086],
        [16708, 25585],
        [16946, 25445],
        [17228, 25956],
        [16972, 26086]])

    expected_pixel_flt = np.array([
        [16972.69654   , 26086.79569047],
        [16708.59742997, 25585.10598028],
        [16946.36805996, 25445.77883044],
        [17228.72418998, 25956.37087012],
        [16972.69654   , 26086.79569047]])

    # ==========================
    # 1. return pixel int index
    # ==========================
    pixel_coord_idx = idp.geotiff.geo2pixel(gis_coord, header, return_index=True)
    np.testing.assert_almost_equal(pixel_coord_idx, expected_pixel_idx)

    # if return index, will cause precision loss
    gis_revert_idx = idp.geotiff.pixel2geo(pixel_coord_idx, header)
    np.testing.assert_almost_equal(gis_revert_idx, gis_coord, decimal=3)

    # ===============================
    # 2. return pixel float position
    # ===============================
    pixel_coord_flt = idp.geotiff.geo2pixel(gis_coord, header)
    np.testing.assert_almost_equal(pixel_coord_flt, expected_pixel_flt)

    # then convert back should have fewer precision loss
    # but seems still have some preoblem
    gis_revert_flt = idp.geotiff.pixel2geo(pixel_coord_idx, header)
    np.testing.assert_almost_equal(gis_revert_flt, gis_coord, decimal=3)


def test_tifffile_crop():
    maize_part_np = idp.geotiff.get_imarray(maize_part_dom)
    # untiled tiff but 2 row as a patch
    with tf.TiffFile(maize_part_dom) as tif:
        page = tif.pages[0]

        # even start + even line number
        top = 30
        left = 40
        height = 100
        width = 150

        maize_np_crop = maize_part_np[top:top+height, left:left+width]
        maize_tf_crop = idp.geotiff.tifffile_crop(page, top, left, height, width)

        # vscode debug command to visualize compare two results
        # fig, ax = plt.subplots(1,2); ax[0].imshow(maize_np_crop); ax[1].imshow(maize_tf_crop); ax[0].axis("equal"); ax[1].axis("equal");plt.show()
        np.testing.assert_equal(maize_np_crop, maize_tf_crop)


        # even start + odd line number
        top = 60
        left = 40
        height = 151
        width = 153

        maize_np_crop_odd = maize_part_np[top:top+height, left:left+width]
        maize_tf_crop_odd = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(maize_np_crop_odd, maize_tf_crop_odd)

        # odd start + even line number:
        top = 91
        left = 40
        height = 230
        width = 153

        maize_np_crop_odd2 = maize_part_np[top:top+height, left:left+width]
        maize_tf_crop_odd2 = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(maize_np_crop_odd2, maize_tf_crop_odd2)

        # odd start + odd line number
        top = 61
        left = 40
        height = 237
        width = 150

        maize_np_crop_odd2 = maize_part_np[top:top+height, left:left+width]
        maize_tf_crop_odd2 = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(maize_np_crop_odd2, maize_tf_crop_odd2)


    lotus_part_np = idp.geotiff.get_imarray(lotus_part_dom)
    with tf.TiffFile(lotus_part_dom) as tif:
        page = tif.pages[0]

        top = 30
        left = 40
        height = 100
        width = 150

        lotus_np_crop = lotus_part_np[top:top+height, left:left+width]
        lotus_tf_crop = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(lotus_np_crop, lotus_tf_crop)


    # read dsm
    lotus_part_np = idp.geotiff.get_imarray(lotus_part_dsm)
    with tf.TiffFile(lotus_part_dsm) as tif:
        page = tif.pages[0]

        top = 30
        left = 40
        height = 100
        width = 150

        lotus_np_crop = lotus_part_np[top:top+height, left:left+width]
        lotus_tf_crop = idp.geotiff.tifffile_crop(page, top, left, height, width)
        np.testing.assert_equal(lotus_np_crop, lotus_tf_crop)


def test_point_query():
    # query one point
    point1 = (368023.004, 3955500.669)
    # query one point list
    point2 = [368023.004, 3955500.669]
    # query several points
    point3 = [
        [368022.581, 3955501.054], 
        [368024.032, 3955500.465]]
    # query several points by numpy
    point4 = np.array(point3)

    header = idp.geotiff.get_header(lotus_full_dsm)
    with tf.TiffFile(lotus_full_dsm) as tif:
        page = tif.pages[0]

        # point 1
        out1 = idp.geotiff.point_query(page, point1, header)
        expect = np.asarray([97.45558])
        np.testing.assert_almost_equal(out1, expect, decimal=3)

        # point 2
        out2 = idp.geotiff.point_query(page, point2, header)
        np.testing.assert_almost_equal(out2, expect, decimal=3)

        # point 3
        out3 = idp.geotiff.point_query(page, point3, header)
        expects = np.array([97.624344, 97.59617])
        np.testing.assert_almost_equal(out3, expects, decimal=3)

        # point 4
        out4 = idp.geotiff.point_query(page, point4, header)
        np.testing.assert_almost_equal(out4, expects, decimal=3)


def test_point_query_raise_error():
    with tf.TiffFile(lotus_full_dsm) as tif:
        page = tif.pages[0]

        # raise type error
        set1 = {1,2}
        set2 = {1,2,3}

        with pytest.raises(TypeError, match=re.escape("Only tuple, list, ndarray are supported")):
            idp.geotiff.point_query(page, set1)

        with pytest.raises(TypeError, match=re.escape("Only tuple, list, ndarray are supported")):
            idp.geotiff.point_query(page, set2)

        # raise index error
        tuple1 = (1,2,3)
        tuple2 = ((1,2,3), (1,2,3))

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, tuple1)

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, tuple2)

        list1 = [1,2,3]
        list2 = [[1,2,3],[1,2,3]]

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, list1)

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, list2)

        ndarray1 = np.array(list1)
        ndarray2 = np.array(list2)

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, ndarray1)

        with pytest.raises(IndexError, match=re.escape("Please only spcify shape like [x, y] or [[x1, y1], [x2, y2], ...]")):
            idp.geotiff.point_query(page, ndarray2)


def test_point_query_raise_warn():
    # warn without given header
    point = (3023.004, 3500.669)
    with tf.TiffFile(lotus_full_dsm) as tif:
        page = tif.pages[0]

        with pytest.warns(UserWarning, match=re.escape("The given pixel coordinates is not integer")):
            out = idp.geotiff.point_query(page, point)


def test_imarray_clip_2d_rgb_rgba():
    photo_path = r"./tests/data/pix4d/lotus_tanashi_full/photos/DJI_0174.JPG"
    roi = np.asarray([
        [2251, 1223], 
        [2270, 1270], 
        [2227, 1263], 
        [2251, 1223]])

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    # -----------------------------------------------
    # 3d rgb
    imarray_rgb = plt.imread(photo_path)
    # imarray_rgb.shape == (3456, 4608, 3)
    im_out_rgb, offsets_rgb = idp.geotiff.imarray_crop(imarray_rgb, roi)

    ax[1].imshow(im_out_rgb)
    ax[1].set_title('rgb')

    # -----------------------------------------------
    # 2d
    imarray_2d = idp.cvtools.rgb2gray(imarray_rgb)

    im_out_2d, offsets_2d = idp.geotiff.imarray_crop(imarray_2d, roi)

    ax[0].imshow(im_out_2d, cmap='gray')
    ax[0].set_title('gray')

    # -----------------------------------------------
    # rgba
    imarray_rgba = np.dstack((imarray_rgb, np.ones((3456, 4608)) * 255))
    # imarray_rgba.shape == (3456, 4608, 4)

    im_out_rgba, offsets_rgba = idp.geotiff.imarray_crop(imarray_rgba, roi)
    ax[2].imshow(im_out_rgba)
    ax[2].set_title('rgba')

    plt.savefig(r"./tests/out/geotiff_test/imarray_clip_test.png")

    # then check the results
    expected_offsets = np.array([2227, 1223])
    np.testing.assert_equal(offsets_2d, expected_offsets)
    np.testing.assert_equal(offsets_rgb, expected_offsets)
    np.testing.assert_equal(offsets_rgba, expected_offsets)

    assert np.all(im_out_rgb == im_out_rgba)

    assert im_out_2d[20,20] == 144.8887
    np.testing.assert_equal(im_out_rgb[20,20,:], np.array([163, 138, 133, 255], dtype=np.uint8))