import re
import pytest
import numpy as np
import matplotlib.pyplot as plt

import easyidp as idp

test_data = idp.data.TestData()

def test_poly2mask_type_int():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    
    xy = np.array([x,y]).T # int type

    width = 11
    height = 10

    mask_shp = idp.cvtools.poly2mask((width, height), xy, engine='shapely')
    mask_skm = idp.cvtools.poly2mask((width, height), xy, engine='skimage')

    # the same results from skimage.polygon > 0.18.3
    wanted_shp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)


    wanted_pil = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    np.testing.assert_equal(mask_shp, wanted_shp)
    # np.testing.assert_equal(mask_pil, wanted_pil)
    np.testing.assert_equal(mask_skm, wanted_shp)


def test_poly2mask_type_float():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    xy = np.array([x,y], dtype=np.float16).T   # get float type

    width = 11
    height = 10

    mask_shp = idp.cvtools.poly2mask((width, height), xy, engine='shapely')
    mask_skm = idp.cvtools.poly2mask((width, height), xy, engine='skimage')

    wanted_shp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    wanted_pil = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    wanted_skm = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    np.testing.assert_equal(mask_shp, wanted_shp)
    np.testing.assert_equal(mask_skm, wanted_skm)

def test_poly2mask_out_of_bound():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    xy = np.array([x,y]).T   

    with pytest.raises(ValueError, match=re.escape("The polygon coords (1, 1, 7, 8) is out of mask boundary [0, 0, 6, 6]")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

def test_poly2mask_wrong_type():
    xy = [[1, 2],[3, 4]]

    with pytest.raises(TypeError, match=re.escape("The `poly_coord` only accept numpy ndarray integer and float types")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([1, 2, 3, 4])
    with pytest.raises(AttributeError, match=re.escape("Only nx2 ndarray are accepted")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(AttributeError, match=re.escape("Only nx2 ndarray are accepted")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    with pytest.raises(TypeError, match=re.escape(f"The `poly_coord` only accept numpy ndarray integer and float types")):
        mask = idp.cvtools.poly2mask((6, 6), xy)


def test_imarray_clip_2d_rgb_rgba():
    photo_path = test_data.pix4d.lotus_photos / "DJI_0174.JPG"
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
    im_out_rgb, offsets_rgb = idp.cvtools.imarray_crop(imarray_rgb, roi)

    ax[1].imshow(im_out_rgb)
    ax[1].set_title('rgb')

    # -----------------------------------------------
    # 2d
    imarray_2d = idp.cvtools.rgb2gray(imarray_rgb)

    im_out_2d, offsets_2d = idp.cvtools.imarray_crop(imarray_2d, roi)

    ax[0].imshow(im_out_2d, cmap='gray')
    ax[0].set_title('gray')

    # -----------------------------------------------
    # rgba
    imarray_rgba = np.dstack((imarray_rgb, np.ones((3456, 4608)) * 255))
    # imarray_rgba.shape == (3456, 4608, 4)

    im_out_rgba, offsets_rgba = idp.cvtools.imarray_crop(imarray_rgba, roi)
    ax[2].imshow(im_out_rgba)
    ax[2].set_title('rgba')

    plt.savefig(test_data.cv.out / "imarray_clip_test.png")

    # then check the results
    expected_offsets = np.array([2227, 1223])
    np.testing.assert_equal(offsets_2d, expected_offsets)
    np.testing.assert_equal(offsets_rgb, expected_offsets)
    np.testing.assert_equal(offsets_rgba, expected_offsets)

    assert np.all(im_out_rgb == im_out_rgba)

    assert im_out_2d[20,20] == 144.8887
    np.testing.assert_equal(im_out_rgb[20,20,:], np.array([163, 138, 133, 255], dtype=np.uint8))

def test_imarray_crop_handle_polygon_hv_with_float():
    # fix issue #61
    img_coord = \
    np.asarray([[7214.31561958, 3741.17729258],
                [6090.04392943, 3062.91735589],
                [6770.00648193, 1952.53149042],
                [7901.26625814, 2624.35137412],
                [7214.31561958, 3741.17729258]])

    np.random.seed(0)
    img_np = np.random.randint(low=0, high=255, size=(10000,10000,3), dtype=np.uint8)

    imarray, offsets = idp.cvtools.imarray_crop(img_np, img_coord)

    assert 1 == 1

def test_roi_smaller_than_one_pixel_error():   # disucssion #39
    one_dim_imarray = np.array([255,255,255])
    polygon_hv = np.array([[1, 1], [2, 2], [1, 3], [1, 1]])
    with pytest.raises(
        ValueError, 
        match=re.escape(
            "Only image dimention=2 (mxn) or 3(mxnxd) are accepted, not current"
        )
    ):
        idp.cvtools.imarray_crop(one_dim_imarray, polygon_hv)