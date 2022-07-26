import os
import re
import pytest
import numpy as np
import pyproj

import easyidp as idp

data_path =  "./tests/data/roi_test"
shp_path =  "./tests/data/shp_test"

lotus_shp = r"./tests/data/pix4d/lotus_tanashi_full/plots.shp"

def test_read_cc_txt():
    results = np.array([
        [-18.42576599, -16.10819054,  -0.63814539],
        [-18.00066757, -18.05295944,  -0.67380333],
        [-16.05021095, -17.63488388,  -0.68102068],
        [-16.46848488, -15.66774559,  -0.6401825 ],
        [-18.42576599, -16.10819054,  -0.63814539]])

    # xyz type:
    xyz_fpath = os.path.join(data_path, "hasu_tanashi_xyz.txt")
    xyz_np = idp.roi.read_cc_txt(xyz_fpath)

    np.testing.assert_almost_equal(xyz_np, results)

    # lxyz type
    lxyz_fpath = os.path.join(data_path, "hasu_tanashi_xyz.txt")
    lxyz_np = idp.roi.read_cc_txt(lxyz_fpath)

    np.testing.assert_almost_equal(lxyz_np, results)


def test_class_roi_init():
    roi = idp.ROI()

    # specify values
    roi["ddfge"] = np.array([[1,2],[4,5]])
    assert len(roi) == 1
    assert "ddfge" in roi.item_label.keys()

def test_class_roi_read_shp():
    shp_test_path = os.path.join(shp_path, "roi.shp")
    roi = idp.ROI()

    roi.read_shp(shp_test_path, name_field=0)

    assert len(roi) == 3
    assert roi.crs.name == 'WGS 84 / UTM zone 54N'

    # also test overwrite read
    roi = idp.ROI()
    roi.read_shp(lotus_shp, name_field=0)

    assert roi.crs.name == "WGS 84"
    assert "N1W1" in roi.keys()

def test_class_read_labelme_json():
    json_path = r"./tests/data/json_test/labelme_demo_img.json"

    roi = idp.ROI(json_path)

    assert roi['1'][0,0] == 2447.239263803681

    # test errors
    err_json = r"./tests/data/json_test/for_read_json.json"
    with pytest.raises(TypeError, match=re.escape("It seems [./tests/data/json_test/for_read_json.json] is not a Labelme json file")):
        roi = idp.ROI(err_json)

    # test warnings
    warn_json = r"./tests/data/json_test/labelme_warn_img.json"
    with pytest.warns(UserWarning, match=re.escape("Only labelme [polygon] shape are accepted, not [points] of [1]")):
        roi = idp.ROI(warn_json)

        assert len(roi) == 0


def test_class_roi_change_crs():
    roi = idp.ROI(lotus_shp)

    lotus_full_dom = "./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"
    obj = idp.GeoTiff(lotus_full_dom)

    roi.change_crs(obj.header["crs"])
    assert roi.crs.name == obj.header["crs"].name

def test_class_roi_get_z_from_dsm():
    # only test whether works, not examine the value is true or not
    roi = idp.ROI(lotus_shp, name_field=0)
    # have different CRS from shp file
    lotus_full_dsm = "./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"

    # only pick 3 plots as testing data
    key_list = list(roi.keys())
    for key in key_list:
        if key not in ["N1W1", "N2E2", "S1W1"]:
            del roi[key]
            
    assert len(roi) == 3

    #######################
    # test mode == points #
    #######################
    ht = 97.63990020751953
    map_ht = 97.56273651123047
    # var name -> roi_mode_kernel_buffer_keepcrs
    roi_p_mean_0_f = roi.copy()
    roi_p_mean_0_f.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=0, keep_crs=False)
    assert roi_p_mean_0_f.crs.name == 'WGS 84 / UTM zone 54N'
    assert roi_p_mean_0_f[0].shape == (5,3)
    np.testing.assert_almost_equal(roi_p_mean_0_f[0][0,0], 368017.7565143015)
    np.testing.assert_almost_equal(roi_p_mean_0_f[0][0,1], 3955511.081022765)
    np.testing.assert_almost_equal(roi_p_mean_0_f[0][0,2], ht)

    roi_p_mean_0_t = roi.copy()
    roi_p_mean_0_t.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=0, keep_crs=True)
    assert roi_p_mean_0_t.crs.name == "WGS 84"
    assert roi_p_mean_0_t[0].shape == (5,3)
    np.testing.assert_almost_equal(roi_p_mean_0_t[0][0,0], 35.73475194328632)  # latitude
    np.testing.assert_almost_equal(roi_p_mean_0_t[0][0,1], 139.54052962153048)  # longitude
    np.testing.assert_almost_equal(roi_p_mean_0_t[0][0,2], ht)

    roi_p_mean_1_f = roi.copy()
    roi_p_mean_1_f.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=1, keep_crs=False)
    assert roi_p_mean_1_f[0].shape == (5,3)
    assert roi_p_mean_1_f[0][0,1] != ht

    roi_p_mean_1d0_f = roi.copy()
    roi_p_mean_1d0_f.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=1.0, keep_crs=False)
    assert roi_p_mean_1d0_f[0].shape == (5,3)
    # buffer 1.0 and buffer 1 should be the same
    assert roi_p_mean_1d0_f[0][0,2] == roi_p_mean_1_f[0][0,2]

    # using full map as results
    roi_p_mean_m1_f = roi.copy()
    roi_p_mean_m1_f.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=-1, keep_crs=False)
    # all the z values should be the same
    assert all(np.all(i[:,2] == map_ht) for i in roi_p_mean_m1_f.values())

    roi_p_mean_m1d0_f = roi.copy()
    roi_p_mean_m1d0_f.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=-1.0, keep_crs=False)
    # all the z values should be the same
    assert all(np.all(i[:,2] == map_ht) for i in roi_p_mean_m1d0_f.values())

    #####################
    # test mode == face #
    #####################
    roi_f_mean_1_f = roi.copy()
    roi_f_mean_1_f.get_z_from_dsm(lotus_full_dsm, mode="face", kernel="mean", buffer=1, keep_crs=False)
    assert roi_f_mean_1_f[0].shape == (5,3)
    np.testing.assert_almost_equal(roi_f_mean_1_f[0][0,0], 368017.7565143015)
    np.testing.assert_almost_equal(roi_f_mean_1_f[0][0,1], 3955511.081022765)

    roi_f_mean_0_f = roi.copy()
    roi_f_mean_0_f.get_z_from_dsm(lotus_full_dsm, mode="face", kernel="mean", buffer=0, keep_crs=False)
    assert roi_f_mean_0_f[0].shape == (5,3)
    np.testing.assert_almost_equal(roi_f_mean_0_f[0][0,0], 368017.7565143015)
    np.testing.assert_almost_equal(roi_f_mean_0_f[0][0,1], 3955511.081022765)

def test_class_roi_get_z_from_dsm_errors():
    roi = idp.ROI(lotus_shp)
    lotus_full_dsm = "./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"

    with pytest.raises(KeyError, match=re.escape("The param 'mode' only accept 'point' or 'face', not 'abcde'")):
        roi.get_z_from_dsm(lotus_full_dsm, mode="abcde")

    with pytest.raises(KeyError, match=re.escape(
        "The param 'kernal' only accept 'mean', 'min', 'max', 'pmin5', 'pmin10', 'pmax5', 'pmax10' not 'abcde'"
        )):
        roi.get_z_from_dsm(lotus_full_dsm, kernel="abcde")

    with pytest.raises(TypeError, match=re.escape(
        "Only 'int' and 'float' are acceptable for 'buffer', not <class 'str'> [abcde]"
        )):
        roi.get_z_from_dsm(lotus_full_dsm, buffer="abcde")

    with pytest.raises(TypeError, match=re.escape("Only geotiff path <str> and <easyidp.GeoTiff> object")):
        roi.get_z_from_dsm("seffed")

    with pytest.raises(TypeError, match=re.escape("Only geotiff path <str> and <easyidp.GeoTiff> object")):
        roi.get_z_from_dsm(23345)

    with pytest.raises(TypeError, match=re.escape("Could not operate without CRS specified")):
        roi.crs = None
        roi.get_z_from_dsm(lotus_full_dsm, buffer="abcde")


def test_class_roi_crop():
    # just ensure it can run, the data examine please check corresponding modules' test

    # data prepare
    lotus_full_dsm = r"./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"
    lotus_full_pcd = r"./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_group1_densified_point_cloud.ply"
    lotus_full_dom = r"./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"
    lotus_full_shp = r"./tests/data/pix4d/lotus_tanashi_full/plots.shp"

    roi = idp.ROI(lotus_full_shp, name_field=0)

    # only pick 3 plots as testing data
    key_list = list(roi.keys())
    for key in key_list:
        if key not in ["N1W1", "N2E2", "S1W1"]:
            del roi[key]

    roi.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=0, keep_crs=False)

    # crop geotiff
    out_dom = roi.crop(lotus_full_dom)
    assert len(out_dom) == 3

    out_dsm = roi.crop(lotus_full_dsm)
    assert len(out_dsm) == 3

    # crop point cloud
    out_pcd = roi.crop(lotus_full_pcd)
    assert len(out_pcd) == 3

def test_class_roi_crop_error():
    roi = idp.ROI()

    with pytest.raises(TypeError, match=re.escape("Could not operate without CRS specified")):
        roi.crop("aaa")

    roi.crs = pyproj.CRS.from_epsg(4326)
    with pytest.raises(TypeError, match=re.escape(
        "Only file path <str> or <easyidp.GeoTiff> object or <easyidp.PointCloud>"
        " object are accepted, not <class 'str'>")):
        roi.crop("aaa")