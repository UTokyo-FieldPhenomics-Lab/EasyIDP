import re
import pytest
import numpy as np
import pyproj

import easyidp as idp

test_data = idp.data.TestData()
from . import roi_select

def test_read_cc_txt():
    results = np.array([
        [-18.42576599, -16.10819054,  -0.63814539],
        [-18.00066757, -18.05295944,  -0.67380333],
        [-16.05021095, -17.63488388,  -0.68102068],
        [-16.46848488, -15.66774559,  -0.6401825 ],
        [-18.42576599, -16.10819054,  -0.63814539]])

    # xyz type:
    xyz_fpath = test_data.roi.xyz_txt
    xyz_np = idp.roi.read_cc_txt(xyz_fpath)

    np.testing.assert_almost_equal(xyz_np, results)

    # lxyz type
    lxyz_fpath = test_data.roi.lxyz_txt
    lxyz_np = idp.roi.read_cc_txt(lxyz_fpath)

    np.testing.assert_almost_equal(lxyz_np, results)


def test_class_roi_init():
    roi = idp.ROI()

    # specify values
    roi["ddfge"] = np.array([[1,2],[4,5]])
    assert len(roi) == 1
    assert "ddfge" in roi.item_label.keys()

def test_class_roi_slice():
    # solve the issue 58
    roi = idp.ROI()

    roi.read_shp(test_data.shp.roi_shp, name_field=0)

    assert len(roi) == 3

    roi = roi[0:2]

    assert len(roi) == 2

    for func_name in ["crop", "back2raw", "get_z_from_dsm"]:
        assert func_name in dir(roi)

def test_class_roi_copy():
    roi = idp.ROI()

    roi.read_shp(test_data.shp.roi_shp, name_field=0)

    roi_copy = roi.copy()

    roi_copy[0][0, 0] = 368000

    assert roi_copy[0][0, 0] == 368000
    assert roi[0][0, 0] == 368101.05641086295

    assert roi.crs == roi_copy.crs
    assert roi.source == roi_copy.source

def test_class_roi_read_shp():
    roi = idp.ROI()

    roi.read_shp(test_data.shp.roi_shp, name_field=0)

    assert len(roi) == 3
    assert roi.crs.name == 'WGS 84 / UTM zone 54N'

    # also test overwrite read
    roi = idp.ROI()
    roi.read_shp(test_data.shp.lotus_shp, name_field=0)

    assert roi.crs.name == "WGS 84"
    assert "N1W1" in roi.keys()

def test_class_read_labelme_json():
    json_path = test_data.json.labelme_demo

    roi = idp.ROI(json_path)

    assert roi['1'][0,0] == 2447.239263803681

    # test errors
    with pytest.raises(TypeError, match=r"It seems \[.*for_read_json\.json\] is not a Labelme json file"):
        # > It seems [C:\\Users\\hwang\\AppData\\Local\\easyidp.data\\data_for_tests\\json_test\\for_read_json.json] is not a Labelme json file.
        roi = idp.ROI(test_data.json.labelme_err)
        

    # test warnings
    with pytest.warns(UserWarning, match=re.escape("Only labelme [polygon] shape are accepted, not [points] of [1]")):
        roi = idp.ROI(test_data.json.labelme_warn)

        assert len(roi) == 0


def test_class_roi_change_crs():
    roi = idp.ROI(test_data.shp.lotus_shp)

    obj = idp.GeoTiff(test_data.pix4d.lotus_dom)

    roi.change_crs(obj.header["crs"])
    assert roi.crs.name == obj.header["crs"].name

def test_class_roi_get_z_from_dsm():
    # only test whether works, not examine the value is true or not
    roi = roi_select.copy()

    assert len(roi) == 3
    # have different CRS from shp file
    lotus_full_dsm = test_data.pix4d.lotus_dsm

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
    np.testing.assert_almost_equal(roi_p_mean_0_t[0][0,0], 139.54052962153048)  # longitude
    np.testing.assert_almost_equal(roi_p_mean_0_t[0][0,1], 35.73475194328632)  # latitude
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


def test_class_roi_get_z_from_dsm_warns():
    # the ROI outside the DSM ranges and cause nan values for z
    # bug report #69
    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    roi = roi[0:3]

    roi[0] = roi[0] + np.array([0,  -0.00029697])  # offset 1 degree lon & lat

    lotus_full_dsm = test_data.pix4d.lotus_dsm

    with pytest.warns(UserWarning, match=re.escape("Z values contains empty attribute [-10000.0] for ['N1W1'], this may be caused by the")):
        roi.get_z_from_dsm(lotus_full_dsm, mode="point", kernel="mean", buffer=0, keep_crs=False)
        print(roi[0])

def test_class_roi_get_z_from_dsm_errors():
    roi = idp.ROI(test_data.shp.lotus_shp)
    lotus_full_dsm = test_data.pix4d.lotus_dsm

    with pytest.raises(KeyError, match=re.escape("The param 'mode' only accept 'point' or 'face', not 'abcde'")):
        roi.get_z_from_dsm(lotus_full_dsm, mode="abcde")

    with pytest.raises(KeyError, match=re.escape(
        "The param 'kernel' only accept 'mean', 'min', 'max', 'pmin5', 'pmin10', 'pmax5', 'pmax10' not 'abcde'"
        )):
        roi.get_z_from_dsm(lotus_full_dsm, kernel="abcde")

    with pytest.raises(TypeError, match=re.escape(
        "Only 'int' and 'float' are acceptable for 'buffer', not <class 'str'> [abcde]"
        )):
        roi.get_z_from_dsm(lotus_full_dsm, buffer="abcde")

    with pytest.raises(FileNotFoundError, match=re.escape("Could not find file")):
        roi.get_z_from_dsm("seffed")

    with pytest.raises(TypeError, match=re.escape("Only geotiff path <str> and <easyidp.GeoTiff> object")):
        roi.get_z_from_dsm(23345)

    with pytest.raises(TypeError, match=re.escape("Could not operate without CRS specified")):
        roi.crs = None
        roi.get_z_from_dsm(lotus_full_dsm, buffer="abcde")

def test_func_insert_z_value_for_roi_error():
    # test catching incorrest ROI.shape
    with pytest.raises(ValueError, match=re.escape("The expected ROI shape should be (n, 3), not given (5, 4)")):
        out = idp.roi._insert_z_value_for_roi(np.ones((5,4)), 3)

    val = np.array(
       [[ 368017.7565143 , 3955511.08102277],
        [ 368019.70190232, 3955511.49811902],
        [ 368020.11263046, 3955509.54636219],
        [ 368018.15769062, 3955509.13563382],
        [ 368017.7565143 , 3955511.08102277]])
    z_value = np.array(
        [[97.25353 ],
         [97.34388 ],
         [97.4322  ],
         [97.305435],
         [97.25353 ]])

    with pytest.raises(ValueError, match=re.escape("The expected z_value shape should be either (n) or (n, 1), not given (5, 1, 1)")):
        out = idp.roi._insert_z_value_for_roi(val, z_value[:, None])

def test_func_insert_z_value_for_roi_error_point_mode():
    # meet the error #72
    val = np.array(
       [[ 368017.7565143 , 3955511.08102277],
        [ 368019.70190232, 3955511.49811902],
        [ 368020.11263046, 3955509.54636219],
        [ 368018.15769062, 3955509.13563382],
        [ 368017.7565143 , 3955511.08102277]])
    z_value = np.array(
        [[97.25353 ],
         [97.34388 ],
         [97.4322  ],
         [97.305435],
         [97.25353 ]])

    out = idp.roi._insert_z_value_for_roi(val, z_value)

    assert out.shape == (5,3)

    # then the following code should runnable
    lotus = idp.data.Lotus()
    roi = idp.ROI(lotus.shp, name_field = "plot_id")
    roi.get_z_from_dsm(lotus.metashape.dsm, mode="point")

    assert roi[0].shape == (5,3)


def test_class_roi_get_z_from_dsm_duplicate_load():
    # fix bug #60
    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    roi = roi[0:3]

    lotus_full_dsm = test_data.pix4d.lotus_dsm

    roi.get_z_from_dsm(lotus_full_dsm)

    assert roi['N1W1'].shape == (5, 3)

    roi.get_z_from_dsm(lotus_full_dsm)

    assert roi['N1W1'].shape == (5, 3)

def test_class_roi_crop():
    # just ensure it can run, the data examine please check corresponding modules' test

    # data prepare
    lotus_full_dsm = test_data.pix4d.lotus_dsm
    lotus_full_pcd = test_data.pix4d.lotus_pcd 
    lotus_full_dom = test_data.pix4d.lotus_dom 
    lotus_full_shp = test_data.shp.lotus_shp 

    roi = roi_select.copy()
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


def test_class_roi_back2raw():
    # single chunk:
    lotus = idp.data.Lotus()

    p4d = idp.Pix4D(project_path=lotus.pix4d.project, 
                    raw_img_folder=lotus.photo,
                    param_folder=lotus.pix4d.param)

    ms = idp.Metashape(project_path=lotus.metashape.project, chunk_id=0)

    roi = roi_select.copy()
    roi.get_z_from_dsm(lotus.pix4d.dsm)

    ms.crs = roi.crs

    out_p4d = roi.back2raw(p4d)
    out_ms = roi.back2raw(ms)

    assert len(out_p4d) == 4
    assert len(out_ms) == 4

def test_class_roi_back2raw_error():
    ms = idp.Metashape(test_data.metashape.lotus_psx)

    roi = idp.ROI(test_data.shp.lotus_shp, name_field=0)

    # test ROI.back2raw error
    with pytest.raises(
        ValueError, 
        match=re.escape(
            "The back2raw function requires 3D roi with shape=(n, 3), but [N1W1] is (5, 2)")):
        out_ms = roi.back2raw(ms)