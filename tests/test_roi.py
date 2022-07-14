import os
import pytest
import numpy as np

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


def test_class_roi_change_crs():
    roi = idp.ROI(lotus_shp)

    lotus_full_dom = "./tests/data/pix4d/lotus_tanashi_full/hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"
    obj = idp.GeoTiff(lotus_full_dom)

    roi.change_crs(obj.header["proj"])
    assert roi.crs.name == obj.header["proj"].name

def test_class_roi_get_z_from_dsm():
    pass