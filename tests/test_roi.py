import os
import pytest
import numpy as np

import easyidp as idp

data_path =  "./tests/data/roi_test"

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