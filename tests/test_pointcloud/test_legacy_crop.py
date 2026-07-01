"""Legacy crop_rois() behavior with ROI and Pix4D.

Merged from tests/test_pointcloud_legacy.py test_class_crop.
Does NOT use the old shared_data fixture — constructs ROI and Pix4D locally.
"""

import shutil
import warnings

import numpy as np

import easyidp as idp


def test_class_crop_rois(test_data):
    """Legacy crop_rois() with multiple ROI keys via Pix4D."""
    roi_all = idp.ROI(test_data.shp.lotus_shp, name_field=0)
    roi_select = idp.ROI()
    for key in ["N1W1", "N1W2", "N2E2", "S1W1"]:
        roi_select[key] = roi_all[key]
        roi_select.crs = roi_all.crs
        roi_select.source = roi_all.source

    roi_select.get_z_from_dsm(
        test_data.pix4d.lotus_dsm, mode="point",
        kernel="mean", buffer=0, keep_crs=False,
    )

    p4d = idp.Pix4D(
        project_path=test_data.pix4d.lotus_folder,
        param_folder=test_data.pix4d.lotus_param,
    )
    p4d.load_pcd(test_data.pix4d.lotus_pcd)

    tif_out_folder = test_data.pcd.out / "class_crop"
    if tif_out_folder.exists():
        shutil.rmtree(tif_out_folder)
    tif_out_folder.mkdir()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        out = p4d.pcd.crop_rois(roi_select, save_folder=tif_out_folder)

    assert len(out) == 4
    assert len(out["N1W1"]) == 15226

    assert (tif_out_folder / "N1W1.ply").exists()

    np.testing.assert_almost_equal(out["N1W1"].offset, p4d.pcd.offset)
    assert np.all(out["N1W1"]._points[:, 0:2] < 300)
