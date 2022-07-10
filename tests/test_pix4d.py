import os
import pytest
import numpy as np
import pyproj
import re
import easyidp as idp
from easyidp.pix4d import _get_full_path as gfp

data_path =  "./tests/data/pix4d"

lotus_full = os.path.join(data_path, "lotus_tanashi_full")
lotus_part = os.path.join(data_path, "lotus_tanashi")
maize_part = os.path.join(data_path, "maize_tanashi")

def test_hidden_match_suffix():
    test_folder = os.path.join(maize_part, "maize_tanashi_raname_empty_test/2_densification/point_cloud")

    out1 = idp.pix4d._match_suffix(test_folder, "xyz")
    assert out1 is None

    out2 = idp.pix4d._match_suffix(test_folder, "ply")
    assert out2 == f"{test_folder}/aaa_empty.ply"

    out3 = idp.pix4d._match_suffix(test_folder, ["laz", "ply"])
    assert out3 == f"{test_folder}/aaa_empty.laz"

    out4 = idp.pix4d._match_suffix(test_folder, ["xyz", "laz", "ply"])
    assert out4 == f"{test_folder}/aaa_empty.laz"

    out4 = idp.pix4d._match_suffix(test_folder, ["xyz", "ply", "234"])
    assert out4 == f"{test_folder}/aaa_empty.ply"


def test_parse_p4d_project_structure():
    # normal cases
    test_folder1 = os.path.join(maize_part, "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d")
    proj_name1 = "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"

    p4d1 = idp.pix4d.parse_p4d_project(test_folder1)
    assert p4d1["project_name"] == proj_name1
    assert p4d1["param"]["project_name"] == proj_name1
    assert gfp(p4d1["pcd"]) == gfp(
        os.path.join(
            test_folder1, 
            f"2_densification/point_cloud/{proj_name1}"
            "_group1_densified_point_cloud.ply"
        )
    )
    assert gfp(p4d1["dsm"]) == gfp(
        os.path.join(test_folder1, 
            f"3_dsm_ortho/1_dsm/{proj_name1}_dsm.tif"
        )
    )
    assert gfp(p4d1["dom"]) == gfp(
        os.path.join(
            test_folder1, 
            f"3_dsm_ortho/2_mosaic/{proj_name1}"
            "_transparent_mosaic_group1.tif"
        )
    )

    # test pix4d folder with wrong output name
    test_folder2 = os.path.join(maize_part, "maize_tanashi_raname_empty_test")

    # wrong project_name force to find outputs by file format
    p4d4 = idp.pix4d.parse_p4d_project(test_folder2)
    assert gfp(p4d4["pcd"]) == gfp(
        os.path.join(
            test_folder2, 
            f"2_densification/point_cloud/aaa_empty.ply"
        )
    )
    assert gfp(p4d4["dsm"]) == gfp(
        os.path.join(test_folder2, f"3_dsm_ortho/1_dsm/bbb_dsm.tif")
    )
    assert gfp(p4d4["dom"]) == gfp(
        os.path.join(test_folder2, f"3_dsm_ortho/2_mosaic/ccc_dom.tif")
    )


def test_parse_p4d_project_warning():
    # can not find output file
    test_folder = os.path.join(maize_part, "maize_tanashi_raname_no_outputs")
    
    with pytest.warns(UserWarning, 
        match=re.escape("Unable to find any")
    ):
        p4d2 = idp.pix4d.parse_p4d_project(test_folder)


def test_parse_p4d_project_structure_error():
    # folder without 1_init
    test_folder1 = lotus_part
    proj_name1 = "hasu_tanashi_20170525_Ins1RGB_30m"

    with pytest.raises(FileNotFoundError, 
        match=re.escape(
            f"Can not find pix4d parameter in given project folder"
        )
    ):
        p4d = idp.pix4d.parse_p4d_project(
            test_folder1, proj_name1
        )

    # no paramter folder
    test_folder2 = os.path.join(maize_part, "maize_tanashi_no_param")
    with pytest.raises(FileNotFoundError, 
        match=re.escape(f"Can not find pix4d parameter in given project folder")
    ):
        p4d = idp.pix4d.parse_p4d_project(test_folder2)

################
# parse params #
################
param_folder = os.path.join(
    maize_part, 
    "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/1_initial/params"
)

def test_parse_p4d_param_folder():
    param = idp.pix4d.parse_p4d_param_folder(param_folder)

    assert param["project_name"] == "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"
    assert os.path.basename(param["xyz"]) == "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_offset.xyz"


def test_read_params():
    param = idp.pix4d.parse_p4d_param_folder(param_folder)
    xyz = idp.pix4d.read_xyz(param["xyz"])

    assert xyz == (368009.0, 3955854.0, 97.0)

    pmat = idp.pix4d.read_pmat(param["pmat"])

    assert "DJI_0954.JPG" in pmat.keys()
    assert pmat["DJI_0954.JPG"].shape == (3,4)

    cicp = idp.pix4d.read_cicp(param["cicp"])
    assert cicp["w_mm"] == 17.49998592
    assert cicp["h_mm"] == 13.124989440000002
    assert cicp["F"] == 15.011754049345175
    assert len(cicp) == 10

    ccp = idp.pix4d.read_ccp(param["ccp"])
    assert "DJI_0954.JPG" in ccp.keys()
    assert "w" in ccp.keys()
    assert "h" in ccp.keys()
    assert "cam_matrix" in ccp["DJI_0954.JPG"].keys()
    assert "rad_distort" in ccp["DJI_0954.JPG"].keys()

    cam_local_pos = np.array(
        [21.54872206687879199194, 
        -29.58734160676452162875,
        30.85702810138878149360])   # == geo_pos - offset
    np.testing.assert_almost_equal(ccp["DJI_0954.JPG"]["cam_pos"], cam_local_pos)

    cam_geo_pos = idp.pix4d.read_campos_geo(param["campos"])
    assert "DJI_0954.JPG" in cam_geo_pos.keys()
    assert cam_geo_pos["DJI_0954.JPG"].shape == (3,)
    np.testing.assert_almost_equal(
        cam_geo_pos["DJI_0954.JPG"],
        np.array([368030.548722,3955824.412658,127.857028])
    )

    ssk = idp.pix4d.read_cam_ssk(param["ssk"])
    assert ssk["label"] == "FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456"
    assert ssk["type"] == "frame"
    assert ssk["pixel_size"] == [3.79774000000000011568, 3.79774000000000011568]
    assert ssk["image_size_in_pixels"] == [ccp["h"], ccp["w"]]
    assert ssk["orientation"] == 1
    assert ssk["photo_center_in_pixels"] == [1727.5, 2303.5]

    crs = idp.pix4d.read_wkt_prj(param["crs"])
    assert crs == pyproj.CRS.from_epsg(32654)


def test_class_read_project():
    p4d = idp.Pix4D()

    param_folder = os.path.join(lotus_full, "params")
    image_folder = os.path.join(lotus_full, "photos")
    p4d.open_project(lotus_full, raw_img_folder=image_folder, param_folder=param_folder)

    assert p4d.software == "pix4d"
    assert p4d.label == "hasu_tanashi_20170525_Ins1RGB_30m"
    assert p4d.meta["p4d_offset"] == (368043.0, 3955495.0, 98.0)

    assert isinstance(p4d.sensors[0], idp.reconstruct.Sensor)
    assert p4d.sensors[0].label == "FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456"

    assert p4d.sensors[0].calibration.f == 3951.0935994102065
    assert p4d.sensors[0].focal_length == 15.005226206224117

    assert p4d.photos[0].label == "DJI_0151.JPG"
    assert p4d.photos[0].path == ''
    assert "photos" in p4d.photos["DJI_0174.JPG"].path
    assert p4d.photos["DJI_0174.JPG"].cam_matrix.shape == (3,3)
    assert p4d.photos["DJI_0174.JPG"].rotation.shape == (3,3)
    assert p4d.photos["DJI_0174.JPG"].location.shape == (3,)
    assert p4d.photos["DJI_0174.JPG"].transform.shape == (3,4)