import os
import pytest
import numpy as np
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

    p4d1 = idp.pix4d.parse_p4d_project_structure(test_folder1)
    assert p4d1["project_name"] == proj_name1
    assert gfp(p4d1["param"]) == gfp(
        os.path.join(test_folder1, "1_initial/params")
    )
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
    proj_name2 = "maize_tanashi_raname_empty_test"

    with pytest.warns(UserWarning, 
        match=re.escape("Unable to find any")
    ):
        p4d2 = idp.pix4d.parse_p4d_project_structure(test_folder2)

    assert p4d2["project_name"] == proj_name2
    assert gfp(p4d2["param"]) == gfp(
        os.path.join(test_folder2, "1_initial/params")
    )
    assert p4d2["pcd"] is None
    assert p4d2["dsm"] is None
    assert p4d2["dom"] is None

    # wrong project_name cause can not found outputs
    with pytest.warns(UserWarning, 
        match=re.escape("Unable to find any")
    ):
        p4d3 = idp.pix4d.parse_p4d_project_structure(
            test_folder2, project_name="aaa_empty"
        )
    
    assert p4d3["project_name"] == "aaa_empty"
    assert gfp(p4d3["param"]) == gfp(
        os.path.join(test_folder2, "1_initial/params")
    )
    assert p4d3["pcd"] is None
    assert p4d3["dsm"] is None
    assert p4d3["dom"] is None

    # warong project_name force to find outputs by file format
    p4d4 = idp.pix4d.parse_p4d_project_structure(test_folder2, force_find=True)
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


def test_parse_p4d_project_structure_error():
    # use maize empty test
    
    # folder without 1_init
    test_folder1 = lotus_part
    proj_name1 = "hasu_tanashi_20170525_Ins1RGB_30m"

    with pytest.raises(FileNotFoundError, 
        match=re.escape(
            f"Current folder [{gfp(test_folder1)}] is not a standard"
        )
    ):
        p4d = idp.pix4d.parse_p4d_project_structure(
            test_folder1, proj_name1
        )

    # no paramter folder
    test_folder2 = os.path.join(maize_part, "maize_tanashi_no_param")
    with pytest.raises(FileNotFoundError, 
        match=re.escape(f"Can not find pix4d parameter in folder")
    ):
        p4d = idp.pix4d.parse_p4d_project_structure(test_folder2)

################
# parse params #
################
param_folder = os.path.join(
    maize_part, 
    "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d/1_initial/params"
)

def test_parse_p4d_param_folder():
    param = idp.pix4d.parse_p4d_param_folder(param_folder)

    assert os.path.basename(param["xyz"]) == "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_offset.xyz"


def test_parse_p4d_param_folder_error():
    with pytest.raises(FileNotFoundError, match=re.escape("Could not find param file")):
        param = idp.pix4d.parse_p4d_param_folder(param_folder, project_name="aaa")

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

    campos = idp.pix4d.read_campos_geo(param["campos"])
    assert "DJI_0954.JPG" in campos.keys()
    assert campos["DJI_0954.JPG"].shape == (3,)
    np.testing.assert_almost_equal(
        campos["DJI_0954.JPG"],
        np.array([368030.548722,3955824.412658,127.857028])
    )