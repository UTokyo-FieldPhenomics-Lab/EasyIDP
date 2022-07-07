import os
import pytest
import re
import easyidp as idp

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


def path_ready(path):
    return os.path.normpath(os.path.realpath(path)) 


def test_parse_p4d_project_structure():
    # normal cases
    test_folder1 = os.path.join(maize_part, "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d")
    proj_name1 = "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"

    p4d1 = idp.pix4d.parse_p4d_project_structure(test_folder1)
    assert p4d1["project_name"] == proj_name1
    assert path_ready(p4d1["param"]) == path_ready(
        os.path.join(test_folder1, "1_initial/params")
    )
    assert path_ready(p4d1["pcd"]) == path_ready(
        os.path.join(
            test_folder1, 
            f"2_densification/point_cloud/{proj_name1}"
            "_group1_densified_point_cloud.ply"
        )
    )
    assert path_ready(p4d1["dsm"]) == path_ready(
        os.path.join(test_folder1, 
            f"3_dsm_ortho/1_dsm/{proj_name1}_dsm.tif"
        )
    )
    assert path_ready(p4d1["dom"]) == path_ready(
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
    assert path_ready(p4d2["param"]) == path_ready(
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
    assert path_ready(p4d3["param"]) == path_ready(
        os.path.join(test_folder2, "1_initial/params")
    )
    assert p4d3["pcd"] is None
    assert p4d3["dsm"] is None
    assert p4d3["dom"] is None

    # warong project_name force to find outputs by file format
    p4d4 = idp.pix4d.parse_p4d_project_structure(test_folder2, force_find=True)
    assert path_ready(p4d4["pcd"]) == path_ready(
        os.path.join(
            test_folder2, 
            f"2_densification/point_cloud/aaa_empty.ply"
        )
    )
    assert path_ready(p4d4["dsm"]) == path_ready(
        os.path.join(test_folder2, f"3_dsm_ortho/1_dsm/bbb_dsm.tif")
    )
    assert path_ready(p4d4["dom"]) == path_ready(
        os.path.join(test_folder2, f"3_dsm_ortho/2_mosaic/ccc_dom.tif")
    )


def test_parse_p4d_project_structure_error():
    # use maize empty test
    
    # folder without 1_init
    test_folder1 = lotus_part
    proj_name1 = "hasu_tanashi_20170525_Ins1RGB_30m"

    with pytest.raises(FileNotFoundError, 
        match=re.escape(f"Current folder [")
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
    
def test_parse_p4d_param_folder():
    pass

def test_read_xyz():
    pass


def test_read_pmat():
    pass


def test_read_cicp():
    pass


def test_read_ccp():
    pass