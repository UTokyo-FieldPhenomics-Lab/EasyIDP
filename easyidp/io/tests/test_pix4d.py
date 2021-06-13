import os
import easyidp
from easyidp.io.pix4d import _match_suffix, get_project_structure


def test_match_suffix():
    test_folder = "data/pix4d/maize_tanashi/maize_tanashi_raname_empty_test/2_densification/point_cloud"
    test_folder = os.path.join(easyidp.__path__[0], test_folder)

    out1 = _match_suffix(test_folder, "xyz")
    assert out1 is None

    out2 = _match_suffix(test_folder, "ply")
    assert out2 == f"{test_folder}/aaa_empty.ply"

    out3 = _match_suffix(test_folder, ["laz", "ply"])
    assert out3 == f"{test_folder}/aaa_empty.laz"

    out4 = _match_suffix(test_folder, ["xyz", "laz", "ply"])
    assert out4 == f"{test_folder}/aaa_empty.laz"

    out4 = _match_suffix(test_folder, ["xyz", "ply", "234"])
    assert out4 == f"{test_folder}/aaa_empty.ply"


def test_get_project_structure():
    test_folder1 = "data/pix4d/maize_tanashi/maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"
    test_folder1 = os.path.join(easyidp.__path__[0], test_folder1)
    proj_name1 = "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d"

    p4d1 = get_project_structure(test_folder1)
    assert p4d1["project_name"] == proj_name1
    assert p4d1["param"] == os.path.join(test_folder1, "1_initial/params").replace('\\', '/')
    assert p4d1["pcd"] == os.path.join(test_folder1, f"2_densification/point_cloud/{proj_name1}_group1_densified_point_cloud.ply").replace('\\', '/')
    assert p4d1["dsm"] == os.path.join(test_folder1, f"3_dsm_ortho/1_dsm/{proj_name1}_dsm.tif").replace('\\', '/')
    assert p4d1["dom"] == os.path.join(test_folder1, f"3_dsm_ortho/2_mosaic/{proj_name1}_transparent_mosaic_group1.tif").replace('\\', '/')

    test_folder2 = "data/pix4d/maize_tanashi/maize_tanashi_raname_empty_test"
    test_folder2 = os.path.join(easyidp.__path__[0], test_folder2)
    proj_name2 = "maize_tanashi_raname_empty_test"

    p4d2 = get_project_structure(test_folder2)
    assert p4d2["project_name"] == proj_name2
    assert p4d2["param"] == os.path.join(test_folder2, "1_initial/params").replace('\\', '/')
    assert p4d2["pcd"] is None
    assert p4d2["dsm"] is None
    assert p4d2["dom"] is None

    p4d3 = get_project_structure(test_folder2, project_name="aaa_empty")
    assert p4d3["project_name"] == "aaa_empty"
    assert p4d3["param"] == os.path.join(test_folder2, "1_initial/params").replace('\\', '/')
    assert p4d3["pcd"] is None
    assert p4d3["dsm"] is None
    assert p4d3["dom"] is None

    p4d4 = get_project_structure(test_folder2, force_find=True)
    assert p4d4["pcd"] == os.path.join(test_folder2, f"2_densification/point_cloud/aaa_empty.ply").replace('\\', '/')
    assert p4d4["dsm"] == os.path.join(test_folder2, f"3_dsm_ortho/1_dsm/bbb_dsm.tif").replace('\\', '/')
    assert p4d4["dom"] == os.path.join(test_folder2, f"3_dsm_ortho/2_mosaic/ccc_dom.tif").replace('\\', '/')
