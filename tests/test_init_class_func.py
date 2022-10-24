import re
import sys
import pytest
import easyidp as idp

def test_class_container():
    # for i in c, 
    # for i in c.keys, 
    # for i in c.values, 
    # for i, j in c.items()
    ctn = idp.Container()

    k = [str(_) for _ in range(5)]

    v = []
    for _ in range(6, 11):
        p = idp.reconstruct.Photo()
        p.label = str(_)
        v.append(p)

    val = {}
    for i, j in zip(k,v):
        ctn[i] = j
        val[int(i)] = j
    
    assert ctn.id_item == val

    # test object iteration
    for idx, value in enumerate(ctn):
        assert value == v[idx]

    for idx, value in ctn.items():
        assert value in v

    for key in ctn.keys():  # [6,7,8,9,10]
        assert key in ['6','7','8','9','10']

    for value in ctn.values():
        assert value in v

    # test get order by slicing
    slice_test = ctn[0:3]

    assert len(slice_test) == 3
    for k in slice_test.keys():
        assert k in ['6', '7', '8']
    


def test_def_parse_photo_relative_path():

    if sys.platform.startswith("win"):
        frame_path = r"Z:\ishii_Pro\sumiPro\2022_soy_weeds_metashape_result\220613_G_M600pro\220613_G_M600pro.files\0\0\frame.zip"
        rel_path = r"../../../../source/220613_G_M600pro/DSC06093.JPG"
        actual_path = idp.parse_relative_path(frame_path, rel_path)
        expected_path = 'Z:\\ishii_Pro\\sumiPro\\2022_soy_weeds_metashape_result\\source\\220613_G_M600pro\\DSC06093.JPG'

        assert actual_path == expected_path

    else :
        frame_path = r"/ishii_Pro/sumiPro/2022_soy_weeds_metashape_result/220613_G_M600pro/220613_G_M600pro.files/0/0/frame.zip"
        rel_path = r"../../../../source/220613_G_M600pro/DSC06093.JPG"

        actual_path = idp.parse_relative_path(frame_path, rel_path)
        expected_path = r'/ishii_Pro/sumiPro/2022_soy_weeds_metashape_result/source/220613_G_M600pro/DSC06093.JPG'


def test_def_parse_photo_relative_path_warn():
    frame_path = r"Z:\ishii_Pro\sumiPro\2022_soy_weeds_metashape_result\220613_G_M600pro\220613_G_M600pro.files\0\0\frame.zip"
    rel_path = "//172.31.12.56/pgg2020a/drone/20201029/goya/DJI_0284.JPG"

    with pytest.warns(UserWarning, match=re.escape("Seems it is an absolute path")):
        get_path = idp.parse_relative_path(frame_path, rel_path)

    assert get_path == rel_path