import re
import sys
import pytest
import easyidp as idp

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