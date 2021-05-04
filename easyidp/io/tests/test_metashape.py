import os
import easyidp
from easyidp.io import metashape


def test_split_project_path():
    test_data = "data/metashape/goya_test.psx"
    folder_path, project_name, ext = metashape._split_project_path(test_data)

    assert folder_path == "data/metashape"
    assert project_name == "goya_test"
    assert ext == ".psx"


def test_check_is_software():
    test_full_path = easyidp.test_full_path("data/metashape/goya_test.psx")

    judge = metashape._check_is_software(test_full_path)

    assert judge == True


def test_get_xml_str_from_zip_file():
    test_full_zip_path = easyidp.test_full_path("data/metashape/goya_test.files/project.zip")
    xml_file = "doc.xml"

    xml_str = metashape._get_xml_str_from_zip_file(test_full_zip_path, xml_file)

    expected_str = """\
<?xml version="1.0" encoding="UTF-8"?>
<document version="1.2.0">
  <chunks next_id="1" active_id="0">
    <chunk id="0" path="0/chunk.zip"/>
  </chunks>
  <meta>
    <property name="Info/LastSavedDateTime" value="2021:03:31 04:19:53"/>
    <property name="Info/LastSavedSoftwareVersion" value="1.7.0.11736"/>
    <property name="Info/OriginalDateTime" value="2021:02:16 06:29:49"/>
    <property name="Info/OriginalSoftwareName" value="Agisoft Metashape"/>
    <property name="Info/OriginalSoftwareVendor" value="Agisoft"/>
    <property name="Info/OriginalSoftwareVersion" value="1.7.0.11736"/>
  </meta>
</document>
"""
    assert xml_str == expected_str


def test_get_chunk_zip_xml():
    test_project_folder = easyidp.test_full_path("data/metashape")
    test_project_name = "goya_test"
    xml_str = metashape._get_chunk_zip_xml(test_project_folder, test_project_name, chunk_id=0)
    assert len(xml_str) == 357539


def test_get_frame_zip_xml():
    test_project_folder = easyidp.test_full_path("data/metashape")
    test_project_name = "goya_test"
    xml_str = metashape._get_frame_zip_xml(test_project_folder, test_project_name, chunk_id=0)
    print(len(xml_str))
    assert len(xml_str) == 519762
