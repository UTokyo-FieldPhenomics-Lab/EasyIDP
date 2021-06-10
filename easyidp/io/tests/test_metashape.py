import os
import pyproj
import numpy as np
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

    assert judge is True


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
    xml_str = metashape._get_frame_zip_xml(test_project_folder, test_project_name, chunk_id=0, frame_path="0/frame.zip")
    print(len(xml_str))
    assert len(xml_str) == 519762


def test_get_chunk_id_by_xml():
    test_full_zip_path = easyidp.test_full_path("data/metashape/goya_test.files/project.zip")
    xml_file = "doc.xml"

    xml_str = metashape._get_xml_str_from_zip_file(test_full_zip_path, xml_file)
    chunk_dict = metashape._get_chunk_ids_from_xml(xml_str)

    assert chunk_dict == {"0": "0/chunk.zip"}


def test_decode_chunk_xml():
    test_project_folder = easyidp.test_full_path("data/metashape")
    test_project_name = "goya_test"
    xml_str = metashape._get_chunk_zip_xml(test_project_folder, test_project_name, chunk_id=0)
    test_proj, _ = metashape._decode_chunk_xml(xml_str)

    # test read chunk meta
    assert test_proj.label == "Chunk 1"
    assert test_proj.enabled is True
    assert len(test_proj.sensors) == 1

    # test chunk.transform matrix
    # following ans_* got from Metashape.pro API, is what expected answer
    ans_chunk_tsf = np.asarray([[-0.8657309759110793, -0.014891856498000285, 0.08977677128648079, 7.650341229333415],
                                [0.06972335119280451, 0.4433439089416722, 0.7458931501276138, 1.8591092785011023],
                                [-0.058483248465068666, 0.7489967755063389, -0.43972184234423906, -0.1835614955366731],
                                [0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(test_proj.transform.matrix, ans_chunk_tsf, decimal=12)

    # test read sensor meta
    assert test_proj.sensors[0].id == 0
    assert test_proj.sensors[0].label == "FC7203 (4.49mm)"
    assert test_proj.sensors[0].type == "frame"
    assert test_proj.sensors[0].width == 4000
    assert test_proj.sensors[0].calibration.f == 3013.7286896854812
    assert test_proj.sensors[0].calibration.cx == 219.81021538195535

    # test read camera meta
    # use id=0 "DJI_0284.JPG" for testing
    # todo: make sensors or photos as set, can use both int index and label to find.
    assert test_proj.photos[0].id == 0
    assert test_proj.photos[0].label == "DJI_0284.JPG"
    assert test_proj.photos[0].sensor_idx == 0
    assert test_proj.photos[0].orientation == 1

    # following ans_* got from Metashape.pro API, is what expected answer
    ans_tsf = np.asarray([[0.998157300665441, -0.06016650140822401, 0.007873705390367042, 7.967148855267111],
                          [-0.03731987005019368, -0.5063938831530705, 0.8614943194267687, -0.3902668180910836],
                          [-0.047845902935536415, -0.8602006900796164, -0.5077061574954894, -1.2222166155933671],
                          [0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(test_proj.photos[0].transform, ans_tsf, decimal=12)

    ans_rot = np.asarray([[0.9999258746317216, -0.01193612435043077, 0.00240295182591878],
                          [0.011945130297057436, 0.9999215509193764, -0.0037690674145338357],
                          [-0.002357775259212237, 0.003797491603681809, 0.9999900099267728]])
    np.testing.assert_array_almost_equal(test_proj.photos[0].rotation, ans_rot, decimal=12)

    ans_translation = np.asarray([0.00493069, -0.00685792, -0.00531006])
    np.testing.assert_array_almost_equal(test_proj.photos[0].translation, ans_translation, decimal=6)

    # Metashape camera.center
    ans_camera_center = np.asarray([7.967148855267111, -0.3902668180910836, -1.2222166155933671])
    np.testing.assert_array_almost_equal(test_proj.photos[0].get_camera_center(), ans_camera_center, decimal=12)

    # for empty 254 DJI_0538.JPG, 255 DJI_0539.JPG, 256 DJI_0540.JPG, all of them should be None
    assert test_proj.photos[254].transform is None
    assert test_proj.photos[255].rotation is None
    assert test_proj.photos[256].translation is None


def test_decode_crs():
    test_local_crs = easyidp.test_full_path("data/metashape/goya_test.psx")
    test_wgs84_crs = easyidp.test_full_path("data/metashape/wheat_tanashi.psx")

    chunk_local = metashape.open_project(test_local_crs)
    chunk_wgs84 = metashape.open_project(test_wgs84_crs)

    assert chunk_local[0].world_crs == pyproj.CRS.from_dict({"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'})

    epsg_4326 = pyproj.CRS.from_epsg(4326)
    assert chunk_wgs84[0].crs.name == epsg_4326.name
    assert chunk_wgs84[0].crs.datum == epsg_4326.datum
    assert chunk_wgs84[0].crs.ellipsoid == epsg_4326.ellipsoid


def test_open_project():
    test_project_folder = easyidp.test_full_path("data/metashape")
    chunks = metashape.open_project(test_project_folder)

    assert chunks is None

    test_project_folder = easyidp.test_full_path("data/metashape/goya_test.psx")
    chunks = metashape.open_project(test_project_folder)

    assert len(chunks) == 1