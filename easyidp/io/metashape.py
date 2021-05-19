import os
import pyproj
import zipfile
import numpy as np
from xml.etree import ElementTree
from easyidp.core.objects import ReconsProject, MetashapeChunkTransform, Sensor, Calibration, Photo


def open_project(path: str):
    """
    Read project data by given Metashape project path.

    sub-functions used in this heading function
    * _split_project_path()
    * _get_project_zip_xml()
    * _get_chunk_ids_from_xml()
    for each chunk:
        * _get_chunk_zip_xml()
        * _decode_chunk_xml()
            * _decode_chunk_transform_tag()
            * _decode_sensor_tag()
                * _decode_calibration_tag()
            * _decode_camera_tag()
            * _decode_frame_tag()
                * _get_frame_zip_xml()
                * _decode_frame_xml()

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------

    """
    if _check_is_software(path):
        folder_path, project_name, ext = _split_project_path(path)
        project_xml_str = _get_project_zip_xml(folder_path, project_name)
        chunk_dict = _get_chunk_ids_from_xml(project_xml_str)
        chunk_list = []
        for chunk_id in chunk_dict.keys():
            chunk_xml_str = _get_chunk_zip_xml(folder_path, project_name, chunk_id=chunk_id)
            recons_proj, frame_path_dict = _decode_chunk_xml(chunk_xml_str)
            for frame_path in frame_path_dict.values():
                frame_xml_str = _get_frame_zip_xml(folder_path, project_name, chunk_id=chunk_id, frame_path=frame_path)
                camera_meta, marker_meta = _decode_frame_xml(frame_xml_str)
                for camera_idx, camera_path in camera_meta.items():
                    recons_proj.photos[camera_idx].path = camera_path

            chunk_list.append(recons_proj)

        return chunk_list
    else:
        return None


def _split_project_path(path: str):
    """
    [inner_function] Get project name, current folder, extension, etc. from given project path.

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------
    folder_path: str
        e.g. "/root/to/metashape/"
    project_name: str
        e.g. "test_proj"
    ext: str:
        e.g. "psx"
    """
    folder_path, file_name = os.path.split(path)
    project_name, ext = os.path.splitext(file_name)

    return folder_path, project_name, ext


def _check_is_software(path: str):
    """
    [inner function] Check if given path is metashape project structure

    Parameters
    ----------
    path: str
        e.g. proj_path="/root/to/metashape/test_proj.psx"

    Returns
    -------
    judge: bool
        true: is metashape projects (has complete project structure)
        false: not a metashape projects (missing some files or path not found)
    """
    judge = False
    if not os.path.exists(path):
        print(f"Could not find Metashape project file [{path}]")
        return judge

    folder_path, project_name, ext = _split_project_path(path)
    data_folder = os.path.join(folder_path, project_name + ".files")

    if not os.path.exists(data_folder):
        print(f"Could not find Metashape project file [{data_folder}]")
        return judge

    judge = True
    return judge


def _get_xml_str_from_zip_file(zip_file, xml_file):
    """
    [inner function] read xml file in zip file

    Parameters
    ----------
    zip_file: str
        the path to zip file, e.g. "data/metashape/goya_test.files/project.zip"
    xml_file: str
        the file name in zip file, e.g. "doc.xml" in previous zip file

    Returns
    -------
    xml_str: str
        the string of readed xml file
    """
    with zipfile.ZipFile(zip_file) as zfile:
        xml = zfile.open(xml_file)
        xml_str = xml.read().decode("utf-8")

    return xml_str


def _get_project_zip_xml(project_folder, project_name):
    """
    [inner function] read project.zip xml files to string
    project path = "/root/to/metashape/test_proj.psx"
    -->  project_folder = "/root/to/metashape/"
    -->  project_name = "test_proj"

    Parameters
    ----------
    project_folder: str
    project_name: str

    Returns
    -------
    xml_str: str
        the string of loaded "doc.xml"
    """
    zip_file = f"{project_folder}/{project_name}.files/project.zip"
    return _get_xml_str_from_zip_file(zip_file, "doc.xml")


def _get_chunk_zip_xml(project_folder, project_name, chunk_id):
    """
    [inner function] read chunk.zip xml file in given chunk
        project path = "/root/to/metashape/test_proj.psx"
    -->  project_folder = "/root/to/metashape/"
    -->  project_name = "test_proj"

    Parameters
    ----------
    project_folder: str
    project_name: str
    chunk_id: int or str
        the chunk id start from 0 of chunk.zip

    Returns
    -------
    xml_str: str
        the string of loaded "doc.xml"
    """
    zip_file = f"{project_folder}/{project_name}.files/{chunk_id}/chunk.zip"
    return _get_xml_str_from_zip_file(zip_file, "doc.xml")


def _get_frame_zip_xml(project_folder, project_name, chunk_id, frame_path):
    """
    [inner function] read frame.zip xml file in given chunk
    project path = "/root/to/metashape/test_proj.psx"
    -->  project_folder = "/root/to/metashape/"
    -->  project_name = "test_proj"

    Parameters
    ----------
    project_folder: str
    project_name: str
    frame_path: str
    chunk_id: int or str
        the chunk id start from 0 of chunk.zip

    Returns
    -------
    xml_str: str
        the string of loaded "doc.xml"
    """
    zip_file = f"{project_folder}/{project_name}.files/{chunk_id}/{frame_path}"
    return _get_xml_str_from_zip_file(zip_file, "doc.xml")


def _get_chunk_ids_from_xml(xml_str):
    """
    Parameters
    ----------
    xml_str: str
        the xml string from project.zip file, for example:

        <document version="1.2.0">
          <chunks next_id="2">
            <chunk id="0" path="0/chunk.zip"/>
          </chunks>
          <meta>
            <property name="Info/LastSavedDateTime" value="2020:06:22 02:23:20"/>
            <property name="Info/LastSavedSoftwareVersion" value="1.6.2.10247"/>
            <property name="Info/OriginalDateTime" value="2020:06:22 02:20:16"/>
            <property name="Info/OriginalSoftwareName" value="Agisoft Metashape"/>
            <property name="Info/OriginalSoftwareVendor" value="Agisoft"/>
            <property name="Info/OriginalSoftwareVersion" value="1.6.2.10247"/>
          </meta>
        </document>

    Returns
    -------
    chunk_dict: dict
        key = chunk_id, value = chunk_path

    """
    chunk_dict = {}
    xml_tree = ElementTree.fromstring(xml_str)

    for chunk in xml_tree[0]:   # tree[0] -> <chunks>
        chunk_dict[chunk.attrib['id']] = chunk.attrib['path']

    return chunk_dict


def _decode_chunk_xml(xml_str):
    """

    Parameters
    ----------
    xml_str: str
        the xml string from chunk.zip file, has the following main structure:

        <chunk version="1.2.0" label="170525" enabled="true">
          <sensors next_id="1">
            <sensor id="0" label="FC550, DJI MFT 15mm F1.7 ASPH (15mm)" type="frame">
              ...  -> _decode_sensor_tag()
            </sensor>
            ...  -> for loop in this function
          </sensors>

          <cameras next_id="266" next_group_id="0">
            <camera id="0" sensor_id="0" label="DJI_0151">
              ... -> _decode_camera_tag()
            </camera>
            ...  -> for loop in this function
          </cameras>

          <frames next_id="1">
            <frame id="0" path="0/frame.zip"/>
          </frames>

          <reference>GEOGCS["WGS 84",DATUM["World Geodetic System 1984",], ..., AUTHORITY["EPSG","4326"]]</reference>

          <transform>
            ...  -> _decode_chunk_transform_tag()
          <transform>

          <region>
            <center>-8.1800672545192263e+00 -2.6103071594817338e+00 -1.0706980639382815e+01</center>
            <size>3.3381093645095824e+01 2.3577857828140260e+01 7.2410767078399658e+00</size>
            <R>-9.8317886026354417e-01 ...  9.9841729020145376e-01</R>   // 9 numbers
          </region>

    Returns
    -------

    """
    xml_tree = ElementTree.fromstring(xml_str)

    recons_proj = ReconsProject(software="metashape")
    recons_proj.label = xml_tree.attrib["label"]
    recons_proj.enabled = bool(xml_tree.attrib["enabled"])

    # metashape chunk.transform.matrix
    transform_tags = xml_tree.findall("./transform")
    if len(transform_tags) == 1:
        recons_proj.transform = _decode_chunk_transform_tag(transform_tags[0])

    for sensor_tag in xml_tree.findall("./sensors/sensor"):
        sensor = _decode_sensor_tag(sensor_tag)
        sensor.calibration.software = recons_proj.software
        recons_proj.sensors[sensor.idx] = sensor

    for camera_tag in xml_tree.findall("./cameras/camera"):
        camera = _decode_camera_tag(camera_tag)
        recons_proj.photos[camera.idx] = camera

    frame_path_dict = {}
    for frame_tag in xml_tree.findall("./frames/frame"):
        frame_zip_idx = frame_tag.attrib["id"]
        frame_zip_path = frame_tag.attrib["path"]
        frame_path_dict[frame_zip_idx] = frame_zip_path

    recons_proj.crs = _decode_chunk_reference_tag(xml_tree.findall("./reference"))

    return recons_proj, frame_path_dict


def _decode_chunk_transform_tag(xml_obj):
    """
    Topic: Camera coordinates to world coordinates using 4x4 matrix
    https://www.agisoft.com/forum/index.php?topic=6176.15

    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of xml_tree.findall("./transform")

        <transform>
          <rotation locked="false">-9.9452052163852955e-01 ...  -5.0513659345945017e-01</rotation>  // 9 numbers
          <translation locked="false">7.6503412293334154e+00 ... -1.8356149553667311e-01</translation>   // 3 numbers
          <scale locked="true">8.7050086657310788e-01</scale>
        </transform>

    Returns
    -------
    transform_dict: dict
       key: ["rotation", "translation", "scale", "transform"]
    """
    transform = MetashapeChunkTransform()
    chunk_rotation_str = xml_obj.findall("./rotation")[0].text
    transform.rotation = np.fromstring(chunk_rotation_str, sep=" ", dtype=np.float).reshape((3, 3))

    chunk_translation_str = xml_obj.findall("./translation")[0].text
    transform.translation = np.fromstring(chunk_translation_str, sep=" ", dtype=np.float)

    transform.scale = float(xml_obj.findall("./scale")[0].text)

    transform.matrix = np.zeros((4, 4))
    transform.matrix[0:3, 0:3] = transform.rotation * transform.scale
    transform.matrix[0:3, 3] = transform.translation
    transform.matrix[3, :] = np.asarray([0, 0, 0, 1])

    return transform


def _decode_chunk_reference_tag(xml_obj):
    """
    change CRS string to pyproj.CRS object
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        if no GPS info provided, the reference tag will look like this:
        <reference>LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],
        UNIT["metre",1,AUTHORITY["EPSG","9001"]]]</reference>

        if given longitude and latitude, often is "WGS84":
          <reference>GEOGCS["WGS 84",DATUM["World Geodetic System 1984",SPHEROID["WGS 84",6378137,298.
          257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM
          ["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG",
          "9102"]],AUTHORITY["EPSG","4326"]]</reference>

    Returns
    -------

    """
    crs_str = xml_obj[0].text
    local_crs = 'LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'

    if crs_str == local_crs:
        crs_obj = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})
    else:
        crs_obj = pyproj.CRS.from_string(crs_str)

    # sometimes, the Photoscan given CRS string can not transform correctly
    # this is the solution for "WGS 84" CRS shown in the previous example
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    if crs_obj.datum == crs_wgs84.datum:
        return crs_wgs84
    else:
        return crs_obj


def _decode_sensor_tag(xml_obj):
    """
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of xml_tree.findall("./sensors/sensor")

        <sensor id="0" label="FC550, DJI MFT 15mm F1.7 ASPH (15mm)" type="frame">
          <resolution width="4608" height="3456"/>
          <property name="pixel_width" value="0.00375578257860832"/>
          <property name="pixel_height" value="0.00375578257860832"/>
          <property name="focal_length" value="15"/>
          <property name="layer_index" value="0"/>
          <bands>
            <band label="Red"/>
            <band label="Green"/>
            <band label="Blue"/>
          </bands>
          <data_type>uint8</data_type>
          <calibration type="frame" class="adjusted">
            ...  -> _decode_calibration_tag()
          </calibration>
          <covariance>
            <params>f cx cy k1 k2 k3 p1 p2</params>
            <coeffs>...</coeffs>
          </covariance>
        </sensor>

    Returns
    -------
    sensor: easyidp.Sensor object
    """
    sensor = Sensor()

    sensor.idx = int(xml_obj.attrib["id"])
    sensor.label = xml_obj.attrib["label"]
    sensor.type = xml_obj.attrib["type"]

    resolution = xml_obj.findall("./resolution")[0]
    sensor.width = int(resolution.attrib["width"])
    sensor.width_unit = "px"
    sensor.height = int(resolution.attrib["height"])
    sensor.height_unit = "px"

    sensor.pixel_width = float(xml_obj.findall("./property/[@name='pixel_width']")[0].attrib["value"])
    sensor.pixel_width_unit = "mm"
    sensor.pixel_height = float(xml_obj.findall("./property/[@name='pixel_height']")[0].attrib["value"])
    sensor.pixel_height_unit = "mm"
    sensor.focal_length = float(xml_obj.findall("./property/[@name='focal_length']")[0].attrib["value"])

    sensor.calibration = _decode_calibration_tag(xml_obj.findall("./calibration")[0])

    return sensor


def _decode_calibration_tag(xml_obj):
    """
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of sensor_tag.findall("./calibration")

        <calibration type="frame" class="adjusted">
          <resolution width="4608" height="3456"/>
          <f>4317.82144109411</f>
          <cx>54.3448479006764</cx>
          <cy>3.5478801249494</cy>
          <k1>0.0167015604921295</k1>
          <k2>-0.0239416622459823</k2>
          <k3>0.0363031944008484</k3>
          <p1>0.0035044847840485</p1>
          <p2>0.00103698463015268</p2>
        </calibration>

    Returns
    -------
    calibration: easyidp.Calibration object
    """
    calibration = Calibration()

    calibration.f = float(xml_obj.findall("./f")[0].text)
    calibration.f_unit = "px"

    calibration.cx = float(xml_obj.findall("./cx")[0].text)
    calibration.cx_unit = "px"
    calibration.cy = float(xml_obj.findall("./cy")[0].text)
    calibration.cy_unit = "px"

    if len(xml_obj.findall("./b1")) == 1:
        calibration.b1 = float(xml_obj.findall("./b1")[0].text)
    if len(xml_obj.findall("./b2")) == 1:
        calibration.b2 = float(xml_obj.findall("./b2")[0].text)

    if len(xml_obj.findall("./k1")) == 1:
        calibration.k1 = float(xml_obj.findall("./k1")[0].text)
    if len(xml_obj.findall("./k2")) == 1:
        calibration.k2 = float(xml_obj.findall("./k2")[0].text)
    if len(xml_obj.findall("./k3")) == 1:
        calibration.k3 = float(xml_obj.findall("./k3")[0].text)
    if len(xml_obj.findall("./k4")) == 1:
        calibration.k4 = float(xml_obj.findall("./k4")[0].text)

    if len(xml_obj.findall("./p1")) == 1:
        calibration.t1 = float(xml_obj.findall("./p1")[0].text)
    if len(xml_obj.findall("./p2")) == 1:
        calibration.t2 = float(xml_obj.findall("./p2")[0].text)
    if len(xml_obj.findall("./p3")) == 1:
        calibration.t3 = float(xml_obj.findall("./p3")[0].text)
    if len(xml_obj.findall("./p4")) == 1:
        calibration.t4 = float(xml_obj.findall("./p4")[0].text)

    return calibration


def _decode_camera_tag(xml_obj):
    """
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of xml_tree.findall("./cameras/camera")

        <camera id="0" sensor_id="0" label="DJI_0151">
          <transform>9.9181741779621246e-01 ... 0 0 0 1</transform>  // 16 numbers
          <rotation_covariance>5.9742650282250832e-04 ... 2.3538470709659123e-04</rotation_covariance>  // 9 numbers
          <location_covariance>2.0254219245789448e-02 ... 2.6760756179895751e-02</location_covariance>  // 9 numbers
          <orientation>1</orientation>

          // sometimes have following reference tag, otherwise need to look into frames.zip xml
          <reference x="139.540561166667" y="35.73454525" z="134.765" yaw="164.1" pitch="0"
                     roll="-0" enabled="true" rotation_enabled="false"/>
        </camera>

        but some camera have empty tags:

        <camera id="254" sensor_id="0" label="DJI_0538.JPG">
          <orientation>1</orientation>
        </camera>

        also need to deal with such situation

    Returns
    -------
    camera: easyidp.Photo object
    """
    camera = Photo()
    camera.idx = int(xml_obj.attrib["id"])
    camera.sensor_idx = int(xml_obj.attrib["sensor_id"])
    camera.label = xml_obj.attrib["label"]
    camera.orientation = int(xml_obj.findall("./orientation")[0].text)

    # deal with camera have empty tags
    transform_tag = xml_obj.findall("./transform")
    if len(transform_tag) == 1:
        transform_str = transform_tag[0].text
        camera.transform = np.fromstring(transform_str, sep=" ", dtype=np.float).reshape((4, 4))

    shutter_rotation_tag = xml_obj.findall("./rolling_shutter/rotation")
    if len(shutter_rotation_tag) == 1:
        shutter_rotation_str = shutter_rotation_tag[0].text
        camera.rotation = np.fromstring(shutter_rotation_str, sep=" ", dtype=np.float).reshape((3, 3))

    shutter_translation_tag = xml_obj.findall("./rolling_shutter/translation")
    if len(shutter_translation_tag) == 1:
        shutter_translation_str = shutter_translation_tag[0].text
        camera.translation = np.fromstring(shutter_translation_str, sep=" ", dtype=np.float)

    return camera


def _decode_frame_xml(xml_str):
    """

    Parameters
    ----------
    xml_str: str
        the xml string from chunk.zip file, for example:

        <?xml version="1.0" encoding="UTF-8"?>
        <frame version="1.2.0">
          <cameras>
            <camera camera_id="0">
              <photo path="//172.31.12.56/pgg2020a/drone/20201029/goya/DJI_0284.JPG">
                <meta>
                  <property name="DJI/AbsoluteAltitude" value="+89.27"/>
                  <property name="DJI/FlightPitchDegree" value="+0.50"/>
                  <property name="DJI/FlightRollDegree" value="-0.40"/>
                  <property name="DJI/FlightYawDegree" value="+51.00"/>
                  <property name="DJI/GimbalPitchDegree" value="+0.00"/>
                  <property name="DJI/GimbalRollDegree" value="+0.00"/>
                  <property name="DJI/GimbalYawDegree" value="+0.30"/>
                  <property name="DJI/RelativeAltitude" value="+1.20"/>
                  <property name="Exif/ApertureValue" value="2.97"/>
                  <property name="Exif/DateTime" value="2020:10:29 14:13:09"/>
                  <property name="Exif/DateTimeOriginal" value="2020:10:29 14:13:09"/>
                  <property name="Exif/ExposureTime" value="0.001"/>
                  <property name="Exif/FNumber" value="2.8"/>
                  <property name="Exif/FocalLength" value="4.49"/>
                  <property name="Exif/FocalLengthIn35mmFilm" value="24"/>
                  <property name="Exif/GPSAltitude" value="89.27"/>
                  <property name="Exif/GPSLatitude" value="35.327145"/>
                  <property name="Exif/GPSLongitude" value="139.989069888889"/>
                  <property name="Exif/ISOSpeedRatings" value="200"/>
                  <property name="Exif/Make" value="DJI"/>
                  <property name="Exif/Model" value="FC7203"/>
                  <property name="Exif/Orientation" value="1"/>
                  <property name="Exif/ShutterSpeedValue" value="9.9657"/>
                  <property name="Exif/Software" value="v02.51.0008"/>
                  <property name="File/ImageHeight" value="3000"/>
                  <property name="File/ImageWidth" value="4000"/>
                  <property name="System/FileModifyDate" value="2020:10:29 14:13:10"/>
                  <property name="System/FileSize" value="5627366"/>
                </meta>
              </photo>
            </camera>
            ...
          </cameras>
          <markers>
            <marker marker_id="0">
              <location camera_id="230" pinned="true" x="3042.73071" y="1298.28467"/>
              <location camera_id="103" pinned="true" x="2838.39014" y="457.134155"/>
              ...
            </marker>
            ...
          </markers>
          <thumbnails path="thumbnails/thumbnails.zip"/>
          <point_cloud path="point_cloud/point_cloud.zip"/>
          <depth_maps id="0" path="depth_maps/depth_maps.zip"/>
          <dense_cloud id="0" path="dense_cloud/dense_cloud.zip"/>
        </frame>
    Returns
    -------

    """
    camera_meta = {}
    marker_meta = {}

    xml_tree = ElementTree.fromstring(xml_str)

    for camera_tag in xml_tree.findall("./cameras/camera"):
        camera_idx = int(camera_tag.attrib["camera_id"])
        camera_path = camera_tag[0].attrib["path"]
        camera_meta[camera_idx] = camera_path

    return camera_meta, marker_meta