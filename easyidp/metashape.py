import os
import pyproj
import zipfile
import numpy as np
import warnings
from xml.etree import ElementTree
import xml.dom.minidom as minidom
from copy import copy as ccopy

import easyidp as idp


class Metashape(idp.reconstruct.Recons):
    # the object for each chunk in Metashape project

    def __init__(self, project_path=None, chunk_id=None):
        super().__init__()
        self.software = "metashape"
        self.transform = idp.reconstruct.ChunkTransform()

        # the whole metashape project (parent) info
        self.project_folder = None
        self.project_name = None
        self.project_chunks_dict = None
        self.crs = None
        self.reference_crs = pyproj.CRS.from_epsg(4326)

        if project_path is not None:
            self._open_whole_project(project_path)
            if chunk_id is not None:
                self.open_chunk(chunk_id)
        else:
            if chunk_id is not None:
                raise LookupError(
                    f"Could not load chunk_id [{chunk_id}] for project_path [{project_path}]")
        

    ###############
    # zip/xml I/O #
    ###############

    def open_chunk(self, chunk_id, project_path=None):
        # given new project path, switch project
        if project_path is not None:  
            self._open_whole_project(project_path)
        else:
            # not given metashape project path when init this class
            if self.project_chunks_dict is None:
                raise FileNotFoundError(
                    f"Not specify Metashape project (project_path="
                    f"'{os.path.join(self.project_folder, self.project_name)}')"
                    f"Please specify `project_path` to this function"
                )
        
        if isinstance(chunk_id, int):
            chunk_id = str(chunk_id)

        if chunk_id in self.project_chunks_dict.keys():
            chunk_content_dict = read_chunk_zip(
                self.project_folder, 
                self.project_name, 
                chunk_id=chunk_id, skip_disabled=False)
            self._chunk_dict_to_object(chunk_content_dict)
        else:
            raise KeyError(
                f"Could not find chunk_id [{chunk_id}] in "
                f"{list(self.project_chunks_dict.keys())}")

    def _open_whole_project(self, project_path):
        _check_is_software(project_path)
        folder_path, project_name, ext = _split_project_path(project_path)

        # chunk_dict.keys() = [1,2,3,4,5] int id
        project_dict = read_project_zip(folder_path, project_name)

        self.project_folder = folder_path
        self.project_name = project_name
        self.project_chunks_dict = project_dict

    def _chunk_dict_to_object(self, chunk_dict):
        self.label = chunk_dict["label"]
        self.enabled = chunk_dict["enabled"]
        self.transform = chunk_dict["transform"]
        self.sensors = chunk_dict["sensors"]
        self.photos = chunk_dict["photos"]
        self.reference_crs = chunk_dict["crs"]

    
    #######################
    # backward projection #
    #######################

    def _local2world(self, points_np):
        return apply_transform_matrix(points_np, self.transform.matrix)

    def _world2local(self, points_np):
        if self.transform.matrix_inv is None:
            self.transform.matrix_inv = np.linalg.inv(self.transform.matrix)

        return apply_transform_matrix(points_np, self.transform.matrix_inv)

    def _world2crs(self, points_np):
        if self.crs is None:
            return convert_proj3d(points_np, self.world_crs, self.reference_crs)
        else:
            return convert_proj3d(points_np, self.world_crs, self.crs)

    def _crs2world(self, points_np):
        if self.crs is None:
            return convert_proj3d(points_np, self.reference_crs, self.world_crs)
        else:
            return convert_proj3d(points_np, self.crs, self.world_crs)

    def _back2raw_one2one(self, points_np, photo_id, distortion_correct=True):
        """Project one ROI(polygon) on one given photo

        Parameters
        ----------
        points_np : ndarray (nx3)
            The 3D coordinates of polygon vertexm, in local coordinates
        photo_id : int | str | <easyidp.Photo> object
            the photo that will project on. Can be photo id (int), photo name (str) or <Photo> object
        distortion_correct : bool, optional
            Whether do distortion correction, by default True (back to raw image);
            If back to software corrected images without len distortion, set it to True. 
            Pix4D support do this operation, seems metashape not supported yet.

        Returns
        -------
        ndarray
            nx2 pixel coordiante of ROI on images
        """
        if isinstance(photo_id, (int, str)):
            camera_i = self.photos[photo_id]
            sensor_i = self.sensors[camera_i.sensor_id]
        elif isinstance(photo_id, idp.reconstruct.Photo):
            camera_i = photo_id
            sensor_i = photo_id.sensor
        else:
            raise TypeError(
                f"Only <int> photo id or <easyidp.reconstruct.Photo> object are accepted, "
                f"not {type(photo_id)}")

        if not camera_i.enabled:
            return None

        t = camera_i.transform[0:3, 3]
        r = camera_i.transform[0:3, 0:3]

        points_np, is_single = _is_single_point(points_np)

        xyz = (points_np - t).dot(r)

        xh = xyz[:, 0] / xyz[:, 2]
        yh = xyz[:, 1] / xyz[:, 2]

        # without distortion
        if distortion_correct:
                        # with distortion
            u, v = sensor_i.calibration.calibrate(xh, yh)

            out = np.vstack([u, v]).T

        else:
            w = sensor_i.width
            h = sensor_i.height
            f = sensor_i.calibration.f
            cx = sensor_i.calibration.cx
            cy = sensor_i.calibration.cy

            k = np.asarray([[f, 0, w / 2 + cx, 0],
                            [0, f, h / 2 + cy, 0],
                            [0, 0, 1, 0]])

            # make [x, y, 1, 1] for multiple points
            pch = np.vstack([xh, yh, np.ones(len(xh)), np.ones(len(xh))]).T
            ppix = pch.dot(k.T)

            out = ppix[:, 0:2]

        if is_single:
            return out[0, :]
        else:
            return out

    def back2raw_crs(self, points_xyz, distortion_correct=True, save_folder=None, ignore=None, log=False):
        """Projs one GIS coordintates ROI (polygon) to all images

        Parameters
        ----------
        points_hv : ndarray (nx3)
            The 3D coordinates of polygon vertexm, in CRS coordinates
        distortion_correct : bool, optional
            Whether do distortion correction, by default True (back to raw image);
            If back to software corrected images without len distortion, set it to True. 
            Pix4D support do this operation, seems metashape not supported yet.
        ignore : str | None, optional
            None: strickly in image area;
            'x': only y (vertical) in image area, x can outside image;
            'y': only x (horizontal) in image area, y can outside image.
        log : bool, optional
            whether print log for debugging, by default False

        Returns
        -------
        dict,
            a dictionary that key = img_name and values= pixel coordinate
        """
        local_coord = self._world2local(self._crs2world(points_xyz))

        # for each raw image in the project / flight
        out_dict = {}
        for photo_name, photo in self.photos.items():
            # skip not enabled photos
            if not photo.enabled:
                continue
            # reverse projection to given raw images
            projected_coord = self._back2raw_one2one(local_coord, photo, distortion_correct=True)

            # find out those correct images
            if log:
                print(f'[Calculator][Judge]{photo.label}w:{photo.sensor.width}h:{photo.sensor.height}->', end='')

            coords = photo.sensor.in_img_boundary(projected_coord, ignore=ignore, log=log)
            if coords is not None:
                out_dict[photo_name] = coords

        return out_dict


    def back2raw(self, roi, save_folder=None, **kwargs):
        """Projects several GIS coordintates ROIs (polygons) to all images

        Parameters
        ----------
        roi : easyidp.ROI | dict
            the <ROI> object created by easyidp.ROI() or dictionary
        save_folder : str, optional
            the folder to save projected preview images and json files, by default ""
        distortion_correct : bool, optional
            Whether do distortion correction, by default True (back to raw image);
            If back to software corrected images without len distortion, set it to True. 
            Pix4D support do this operation, seems metashape not supported yet.
        ignore : str | None, optional
            None: strickly in image area;
            'x': only y (vertical) in image area, x can outside image;
            'y': only x (horizontal) in image area, y can outside image.
        log : bool, optional
            whether print log for debugging, by default False
        """
        out_dict = {}

        before_crs = ccopy(self.crs)
        self.crs = ccopy(roi.crs)
        for k, points_xyz in roi.items():
            if isinstance(save_folder, str) and os.path.isdir(save_folder):
                save_path = os.path.join(save_folder, k)
            else:
                save_path = None

            one_roi_dict= self.back2raw_crs(points_xyz, save_folder=save_path, **kwargs)

            out_dict[k] = one_roi_dict

        self.crs = before_crs
        return out_dict

    def get_photo_position(self, to_crs=None):
        """Get all photos' center geo position (on given CRS)

        Parameters
        ----------
        to_crs : pyproj.CRS, optional
            Transformed to another geo coordinate, by default None, the project.crs

        Returns
        -------
        dict
            The dictionary contains "photo.label": [x, y, z] coordinates
        """

        # change the out crs
        before_crs = ccopy(self.crs)
        if isinstance(to_crs, pyproj.CRS) and not to_crs.equals(self.crs):
            self.crs = ccopy(to_crs)

        out = {}
        for p in self.photos:
            if p.enabled:
                pos = self._world2crs(self._local2world(p.transform[0:3, 3]))
                out[p.label] = pos
                p.position = pos

        self.crs = before_crs

        return out


    def sort_img_by_distance(self, img_dict_all, roi, distance_thresh=None, num=None):
        """Advanced wrapper of sorting back2raw img_dict results by distance from photo to roi

        Parameters
        ----------
        img_dict_all : dict
            All output dict of roi.back2raw(...)
            e.g. img_dict = roi.back2raw(...) -> img_dict
        roi: idp.ROI
            Your roi variable
        num : None or int
            Keep the closest {x} images
        distance_thresh : None or float
            Keep the images closer than this distance to ROI.

        Returns
        -------
        dict
            the same structure as output of roi.back2raw(...)
        """
        return idp.reconstruct.sort_img_by_distance(self, img_dict_all, roi, distance_thresh, num)

###############
# zip/xml I/O #
###############

def read_project_zip(project_folder, project_name):
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
    project_dict: dict
        key = chunk_id, value = chunk_path

    Notes
    -----
    xml_str example:

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
    """
    project_dict = {}

    zip_file = f"{project_folder}/{project_name}.files/project.zip"
    xml_str = _get_xml_str_from_zip_file(zip_file, "doc.xml")
    xml_tree = ElementTree.fromstring(xml_str)

    for chunk in xml_tree[0]:   # tree[0] -> <chunks>
        project_dict[chunk.attrib['id']] = chunk.attrib['path']

    return project_dict


def read_chunk_zip(project_folder, project_name, chunk_id, skip_disabled=False):
    """[inner function] read chunk.zip xml file in given chunk
    project path = "/root/to/metashape/test_proj.psx"
    -->  project_folder = "/root/to/metashape/"
    -->  project_name = "test_proj"

    Parameters
    ----------
    project_folder: str
    project_name: str
    chunk_id: int or str
        the chunk id start from 0 of chunk.zip
    skip_disabled: bool
        return None if chunk enabled == False


    Returns
    -------
    chunk_dict or None

    Notes
    -----
    Example for xml_str:

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
    """
    frame_zip_file = f"{project_folder}/{project_name}.files/{chunk_id}/chunk.zip"
    xml_str = _get_xml_str_from_zip_file(frame_zip_file, "doc.xml")
    xml_tree = ElementTree.fromstring(xml_str)

    chunk_dict = {}

    chunk_dict["label"] = xml_tree.attrib["label"]
    chunk_dict["enabled"] = bool(xml_tree.attrib["enabled"])

    if skip_disabled and not chunk_dict["enabled"]:
        return None

    # metashape chunk.transform.matrix
    transform_tags = xml_tree.findall("./transform")
    if len(transform_tags) == 1:
        chunk_dict["transform"] = _decode_chunk_transform_tag(transform_tags[0])

    sensors = idp.Container()
    for sensor_tag in xml_tree.findall("./sensors/sensor"):
        debug_meta = {
            "project_folder": project_folder, 
            "project_name"  : project_name,
            "chunk_id": chunk_id,
            "chunk_path": frame_zip_file
        }

        sensor = _decode_sensor_tag(sensor_tag, debug_meta)
        if sensor.calibration is not None:
            sensor.calibration.software = "metashape"
        sensors[sensor.id] = sensor
    chunk_dict["sensors"] = sensors

    photos = idp.Container()
    for camera_tag in xml_tree.findall("./cameras/camera"):
        camera = _decode_camera_tag(camera_tag)
        camera.sensor = sensors[camera.sensor_id]
        photos[camera.id] = camera
    chunk_dict["photos"] = photos

    for frame_tag in xml_tree.findall("./frames/frame"):
        # frame_zip_idx = frame_tag.attrib["id"]
        frame_zip_path = frame_tag.attrib["path"]

        frame_zip_file = f"{project_folder}/{project_name}.files/{chunk_id}/{frame_zip_path}"
        frame_xml_str = _get_xml_str_from_zip_file(frame_zip_file, "doc.xml")

        camera_meta, marker_meta = _decode_frame_xml(frame_xml_str)
        for camera_idx, camera_path in camera_meta.items():
            chunk_dict["photos"][camera_idx]._path = camera_path
            # here need to resolve absolute path
            # <photo path="../../../../source/220613_G_M600pro/DSC06035.JPG">
            # this is the root to 220613_G_M600pro.files\0\0\frame.zip"
            if "../../../" in camera_path:
                chunk_dict["photos"][camera_idx].path = idp.parse_relative_path(
                    frame_zip_file, camera_path
                )
            else:
                chunk_dict["photos"][camera_idx].path = camera_path

    chunk_dict["crs"] = _decode_chunk_reference_tag(xml_tree.findall("./reference"))

    return chunk_dict


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
    raise error if not a metashape projects (missing some files or path not found)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find Metashape project file [{path}]")

    folder_path, project_name, ext = _split_project_path(path)
    data_folder = os.path.join(folder_path, project_name + ".files")

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Could not find Metashape project file [{data_folder}]")


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
    transform = idp.reconstruct.ChunkTransform()
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
    # metashape default value
    local_crs = 'LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]'

    if crs_str == local_crs:
        crs_obj = pyproj.CRS.from_dict({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'})
    else:
        crs_obj = pyproj.CRS.from_string(crs_str)

    '''
    sometimes, the Photoscan given CRS string can not transform correctly
    this is the solution for "WGS 84" CRS shown in the previous example
    
    <reference>GEOGCS["WGS 84",DATUM["World Geodetic System 1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9102"]],AUTHORITY["EPSG","4326"]]</reference>
    
    only works when pyproj==2.6.1.post
    
    when pyproj==3.3.1
    crs_obj.datum = 
      DATUM["World Geodetic System 1984",
      ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],
      ID["EPSG",6326]]
    pyproj.CRS.from_epsg(4326).datum = 
      ENSEMBLE["World Geodetic System 1984 ensemble",
      MEMBER["World Geodetic System 1984 (Transit)",
          ID["EPSG",1166]],
      MEMBER["World Geodetic System 1984 (G730)",
          ID["EPSG",1152]],
      MEMBER["World Geodetic System 1984 (G873)",
          ID["EPSG",1153]],
      MEMBER["World Geodetic System 1984 (G1150)",
          ID["EPSG",1154]],
      MEMBER["World Geodetic System 1984 (G1674)",
          ID["EPSG",1155]],
      MEMBER["World Geodetic System 1984 (G1762)",
          ID["EPSG",1156]],
      MEMBER["World Geodetic System 1984 (G2139)",
          ID["EPSG",1309]],
      ELLIPSOID["WGS 84",6378137,298.257223563,
          LENGTHUNIT["metre",1],
          ID["EPSG",7030]],
      ENSEMBLEACCURACY[2.0],
      ID["EPSG",6326]]
    '''
    crs_wgs84 = pyproj.CRS.from_epsg(4326)

    obj_d = crs_obj.datum.to_json_dict()
    # {'$schema': 'https://proj.org/sch...chema.json', 
    # 'type': 'GeodeticReferenceFrame', 
    # 'name': 'World Geodetic System 1984', 
    # 'ellipsoid': {'name': 'WGS 84', 'semi_major_axis': 6378137, 'inverse_flattening': 298.257223563}, 
    # 'id': {'authority': 'EPSG', 'code': 6326}}

    wgs84_d = crs_wgs84.datum.to_json_dict()
    # {'$schema': 'https://proj.org/sch...chema.json', 
    # 'type': 'DatumEnsemble', 
    # 'name': 'World Geodetic Syste...4 ensemble', 
    # 'members': [{...}, {...}, {...}, {...}, {...}, {...}, {...}], 
    # 'ellipsoid': {'name': 'WGS 84', 'semi_major_axis': 6378137, 'inverse_flattening': 298.257223563}, 
    # 'accuracy': '2.0', 
    # 'id': {'authority': 'EPSG', 'code': 6326}}

    if  obj_d["id"] == wgs84_d["id"] and \
        obj_d["ellipsoid"] == wgs84_d["ellipsoid"]:
        return crs_wgs84
    else:
        return crs_obj


def _decode_sensor_tag(xml_obj, debug_meta={}):
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
    sensor = idp.reconstruct.Sensor()

    sensor.id = int(xml_obj.attrib["id"])
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

    calib_tag = xml_obj.findall("./calibration")
    if len(calib_tag) != 1:
        # load the debug info
        if len(debug_meta) == 0:  # not specify input
            debug_meta = {
                "project_folder": 'project_folder', 
                "project_name"  : 'project_name',
                "chunk_id": 'chunk_id',
                "chunk_path": 'chunk_id/chunk.zip'
            }

        xml_str = minidom.parseString(ElementTree.tostring(xml_obj)).toprettyxml(indent="  ")
        # remove the first line <?xml version="1.0" ?> and empty lines
        xml_str = os.linesep.join([s for s in xml_str.splitlines() if s.strip() and '?xml version=' not in s])
        warnings.warn(f"The sensor tag in [{debug_meta['chunk_path']}] has {len(calib_tag)} <calibration> tags, but expected 1\n"
                      f"\n{xml_str}\n\nThis may cause by importing photos but delete them before align processing in metashape, "
                      f"and leave the 'ghost' empty sensor tag, this is just a warning and should have no effect to you")
        sensor.calibration = None
    else:
        sensor.calibration = _decode_calibration_tag(xml_obj.findall("./calibration")[0])
        sensor.calibration.sensor = sensor

    return sensor


def _decode_calibration_tag(xml_obj):
    """
    Parameters
    ----------
    xml_obj: xml.etree.ElementTree() object
        one element of sensor_tag.findall("./calibration")
    Returns
    -------
    calibration: easyidp.Calibration object

    Notes
    -----

    Calibration tag example 1:

    .. code-block:: xml

        <calibration type="frame" class="adjusted">
            <resolution width="5472" height="3648"/>
            <f>3648</f>
            <k1>-0.01297256557769369</k1>
            <k2>0.00071786332278828374</k2>
            <k3>0.007914514308754287</k3>
            <p1>-0.0011379213280524752</p1>
            <p2>-0.0014457166089453365</p2>
        </calibration>

    Calibration tag example 2:

    .. code-block:: xml

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

    """
    calibration = idp.reconstruct.Calibration()

    calibration.f = float(xml_obj.findall("./f")[0].text)
    calibration.f_unit = "px"

    try:
        calibration.cx = float(xml_obj.findall("./cx")[0].text)
    except IndexError:  # example 1, no cx tag
        calibration.cx = 0
    calibration.cx_unit = "px"

    try:
        calibration.cy = float(xml_obj.findall("./cy")[0].text)
    except IndexError:  # example 1, no cx tag
        calibration.cy = 0
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

    Notes
    -----

    Camera Tag example 1:

    .. code-block:: xml

        <camera id="0" sensor_id="0" label="DJI_0151">
            <transform>
            0.99893511, -0.04561155,  0.00694542, -5.50542042
            -0.04604262, -0.9951647 ,  0.08676   , 13.25994938
            0.00295458, -0.0869874 , -0.99620503,  2.15491524
            0.        ,  0.        ,  0.        ,  1.         
            </transform>  // 16 numbers
            <rotation_covariance>5.9742650282250832e-04 ... 2.3538470709659123e-04</rotation_covariance>  // 9 numbers
            <location_covariance>2.0254219245789448e-02 ... 2.6760756179895751e-02</location_covariance>  // 9 numbers
            <orientation>1</orientation>

            // sometimes have following reference tag, otherwise need to look into frames.zip xml
            <reference x="139.540561166667" y="35.73454525" z="134.765" yaw="164.1" pitch="0"
                        roll="-0" enabled="true" rotation_enabled="false"/>
        </camera>

    Camera tag example 2:
    
    some camera have empty tags, also need to deal with such situation

    .. code-block:: xml

        <camera id="254" sensor_id="0" label="DJI_0538.JPG">
          <orientation>1</orientation>
        </camera>

    Camera tag example 3:

    .. code-block:: xml

        <camera id="62" sensor_id="0" label="DJI_0121">
            <orientation>1</orientation>
            <reference x="139.54051557" y="35.739036839999997" z="106.28" yaw="344.19999999999999" pitch="0" roll="-0" enabled="true" rotation_enabled="false"/>
        </camera>

    Returns
    -------
    camera: easyidp.Photo object
    """
    camera = idp.reconstruct.Photo()
    camera.id = int(xml_obj.attrib["id"])
    camera.sensor_id = int(xml_obj.attrib["sensor_id"])
    camera.label = xml_obj.attrib["label"]
    camera.orientation = int(xml_obj.findall("./orientation")[0].text)
    #camera.enabled = bool(xml_obj.findall("./reference")[0].attrib["enabled"])

    # deal with camera have empty tags
    transform_tag = xml_obj.findall("./transform")
    if len(transform_tag) == 1:
        transform_str = transform_tag[0].text
        camera.transform = np.fromstring(transform_str, sep=" ", dtype=np.float).reshape((4, 4))
    else:
        # have no transform, can not do the reverse caluclation
        camera.enabled = False

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
        # here need to resolve absolute path
        # <photo path="../../../../source/220613_G_M600pro/DSC06035.JPG">
        # this is the root to 220613_G_M600pro.files\0\0\frame.zip"
        # not the real path
        camera_path = camera_tag[0].attrib["path"]
        camera_meta[camera_idx] = camera_path

    return camera_meta, marker_meta


###############
# calculation #
###############

def apply_transform_matrix(points_xyz, matrix):
    """
    Transforms a point or points in homogeneous coordinates.
    equal to Metashape.Matrix.mulp() or Metashape.Matrix.mulv()

    Parameters
    ----------
    matrix: np.ndarray
        4x4 transform numpy array
    points_df: np.ndarray
        # 1x3 single point
        >>> np.array([1,2,3])
           x  y  z
        0  1  2  3

        # nx3 points
        >>> np.array([[1,2,3], [4,5,6], ...])
           x  y  z
        0  1  2  3
        1  4  5  6
        ...

    Returns
    -------
    out: pd.DataFrame
        same size as input points_np
           x  y  z
        0  1  2  3
        1  4  5  6
    """
    points_xyz, is_single = _is_single_point(points_xyz)

    point_ext = np.insert(points_xyz, 3, 1, axis=1)
    dot_matrix = point_ext.dot(matrix.T)
    dot_points = dot_matrix[:, 0:3] / dot_matrix[:, 3][:, np.newaxis]

    if is_single:
        return dot_points[0,:]
    else:
        return dot_points


def convert_proj3d(points_np, crs_origin, crs_target, is_xyz=True):
    """
    Transform a point or points from one CRS to another CRS, by pyproj.CRS.Transformer function

    Parameters
    ----------
    points_np : np.ndarray
        the nx3 3D coordinate points
    crs_origin : pyproj.CRS object
        the CRS of points_np
    crs_target : pyproj.CRS object
        the CRS of target
    is_xyz: bool, default false
        The format of points_np; 
        True: x, y, z; False: lon, lat, alt

    Returns
    -------
    np.ndarray

    Notes
    -----
    ``point_np`` and ``fmt`` parameters

    .. tab:: is_xyz = True

        points_np in this format:

        .. code-block:: text

               x  y  z
            0  1  2  3

    .. tab:: is_xyz = False

        points_np in this format:

        .. code-block:: text

                lon  lat  alt
            0    1    2    3
            1    4    5    6

    .. caution::

        pyproj.CRS order: (lat, lon, alt)
        points order in EasyIDP are commonly (lon, lat, alt)

        But if is xyz format, no need to change order

    """
    ts = pyproj.Transformer.from_crs(crs_origin, crs_target)

    points_np, is_single = _is_single_point(points_np)

    # check unit to know if is (lon, lat, lat) -> degrees or (x, y, z) -> meters
    x_unit = crs_origin.coordinate_system.axis_list[0].unit_name
    y_unit = crs_origin.coordinate_system.axis_list[1].unit_name
    if x_unit == "degree" and y_unit == "degree": 
        is_xyz = False
    else:
        is_xyz = True

    if is_xyz:
        if crs_target.is_geocentric:
            x, y, z = ts.transform(*points_np.T)
            out =  np.vstack([x, y, z]).T
        elif crs_target.is_geographic:
            lon, lat, alt = ts.transform(*points_np.T)
            # the pyproj output order is reversed
            out = np.vstack([lat, lon, alt]).T
        elif crs_target.is_projected:
            lat_m, lon_m, alt_m = ts.transform(*points_np.T)
            out = np.vstack([lat_m, lon_m, alt_m]).T
        else:
            raise TypeError(f"Given crs is neither `crs.is_geocentric=True` nor `crs.is_geographic` nor `crs.is_projected`")
    else:   
        lon, lat, alt = points_np[:,0], points_np[:,1], points_np[:,2]
        
        if crs_target.is_geocentric:
            x, y, z = ts.transform(lat, lon, alt)
            out = np.vstack([x, y, z]).T
        elif crs_target.is_geographic:
            lat, lon, alt = ts.transform(lat, lon, alt)
            out = np.vstack([lon, lat, alt]).T
        elif crs_target.is_projected and crs_target.is_derived:
            lat_m, lon_m, alt_m = ts.transform(lat, lon, alt)
            out = np.vstack([lon_m, lat_m, alt_m]).T
        else:
            raise TypeError(f"Given crs is neither `crs.is_geocentric=True` nor `crs.is_geographic` nor `crs.is_projected`")
    
    if is_single:
        return out[0, :]
    else:
        return out

def _is_single_point(points_np):
    # check if only contains one point
    if points_np.shape == (3,):
        # with only single point
        return np.array([points_np]), True
    else:
        return points_np, False