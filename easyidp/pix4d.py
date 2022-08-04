import os
import numpy as np
import warnings
from pathlib import Path

import easyidp as idp

class Pix4D(idp.reconstruct.Recons):

    def __init__(self, project_path=None, raw_img_folder=None, param_folder=None):
        super().__init__()
        self.software = "pix4d"

        if project_path is not None:
            self.open_project(project_path, raw_img_folder, param_folder)

    def open_project(self, project_path, raw_img_folder=None, param_folder=None):

        p4d_dict = parse_p4d_project(str(project_path), param_folder)

        # project / chunk name
        self.label = p4d_dict["project_name"]

        # pix4d point cloud offset?
        self.meta["p4d_offset"] = read_xyz(p4d_dict["param"]["xyz"])

        #####################
        # info for Sensor() #
        #####################
        ccp = read_ccp(p4d_dict["param"]["ccp"])
        cicp = read_cicp(p4d_dict["param"]["cicp"])
        ssk = read_cam_ssk(p4d_dict["param"]["ssk"])
        '''
        CCP:
        {'w': 4608, 
         'h': 3456, 
         'Image1.JPG': 
            {'cam_matrix': array([[...]]),   # (3x3) K
             'rad_distort': array([ 0.03833474, ...02049799]),
             'tan_distort': array([0.00240852, 0...00292562]), 
             'cam_pos': array([ 21.54872207,...8570281 ]),  (3x1) t
             'cam_rot': array([[ 0.78389904,...99236  ]])}, (3x3) R
             
         'Image2.JPG':
            ...
        }


        CICP:
        {'w_mm','h_mm', 'F', 'Px', 'Py', 'K1', 'K2', 'K3', 'T1', 'T2'}
        #Focal Length mm assuming a sensor width of 17.49998592000000030566x13.12498944000000200560mm
        F 15.01175404934517487732
        #Principal Point mm
        Px 8.48210511970419922534
        Py 6.33434629978042273990
        #Symmetrical Lens Distortion Coeffs
        K1 0.03833474118270804865
        K2 -0.01750917966495743258
        K3 0.02049798716391852335
        #Tangential Lens Distortion Coeffs
        T1 0.00240851666319534747
        T2 0.00292562392135245920
        '''

        sensor = idp.reconstruct.Sensor()
        # seems pix4d only support one camera kind
        sensor.id = 0
        sensor.label = ssk["label"]
        sensor.type = ssk["type"]

        sensor.h_mm = cicp["h_mm"]
        sensor.w_mm = cicp["w_mm"]
        sensor.focal_length = cicp["F"]

        sensor.height = ccp["h"]
        sensor.width = ccp["w"]

        # ensure the order is correct
        # ssk -> image_size_in_pixels: 3456 4608
        # it should be the orientation == 1
        if ssk["image_size_in_pixels"] == [ccp["h"], ccp["w"]]:
            sensor.pixel_size = ssk["pixel_size"]
            # the scale of each pixel, unit is mm
            sensor.pixel_height, sensor.pixel_width = ssk["pixel_size"]

            # calibration params
            sensor.calibration.cy, sensor.calibration.cx = ssk["photo_center_in_pixels"]
        # the order is reversed, probably orientatio == 0
        elif ssk["image_size_in_pixels"] == [ccp["w"], ccp["h"]]:
            warnings.warn(f"It seems the orientation = {ssk['orientation']} and the height and width are reversed")
            sensor.pixel_size = ssk["pixel_size"][::-1]   # reverse
            sensor.pixel_width, sensor.pixel_height = ssk["pixel_size"]
            sensor.calibration.cx, sensor.calibration.cy = ssk["photo_center_in_pixels"]
        else:
            raise NotImplementedError("Have no examples with sensor orientation != 1")

        # save calibration values, pix4d unit is pixel for this value
        sensor.calibration.software = self.software
        sensor.calibration.type = sensor.type
        sensor.calibration.f = sensor.focal_length * sensor.width / sensor.w_mm
        sensor.calibration.k1 = cicp["K1"]
        sensor.calibration.k2 = cicp["K2"]
        sensor.calibration.k3 = cicp["K3"]

        sensor.calibration.t1 = cicp["T1"]
        sensor.calibration.t2 = cicp["T2"]

        self.sensors[0] = sensor

        ####################
        # info for Photo() #
        ####################
        pmat = read_pmat(p4d_dict["param"]["pmat"])

        for i, img_label in enumerate(pmat.keys()):
            img = idp.reconstruct.Photo(self.sensors[0])
            img.id = i
            img.label = img_label
            img.sensor_id = 0
            img.enabled = True

            # find raw image path
            if raw_img_folder is not None:
                img_list = os.listdir(raw_img_folder)
                if img_label in img_list:
                    img_full_path = os.path.join(raw_img_folder, img_label)
                    img.path = idp.get_full_path(img_full_path)
                else:
                    warnings.warn(
                        f"Could not find [{img_label}] in given raw_img_folder"
                        "[{raw_img_folder}]"
                    )

            # pix4d reverse calculation only need pmatrix
            img.transform = pmat[img_label]  # pmatrix

            # and just in case need this:
            img.cam_matrix = ccp[img_label]["cam_matrix"]  # K
            img.location = ccp[img_label]["cam_pos"]  # t
            img.rotation = ccp[img_label]["cam_rot"]   # R

            self.photos[i] = img

        ######################
        # info for seelf.CRS #
        ######################
        self.crs = idp.shp.read_proj(p4d_dict["param"]["crs"])

        ################################
        # info for outpus(PCD|DOM|DSM) #
        ################################
        if isinstance(p4d_dict["pcd"], (Path, str)) and Path(p4d_dict["pcd"]).exists():
            self.load_pcd(p4d_dict["pcd"])

        if isinstance(p4d_dict["dom"], (Path, str)) and Path(p4d_dict["dom"]).exists():
            self.load_dom(p4d_dict["dom"])

        if isinstance(p4d_dict["dsm"], (Path, str)) and Path(p4d_dict["dsm"]).exists():
            self.load_dsm(p4d_dict["dsm"])

    def load_pcd(self, pcd_path):
        """Manual load the point cloud file generated by this Pix4D project

        Parameters
        ----------
        pcd_path : str
            The path to point cloud file

        Caution
        -------
        The pix4d produced point cloud is offsetted (xyz-offset)
        This function already handle adding offset back to point cloud
        It you need manual specify ``idp.PointCloud()`` by yourself, please do:

        .. code-block:: python

            >>> p4d = idp.Pix4D(project_path, param_folder)
            # wrong
            >>> pcd = idp.PointCloud(lotus_full_pcd)
            # correct
            >>> pcd = idp.PointCloud(lotus_full_pcd, offset=p4d.meta['p4d_offset'])

        Examples
        --------

        .. code-block:: python

            >>> import easyidp as idp
            >>> ... # define your project_path and param_folder path
            >>> p4d = idp.Pix4D(project_path, param_folder)
            >>> p4d.pcd
            None
            >>> p4d.load_pcd(pcd_path)
            >>> p4d.pcd
            <easyidp.PointCloud> object

        """
        if os.path.exists(pcd_path):
            self.pcd = idp.PointCloud(pcd_path, offset=self.meta["p4d_offset"])

    def load_dom(self, geotiff_path):
        """Manual load the DOM file generated by this Pix4D project

        Parameters
        ----------
        geotiff_path : str
            The path to DOM file

        Examples
        --------

        .. code-block:: python

            >>> import easyidp as idp
            >>> ... # define your project_path and param_folder path
            >>> p4d = idp.Pix4D(project_path, param_folder)
            >>> p4d.dom
            None
            >>> p4d.load_dom(dom_path)
            >>> p4d.dom
            <easyidp.GeoTiff> object

        """
        if os.path.exists(geotiff_path):
            self.dom = idp.GeoTiff(geotiff_path)

    def load_dsm(self, geotiff_path):
        """Manual load the DSM file generated by this Pix4D project

        Parameters
        ----------
        geotiff_path : str
            The path to DSM file

        Examples
        --------

        .. code-block:: python

            >>> import easyidp as idp
            >>> ... # define your project_path and param_folder path
            >>> p4d = idp.Pix4D(project_path, param_folder)
            >>> p4d.dsm
            None
            >>> p4d.load_dsm(dsm_path)
            >>> p4d.dsm
            <easyidp.GeoTiff> object

        """
        if os.path.exists(geotiff_path):
            self.dsm = idp.GeoTiff(geotiff_path)

    ################################
    # code for reverse calculation #
    ################################
    def _check_photo_type(self, photo):
        if isinstance(photo, str):
            if photo in self.photos.keys():
                return self.photos[photo]
            else:
                raise KeyError(
                    f"Could not find given image name [{photo}] in "
                    f"[{self.photos[0].label}, {self.photos[1].label}, ..., {self.photos[-1].label}]")
        elif isinstance(photo, idp.reconstruct.Photo):
            return photo
        else:
            raise TypeError("Only support <easyidp.Photo> or image_name string")

    def _external_internal_calc(self, points, photo, distort_correct=True):
        """Calculate backward projection by camera parameters, seems not correct, deprecated.

        Parameters
        ----------
        points : np.ndarray
            the nx3 xyz numpy matrix
        photo : str | easyidp.Photo object
            if str -> photo name / keys
        distort_correct : bool, optional
            whether calibrate lens distortion, by default True

        Returns
        -------
        coords_b
            2d ndarray, 'lower-left' coordiantes
        """
        photo = self._check_photo_type(photo)
        T = photo.location   # v1.0: T = param.img[image_name].cam_pos
        R = photo.rotation   # v1.0: R = param.img[image_name].cam_rot

        X_prime = (points - T).dot(R)
        xh = X_prime[:, 0] / X_prime[:, 2]
        yh = X_prime[:, 1] / X_prime[:, 2]

        # olderversion
        # f = param.F * param.img[image_name].w / param.w_mm
        # cx = param.Px * param.img[image_name].w / param.w_mm
        # cy = param.Py * param.img[image_name].h / param.h_mm
        calibration = self.sensors[photo.sensor_id].calibration

        if distort_correct:
            xb, yb = calibration.calibrate(xh, yh)
        else:
            f = calibration.f
            cx = calibration.cx
            cy = calibration.cy

            xb = f * xh + cx
            yb = f * yh + cy

        #xa = xb
        #ya = param.img[image_name].h - yb
        #coords_a = np.hstack([xa[:, np.newaxis], ya[:, np.newaxis]])
        #coords_b = np.hstack([xb[:, np.newaxis], yb[:, np.newaxis]])
        return np.vstack([xb, yb]).T


    def _pmatrix_calc(self, points, photo, distort_correct=True):
        """Calculate backward projection by pix4d pmatrix

        Parameters
        ----------
        points : np.ndarray
            the nx3 xyz numpy matrix
        photo : str | easyidp.Photo object
            if str -> photo name / keys
        distort_correct : bool, optional
            whether calibrate lens distortion, by default True

        Returns
        -------
        coords_b
            2d ndarray, 'lower-left' coordiantes
        """
        photo = self._check_photo_type(photo)

        xyz1_prime = np.insert(points, 3, 1, axis=1)
        xyz = (xyz1_prime).dot(photo.transform.T)  # v1.0: param.img[image_name].pmat.T
        u = xyz[:, 0] / xyz[:, 2]   # already the pixel coords
        v = xyz[:, 1] / xyz[:, 2]

        calibration = self.sensors[photo.sensor_id].calibration
        if distort_correct:
            xh, yh = calibration.calibrate(u, v)
            coords_b = np.vstack([xh, yh]).T
        else:
            coords_b = np.vstack([u, v]).T

        return coords_b


    def back2raw_crs(self, points_xyz, distort_correct=True, ignore=None, save_folder=None, log=False):
        """Projects one GIS coordintates ROI (polygon) to all images

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
        save_folder : str | default ""
            The folder to contain the output results (preview images and json coords)
        log : bool, optional
            whether print log for debugging, by default False

        Returns
        -------
        dict,
            a dictionary that key = img_name and values= pixel coordinate
        """
        out_dict = {}
        sensor = self.sensors[0]
        # seems all teh calculation is based on no offset coordinate
        points_xyz = points_xyz - self.meta["p4d_offset"]

        for photo_name, photo in self.photos.items():
            if log:
                print(f'[Calculator][Judge]{photo.label}w:{photo.w}h:{photo.h}->', end='')
            #if method == 'exin':
            #    projected_coords = self._external_internal_calc(points, photo, distort_correct)
            projected_coords = self._pmatrix_calc(points_xyz, photo, distort_correct)
            coords = sensor.in_img_boundary(projected_coords, ignore=ignore, log=log)
            if coords is not None:
                out_dict[photo.label] = coords

        if isinstance(save_folder, str) and os.path.isdir(save_folder):
            # if not os.path.exists(save_folder):
            #     os.makedirs(save_folder)
            # save to json file
            # save to one image file ()
            pass

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
        for k, points_xyz in roi.items():
            if isinstance(save_folder, str) and os.path.isdir(save_folder):
                save_path = os.path.join(save_folder, k)
            else:
                save_path = None

            one_roi_dict= self.back2raw_crs(points_xyz, save_folder=save_path, **kwargs)

            out_dict[k] = one_roi_dict

        return out_dict


####################
# code for file IO #
####################

def _match_suffix(folder, ext):
    """
    find the *first* file by given suffix, e.g. *.tif
    Parameters
    ----------
    folder: str
        the path of folder want to find
    ext: str | list
        str -> e.g. "tif"
        list -> e.g. ["ply", "laz"]
            will find all suffix files

    Returns
    -------
    find_path: str | None
        if find -> str of path
        if not find -> None
    """
    find_path = None
    if Path(folder).exists():
        if isinstance(ext, str):
            one_ext = True
        else:
            one_ext = False

        if one_ext:
            for file in os.listdir(folder):
                if file.endswith(ext):
                    find_path = f"{folder}/{file}"
                    return find_path
        else:
            for ex in ext:
                for file in os.listdir(folder):
                    if file.endswith(ex):
                        find_path = f"{folder}/{file}"
                        return find_path

    return find_path


def parse_p4d_param_folder(param_path:str):
    param_dict = {}
    # keys = ["project_name", "xyz", "pmat", "cicp", "ccp"]

    param_files = os.listdir(param_path)

    if len(param_files) < 6:
        raise FileNotFoundError(
            f"Given param folder [{idp.get_full_path(param_path)}] "
            "does not have enough param files to parse"
        )

    project_name = os.path.commonprefix(param_files)
    # > "maize_tanashi_3NA_20190729_Ins1Rgb_30m_pix4d_"
    if project_name[-1] == '_':
        project_name = project_name[:-1]
    param_dict["project_name"] = project_name

    xyz_file = f"{param_path}/{project_name}_offset.xyz"
    if os.path.exists(xyz_file):
        param_dict["xyz"] = xyz_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{xyz_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    pmat_file = f"{param_path}/{project_name}_pmatrix.txt"
    if os.path.exists(pmat_file):
        param_dict["pmat"] = pmat_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{pmat_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    cicp_file = f"{param_path}/{project_name}_pix4d_calibrated_internal_camera_parameters.cam"
    # two files with the same string
    # {project_name}_      calibrated_internal_camera_parameters.cam
    # {project_name}_pix4d_calibrated_internal_camera_parameters.cam
    if os.path.exists(cicp_file):
        param_dict["cicp"] = cicp_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{cicp_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    ccp_file = f"{param_path}/{project_name}_calibrated_camera_parameters.txt"
    if os.path.exists(ccp_file):
        param_dict["ccp"] = ccp_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{ccp_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    campos_file = f"{param_path}/{project_name}_calibrated_images_position.txt"
    if os.path.exists(campos_file):
        param_dict["campos"] = campos_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{campos_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    ssk_file = f"{param_path}/{project_name}_camera.ssk"
    if os.path.exists(ssk_file):
        param_dict["ssk"] = ssk_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{ssk_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    prj_file = f"{param_path}/{project_name}_wkt.prj"
    if os.path.exists(prj_file):
        param_dict["crs"] = prj_file
    else:
        raise FileNotFoundError(
            f"Could not find param file [{ssk_file}] in param folder [{param_path}]"
            "please check whether the `param folder` has correct path."
        )

    return param_dict


def parse_p4d_project(project_path:str, param_folder=None):
    """
    A fuction to automatically analyze related subfiles in pix4d project folder

    Parameters
    ----------
    project_path: str
        the path to pix4d project file, that folder should contains the following sub-folder:

        \project_path
        |--- 1_initial\
        |--- 2_densification\
        |___ 3_dsm_ortho\

    param_folder: str, default None
        if not given, it will parse as a standard pix4d project, and trying
            to get the project name from `1_initial/param` folder
        if it is not a standard pix4d project (re-orgainzed folder), need manual
            specify the path to param folder, in order to parse project_name
            for later usage.

    Returns
    -------
    p4d: dict
        a python dictionary that contains the path to each file.
        {
            "project_name": the prefix of whole project file.
            "param": the folder of parameters
            "pcd": the point cloud file
            "dom": the digital orthomosaic file
            "dsm": the digital surface model file
            "undist_raw": the undistorted images corrected by the pix4d software (when original image unable to find)
        }

    Notes
    -----
    Project_name can be extracted from parameter folder prefix in easyidp 2.0, no need manual specify.
    To find the outputs, it will pick the first file that fits the expected file format.
    """
    p4d = {"param": None, "pcd": None, "dom": None, "dsm": None, "undist_raw": None, "project_name": None}

    project_path = idp.get_full_path(project_path)
    sub_folder = os.listdir(project_path)

    #################################
    # parse defalt 1_initial folder #
    #################################
    if param_folder is None:
        param_folder = project_path / "1_initial" / "params"

    undist_folder = project_path / "1_initial" / "images" / "undistorted_images"

    # check whether a correct pix4d project folder
    if '1_initial' not in sub_folder and param_folder is None:
        raise FileNotFoundError(f"Current folder [{project_path}] is not a standard pix4d projects folder, please manual speccify `param_folder`")

    if os.path.exists(param_folder) and len(os.listdir(param_folder)) > 0:
        param = parse_p4d_param_folder(param_folder)
        p4d["param"] = param
        project_name = param["project_name"]
        p4d["project_name"] = param["project_name"]
    else:
        raise FileNotFoundError(
            "Can not find pix4d parameter in given project folder"
        )

    if os.path.exists(undist_folder):
        p4d["undist_raw"] = undist_folder

    ######################
    # parse output files #
    ######################

    # point cloud file
    pcd_folder = project_path / "2_densification" / "point_cloud"
    ply_file   = pcd_folder / f"{project_name}_group1_densified_point_cloud.ply"
    laz_file   = pcd_folder / f"{project_name}_group1_densified_point_cloud.laz"
    las_file   = pcd_folder / f"{project_name}_group1_densified_point_cloud.las"

    if pcd_folder.exists():
        if ply_file.exists():
            p4d["pcd"] = ply_file
        elif las_file.exists():
            p4d["pcd"] = las_file
        elif laz_file.exists():
            p4d["pcd"] = laz_file
        else:
            force = _match_suffix(pcd_folder, ["ply", "las", "laz"])
            if force is not None:
                p4d["pcd"] = force
            else:
                warnings.warn(
                    f"Unable to find any point cloud output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{pcd_folder}]. Please specify manually."
                )

    # digital surface model DSM file
    dsm_folder = project_path / "3_dsm_ortho" / "1_dsm"
    dsm_file = dsm_folder / f"{project_name}_dsm.tif"

    if dsm_folder.exists():
        if dsm_file.exists():
            p4d["dsm"] = dsm_file
        else:
            force = _match_suffix(dsm_folder, "tif")
            if force is not None:
                p4d["dsm"] = force
            else:
                warnings.warn(
                    f"Unable to find any DSM output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{dense_folder}]. Please specify manually."
                )

    dom_folder = project_path / "3_dsm_ortho" / "2_mosaic"
    dom_file = dom_folder / f"{project_name}_transparent_mosaic_group1.tif"
    if dom_folder.exists():
        if dom_file.exists():
            p4d["dom"] = dom_file
        else:
            force = _match_suffix(dom_folder, "tif")
            if force is not None:
                p4d["dom"] = force
            else:
                warnings.warn(
                    f"Unable to find any DOM output file "
                    "[*.ply, *.las, *.laz] in the project folder "
                    "[{dense_folder}]. Please specify manually."
                )

    return p4d


def read_xyz(xyz_path):
    """
    read pix4d file PROJECTNAME_offset.xyz
    Parameters
    ----------
    xyz_path: str
        the path to target offset.xyz file

    Returns
    -------
    x, y, z: float
    """
    with open(xyz_path, 'r') as f:
        x, y, z = f.read().split(' ')
    return np.array([float(x), float(y), float(z)])


def read_pmat(pmat_path):
    """
    read pix4d file PROJECTNAME_pmatrix.txt
    Parameters
    ----------
    pmat_path: str
        the path of pmatrix file.type

    Returns
    -------
    pmat_dict: dict
        pmat_dict = {"DJI_0000.JPG": nparray(3x4), ... ,"DJI_9999.JPG": nparray(3x4)}

    """
    pmat_nb = np.loadtxt(pmat_path, dtype=np.float, delimiter=None, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,))
    pmat_names = np.loadtxt(pmat_path, dtype=str, delimiter=None, usecols=0)

    pmat_dict = {}

    for i, name in enumerate(pmat_names):
        pmat_dict[name] = pmat_nb[i, :].reshape((3, 4))

    return pmat_dict


def read_cicp(cicp_path):
    """Read PROJECTNAME_pix4d_calibrated_internal_camera_parameters.cam file
    Used for Sensor() object

    Parameters
    ----------
    cicp_path: str

    Returns
    -------
    cicp_dict: dict

    Notes
    -----
    It is the info about sensor
    maize_cicp.txt:
        Pix4D camera calibration file 0
        #Focal Length mm assuming a sensor width of 17.49998592000000030566x13.12498944000000200560mm
        F 15.01175404934517487732
        #Principal Point mm
        Px 8.48210511970419922534
        Py 6.33434629978042273990
        #Symmetrical Lens Distortion Coeffs
        K1 0.03833474118270804865
        K2 -0.01750917966495743258
        K3 0.02049798716391852335
        #Tangential Lens Distortion Coeffs
        T1 0.00240851666319534747
        T2 0.00292562392135245920
    """
    with open(cicp_path, 'r') as f:
        key_pool = ['F', 'Px', 'Py', 'K1', 'K2', 'K3', 'T1', 'T2']
        cam_dict = {}
        for line in f.readlines():
            sp_list = line.split(' ')
            if len(sp_list) == 2:  # lines with param.name in front
                lead, contents = sp_list[0], sp_list[1]
                if lead in key_pool:
                    cam_dict[lead] = float(contents[:-1])
            elif len(sp_list) == 9:
                # one example:
                # > Focal Length mm assuming a sensor width of 12.82x8.55mm\n
                w_h = sp_list[8].split('x')
                cam_dict['w_mm'] = float(w_h[0])  # extract e.g. 12.82
                cam_dict['h_mm'] = float(w_h[1][:-4])  # extract e.g. 8.55
    return cam_dict


def read_ccp(ccp_path):
    """Read PROJECTNAME_calibrated_camera_parameters.txt
    Used for Photo() object

    Parameters
    ----------
    ccp_path: str

    Returns
    -------
    img_configs: dict
        {'w': 4608, 
         'h': 3456, 
         'Image1.JPG': 
            {'cam_matrix': array([[...]]), 
             'rad_distort': array([ 0.03833474, ...02049799]),
             'tan_distort': array([0.00240852, 0...00292562]), 
             'cam_pos': array([ 21.54872207,...8570281 ]), 
             'cam_rot': array([[ 0.78389904,...99236  ]])}, 
             
         'Image2.JPG':
            ...
        }

    Notes
    -----
    It is the camera position info
    """
    with open(ccp_path, 'r') as f:
        '''
        # for each block
        1   fileName imageWidth imageHeight 
        2-4 camera matrix K [3x3]
        5   radial distortion [3x1]
        6   tangential distortion [2x1]
        7   camera position t [3x1]
        8-10   camera rotation R [3x3]
        '''
        lines = f.readlines()

    img_configs = {}

    file_name = ""
    cam_mat_line1 = ""
    cam_mat_line2 = ""
    cam_rot_line1 = ""
    cam_rot_line2 = ""
    for i, line in enumerate(lines):
        if i < 8:
            pass
        else:
            block_id = (i - 7) % 10
            if block_id == 1:  # [line]: fileName imageWidth imageHeight
                file_name, w, h = line[:-1].split()  # ignore \n character
                img_configs[file_name] = {}
                img_configs['w'] = int(w)
                img_configs['h'] = int(h)
            elif block_id == 2:
                cam_mat_line1 = np.fromstring(line, dtype=np.float, sep=' ')
            elif block_id == 3:
                cam_mat_line2 = np.fromstring(line, dtype=np.float, sep=' ')
            elif block_id == 4:
                cam_mat_line3 = np.fromstring(line, dtype=np.float, sep=' ')
                img_configs[file_name]['cam_matrix'] = np.vstack([cam_mat_line1, cam_mat_line2, cam_mat_line3])
            elif block_id == 5:
                img_configs[file_name]['rad_distort'] = np.fromstring(line, dtype=np.float, sep=' ')
            elif block_id == 6:
                img_configs[file_name]['tan_distort'] = np.fromstring(line, dtype=np.float, sep=' ')
            elif block_id == 7:
                img_configs[file_name]['cam_pos'] = np.fromstring(line, dtype=np.float, sep=' ')
            elif block_id == 8:
                cam_rot_line1 = np.fromstring(line, dtype=np.float, sep=' ')
            elif block_id == 9:
                cam_rot_line2 = np.fromstring(line, dtype=np.float, sep=' ')
            elif block_id == 0:
                cam_rot_line3 = np.fromstring(line, dtype=np.float, sep=' ')
                cam_rot = np.vstack([cam_rot_line1, cam_rot_line2, cam_rot_line3])
                img_configs[file_name]['cam_rot'] = cam_rot

    return img_configs


def read_campos_geo(campos_path):
    """Read PROJECTNAME_calibrated_images_position.txt
    Used for Photo.location()? -> geo_location?

    Parameters
    ----------
    campos_path : str

    Notes
    -----
    the geo position of each camera
        DJI_0954.JPG,368030.548722,3955824.412658,127.857028
        DJI_0955.JPG,368031.004387,3955824.824967,127.381322
        DJI_0956.JPG,368033.252520,3955826.479610,127.080709
        DJI_0957.JPG,368032.022104,3955826.060493,126.715974
        DJI_0958.JPG,368031.901165,3955826.109158,126.666393
        DJI_0959.JPG,368030.686490,3955830.981070,127.327741

    Returns
    -------
    campos_dict:
        {"Image1.JPG": np.array([x, y ,z]), 
         "Image2.JPG": ...
         ...}
    """
    with open(campos_path, 'r') as f:
        cam_dict = {}
        for line in f.readlines():
            sp_list = line.split(',')
            if len(sp_list) == 4: 
                cam_dict[sp_list[0]] = np.array(sp_list[1:], dtype=np.float)

    return cam_dict


def read_cam_ssk(ssk_path):
    """Get the camera model name, used for Sensor() object

    Parameters
    ----------
    ssk_path : str

    Returns
    -------
    ssk_info : dict
        {
            "label": str
            "type": str   # frame / fisheye ...
            "pixel_size": [h, w]
            "pixel_size_unit": "mm"
            "image_size_in_pixels": [h ,w]
            "orientation": 1   # guess 0 -> w, h?
            "photo_center_in_pixels": [h, w]
        }

    Notes
    -----
    SSK file contents:
    begin camera_parameters FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456 (RGB)(1)
        focal_length:                      15.00522620622411729130
        ppac:                              0.02793232590500918308 -0.02181191393910364776
        ppbs:                               0 0
        film_format:                       13.12498944000000200560 17.49998592000000030566
        lens_distortion_flag:              off
        io_required:                       yes
        camera_type:                       frame
        media_type:                        digital
        pixel_size:                        3.79774000000000011568 3.79774000000000011568
        image_size_in_pixels:              3456 4608
        scanline_orientation:              4
        photo_coord_sys_orientation:       1
        photo_coord_sys_origin:            1727.50000000000000000000 2303.50000000000000000000
        focal_length_calibration_flag:     off
        calibrated_focal_length_stddev:    0.03
        ppac_calibration_flag:             off
        calibrated_ppac_stddevs:           0.003   0.003
        self_calibration_enabled_params:   0
        antenna_offsets:                   0   0   0
    end camera_parameters
    """
    with open(ssk_path, 'r') as f:
        ssk_info = {}
        for line in f.readlines():
            if "begin camera_parameters" in line:
                # > ['begin', 'camera_parameters', 'FC550_DJIMFT15mmF1.7ASPH_15.0_4608x3456', '(RGB)(1)']
                ssk_info["label"] = line.split(' ')[2]
            elif "camera_type" in line:
                # > ['', 'camera_type:', '', '',... '', '', 'frame\n']
                ssk_info["type"] = str(line.split(' ')[-1][:-1])  # last and rm \n
            elif "pixel_size" in line:
                # > ['', 'pixel_size:', '', ..., '', '3.79774000000000011568', '3.79774000000000011568']
                # the order is h, w, if orientation == 1
                # because: image_size_in_pixels: 3456 4608
                lsp = line.split(' ')
                ssk_info["pixel_size"] = [float(lsp[-2]), float(lsp[-1])]
            elif "image_size_in_pixels" in line:
                # double check with ccp imageWidth & imageHeight
                lsp = line.split(' ')
                ssk_info["image_size_in_pixels"] = [int(lsp[-2]), int(lsp[-1])]
            elif "photo_coord_sys_orientation" in line:
                ssk_info["orientation"] = int(line.split(' ')[-1])
            elif "photo_coord_sys_origin" in line:
                lsp = line.split(' ')
                ssk_info["photo_center_in_pixels"] = [float(lsp[-2]), float(lsp[-1])]

    return ssk_info