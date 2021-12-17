import numpy as np
import json
from easyric.io import shp, json, geotiff

#################
# basic wrapper #
#################

def external_internal_calc(param, points, image_name, distort_correct=True):
    """
    params
        p4d:        the Pix4D configure class
        points:     the nx3 xyz numpy matrix
        image_name: the string of target projection image name

    return
        2d ndarray, 'lower-left' coordiantes

    example
    >>> from calculate.geo2raw external_internal_calc
    >>> coords = external_internal_calc(param, points=shp_mean_dict['2'], image_name=test_img)
    >>> coords[:5,:]
    array([[205.66042808, 364.060308  ],
           [211.74289725, 366.75581059],
           [220.85402254, 365.72989203],
           [231.23327229, 361.99041811],
           [237.64123696, 356.14638116]])
    """

    T = param.img[image_name].cam_pos
    R = param.img[image_name].cam_rot

    X_prime = (points - T).dot(R)
    xh, yh = X_prime[:, 0] / X_prime[:, 2], X_prime[:, 1] / X_prime[:, 2]

    f = param.F * param.img[image_name].w / param.w_mm
    cx = param.Px * param.img[image_name].w / param.w_mm
    cy = param.Py * param.img[image_name].h / param.h_mm

    xb = f * xh + cx
    yb = f * yh + cy

    if distort_correct:
        xb, yb = distortion_correction(param, xb, yb, image_name)

    #xa = xb
    #ya = param.img[image_name].h - yb
    #coords_a = np.hstack([xa[:, np.newaxis], ya[:, np.newaxis]])
    coords_b = np.hstack([xb[:, np.newaxis], yb[:, np.newaxis]])

    return coords_b


def pmatrix_calc(param, points, image_name, distort_correct=True):
    """
    params
        p4d:        the Pix4D configure class
        points:     the nx3 xyz numpy matrix, should be the geo_coordinate - offsets
        image_name: the string of target projection image name

    return
        2d ndarray, 'lower-left' coordiantes

    example
    >>> from calculate.geo2raw import pmatrix_calc
    >>> coords = external_internal_calc(param, points=shp['2'], image_name=test_img)
    >>> coords[:5,:]
    array([[205.66042808, 364.060308  ],
           [211.74289725, 366.75581059],
           [220.85402254, 365.72989203],
           [231.23327229, 361.99041811],
           [237.64123696, 356.14638116]])
    """
    xyz1_prime = np.insert(points, 3, 1, axis=1)
    xyz = (xyz1_prime).dot(param.img[image_name].pmat.T)
    u = xyz[:, 0] / xyz[:, 2]
    v = xyz[:, 1] / xyz[:, 2]
    #coords_a = np.vstack([u, p4d.img[image_name].h - v]).T
    if distort_correct:
        xh, yh = distortion_correction(param, u, v, image_name)
        coords_b = np.vstack([xh, yh]).T
    else:
        coords_b = np.vstack([u, v]).T

    return coords_b


def distortion_correction(param, u, v, image_name):
    """
    Convert pix4d produced undistorted images pixel coordinate to original image pixel coordinate
    :param param: p4d class
    :param u: the x pixel coordinate
    :param v: the y pixel coordinate
    :param image_name: the image name of current image
    :return: px, py, the pixel coordinate on the raw image
    """
    f = param.F * param.img[image_name].w / param.w_mm
    cx = param.Px * param.img[image_name].w / param.w_mm
    cy = param.Py * param.img[image_name].h / param.h_mm

    xh = (u - cx) / f
    yh = (v - cy) / f

    r2 = xh ** 2 + yh ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3
    a1 = 1 + param.K1 * r2 + param.K2 * r4 + param.K3 * r6
    xhd = a1 * xh + 2 * param.T1 * xh * yh + param.T2 * (r2 + 2 * xh ** 2)
    yhd = a1 * yh + 2 * param.T2 * xh * yh + param.T1 * (r2 + 2 * yh ** 2)

    xb = f * xhd + cx
    yb = f * yhd + cy

    return xb, yb

####################
# advanced wrapper #
####################

def in_img_boundary(reprojected_coords, img_size, ignore=None, log=False):
    w, h = img_size
    coord_min = reprojected_coords.min(axis=0)
    x_min, y_min = coord_min[0], coord_min[1]
    coord_max= reprojected_coords.max(axis=0)
    x_max, y_max = coord_max[0], coord_max[1]

    if ignore is None:
        if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
            if log: print('X ', (x_min, x_max, y_min, y_max))
            return None
        else:
            if log: print('O ', (x_min, x_max, y_min, y_max))
            return reprojected_coords
    elif ignore=='x':
        if y_min < 0 or y_max > h:
            if log: print('X ', (x_min, x_max, y_min, y_max))
            return None
        else:
            reprojected_coords[reprojected_coords[:, 0] < 0, 0] = 0
            reprojected_coords[reprojected_coords[:, 0] > w, 0] = w
            if log: print('O ', (x_min, x_max, y_min, y_max))
            return reprojected_coords
    elif ignore=='y':
        if x_min < 0 or x_max > w:
            if log: print('X ', (x_min, x_max, y_min, y_max))
            return None
        else:
            reprojected_coords[reprojected_coords[:, 1] < 0, 1] = 0
            reprojected_coords[reprojected_coords[:, 1] > h, 1] = h
            if log: print('O ', (x_min, x_max, y_min, y_max))
            return reprojected_coords


def get_img_coords_dict(param, points, method='pmat', distort_correct=True, ignore=None, log=False):
    """
    :param param: the p4d project objects
    :param points: should be the geo coordinate - offsets
    :param method: string
        'exin' use external and internal parameters (seems not so accurate in some cases),
        'pmat' use pmatrix to calculate (recommended method for common digital camera, fisheye camera not suit)
    :param log: boolean, whether print logs in console
    :return:
    """
    out_dict = {}
    for im in param.img:
        if log:
            print(f'[Calculator][Judge]{im.name}w:{im.w}h:{im.h}->', end='')
        if method == 'exin':
            projected_coords = external_internal_calc(param, points, im.name, distort_correct)
        else:
            projected_coords = pmatrix_calc(param, points, im.name, distort_correct)
        coords = in_img_boundary(projected_coords, (im.w, im.h), ignore=ignore, log=log)
        if coords is not None:
            out_dict[im.name] = coords

    return out_dict


def filter_closest_img(p4d, img_dict, plot_geo, dist_thresh=None, num=None):
    """[summary]

    Parameters
    ----------
    img_dict : dict
        The outputs dict of geo2raw.get_img_coords_dict()
    plot_geo : nx3 ndarray
        The plot boundary polygon vertex coordinates
    num : None or int
        Keep the closest {x} images
    dist_thresh : None or float
        If given, filter the images smaller than this distance first

    Returns
    -------
    dict
        the same structure as output of geo2raw.get_img_coords_dict()
    """
    dist_geo = []
    dist_name = []
    
    img_dict_sort = {}
    for img_name, img_coord in img_dict.items():

        xmin_geo, ymin_geo = plot_geo[:,0:2].min(axis=0)
        xmax_geo, ymax_geo = plot_geo[:,0:2].max(axis=0)
        xctr_geo = (xmax_geo + xmin_geo) / 2
        yctr_geo = (ymax_geo + ymin_geo) / 2

        ximg_geo, yimg_geo, _ = p4d.img[img_name].cam_pos

        image_plot_dist = np.sqrt((ximg_geo-xctr_geo) ** 2 + (yimg_geo - yctr_geo) ** 2)

        if dist_thresh is not None and image_plot_dist > dist_thresh:
            # skip those image-plot geo distance greater than threshold
            continue
        else:
            # if not given dist_thresh, record all
            dist_geo.append(image_plot_dist)
            dist_name.append(img_name)

    if num is None:
        # not specify num, use all
        num = len(dist_name)
    else:
        num = min(len(dist_name, num))

    dist_geo_idx = np.asarray(dist_geo).argsort()[:num]
    img_dict_sort = {dist_name[idx]:img_dict[dist_name[idx]] for idx in dist_geo_idx}
    
    return img_dict_sort



def get_shp_result(p4d, shp_dict, method='pmat', distort_correct=True, json_path=None, ignore=None):
    #def get_shp_result(p4d, shp_dict, get_z_by="mean", shp_proj=None, geotiff_proj=None, json_path=None):
    result_dict = {}
    #json_dict = {}
    #shp_dict = shp.read_shp3d(shp_path, dsm_path=p4d.dsm_file,
    #                          get_z_by=get_z_by,
    #                          shp_proj=shp_proj, geotiff_proj=geotiff_proj)

    for shp_key in shp_dict.keys():
        result_dict[shp_key] = {}

        points = shp_dict[shp_key]
        #for method in ["exin", "pmat"]:
        result_dict[shp_key][method] = {}

        img_coord_dict = get_img_coords_dict(p4d, points - p4d.offset.np, method, distort_correct, ignore=ignore)
        for im_name, coord in img_coord_dict.items():
            result_dict[shp_key][method][im_name] = coord

    if json_path:
        json.dict2json(result_dict, json_path)

    return result_dict