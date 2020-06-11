import numpy as np
import json

#################
# basic wrapper #
#################

def external_internal_calc(p4d, points, image_name):
    """
    params
        p4d:        the Pix4D configure class
        points:     the nx3 xyz numpy matrix
        image_name: the string of target projection image name

    return
        2d ndarray, 'lower-left' coordiantes

    example
    >>> from calculation import external_internal_calc
    >>> coords = external_internal_calc(p4d, points=shp_mean_dict['2'], image_name=test_img)
    >>> coords[:5,:]
    array([[205.66042808, 364.060308  ],
           [211.74289725, 366.75581059],
           [220.85402254, 365.72989203],
           [231.23327229, 361.99041811],
           [237.64123696, 356.14638116]])
    """

    T = p4d.img[image_name].cam_pos
    R = p4d.img[image_name].cam_rot

    X_prime = (points - T).dot(R)
    xh, yh = X_prime[:, 0] / X_prime[:, 2], X_prime[:, 1] / X_prime[:, 2]

    r2 = xh ** 2 + yh ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3
    a1 = 1 + p4d.K1 * r2 + p4d.K2 * r4 + p4d.K3 * r6
    xhd = a1 * xh + 2 * p4d.T1 * xh * yh + p4d.T2 * (r2 + 2 * xh ** 2)
    yhd = a1 * yh + 2 * p4d.T2 * xh * yh + p4d.T1 * (r2 + 2 * yh ** 2)

    f = p4d.F * p4d.img[image_name].w / p4d.w_mm
    cx = p4d.Px * p4d.img[image_name].w / p4d.w_mm
    cy = p4d.Py * p4d.img[image_name].h / p4d.h_mm

    xd = f * xhd + cx
    yd = f * yhd + cy

    xa = xd
    ya = p4d.img[image_name].h - yd
    coords_a = np.hstack([xa[:, np.newaxis], ya[:, np.newaxis]])

    return coords_a

def pmatrix_calc(p4d, points, image_name):
    """
    params
        p4d:        the Pix4D configure class
        points:     the nx3 xyz numpy matrix
        image_name: the string of target projection image name

    return
        2d ndarray, 'lower-left' coordiantes

    example
    >>> from calculation import pmatrix_calc
    >>> coords = external_internal_calc(p4d, points=shp_mean_dict['2'], image_name=test_img)
    >>> coords[:5,:]
    array([[205.66042808, 364.060308  ],
           [211.74289725, 366.75581059],
           [220.85402254, 365.72989203],
           [231.23327229, 361.99041811],
           [237.64123696, 356.14638116]])
    """
    xyz1_prime = np.insert(points, 3, 1, axis=1)
    xyz = (xyz1_prime).dot(p4d.img[image_name].pmat.T)
    u = xyz[:, 0] / xyz[:, 2]
    v = xyz[:, 1] / xyz[:, 2]
    coords_a = np.vstack([u, p4d.img[image_name].h - v]).T

    return coords_a

####################
# advanced wrapper #
####################
def in_img_boundary(reprojected_coords, img_size):
    w, h = img_size
    coord_min = reprojected_coords.min(axis=1)
    x_min, y_min = coord_min[0], coord_min[1]
    coord_max= reprojected_coords.max(axis=1)
    x_max, y_max = coord_max[0], coord_max[1]

    if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
        return None
    else:
        return reprojected_coords

def get_img_name_and_coords(p4d, points, method="exin"):
    """
    ::Method::
        exin: using external_internal files
        pmat: using pmatrix files
    """
    in_img_list = []
    coords_list = []
    for im in p4d.img:
        if method == 'ex_in':
            projected_coords = external_internal_calc(p4d, points, im.name)
        else:
            projected_coords = pmatrix_calc(p4d, points, im.name)
        coords = in_img_boundary(projected_coords, (im.w, im.h))
        if coords is not None:
            in_img_list.append(im.name)
            coords_list.append(coords)

    return in_img_list, coords_list


def get_shp_result(p4d, shp_path, get_z_by="mean", z_shift=0, json_save_path=None):
    result_dict = {}
    json_dict = {}
    shp_dict = p4d.read_shp(shp_path, get_z_by, z_shift)
    for shp_key in shp_dict.keys():
        result_dict[shp_key] = {}
        json_dict[shp_key] = {}

        points = shp_dict[shp_key]
        for method in ["exin", "pmat"]:
            result_dict[shp_key][method] = {}
            json_dict[shp_key][method] = {}

            in_img_list, coords_list = get_img_name_and_coords(p4d, points, method)
            for im_name, coord in zip(in_img_list, coords_list):
                result_dict[shp_key][method][im_name] = coord
                json_dict[shp_key][method][im_name] = [c.tolist() for c in coord]

    if isinstance(json_save_path, str) and json_save_path[-5:] == '.json':
        with open(json_save_path, 'w') as result_file:
            json.dump(json_dict, result_file)

    return result_dict