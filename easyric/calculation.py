import numpy as np
import json

#################
# basic wrapper #
#################

def external_internal_calc(param, points, image_name):
    """
    params
        p4d:        the Pix4D configure class
        points:     the nx3 xyz numpy matrix
        image_name: the string of target projection image name

    return
        2d ndarray, 'lower-left' coordiantes

    example
    >>> from calculation import external_internal_calc
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

    r2 = xh ** 2 + yh ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3
    a1 = 1 + param.K1 * r2 + param.K2 * r4 + param.K3 * r6
    xhd = a1 * xh + 2 * param.T1 * xh * yh + param.T2 * (r2 + 2 * xh ** 2)
    yhd = a1 * yh + 2 * param.T2 * xh * yh + param.T1 * (r2 + 2 * yh ** 2)

    f = param.F * param.img[image_name].w / param.w_mm
    cx = param.Px * param.img[image_name].w / param.w_mm
    cy = param.Py * param.img[image_name].h / param.h_mm

    xb = f * xhd + cx
    yb = f * yhd + cy

    #xa = xb
    #ya = param.img[image_name].h - yb
    #coords_a = np.hstack([xa[:, np.newaxis], ya[:, np.newaxis]])
    coords_b = np.hstack([xb[:, np.newaxis], yb[:, np.newaxis]])

    return coords_b


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
    >>> coords = external_internal_calc(p4d, points=shp['2'], image_name=test_img)
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
    #coords_a = np.vstack([u, p4d.img[image_name].h - v]).T
    coords_b = np.vstack([u, v]).T

    return coords_b


####################
# advanced wrapper #
####################

def in_img_boundary(reprojected_coords, img_size, log=False):
    w, h = img_size
    coord_min = reprojected_coords.min(axis=0)
    x_min, y_min = coord_min[0], coord_min[1]
    coord_max= reprojected_coords.max(axis=0)
    x_max, y_max = coord_max[0], coord_max[1]

    if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
        if log: print('X ', (x_min, x_max, y_min, y_max))
        return None
    else:
        if log: print('O ', (x_min, x_max, y_min, y_max))
        return reprojected_coords


def get_img_name_and_coords(param, points, method='pmat', log=False):
    """
    ::Method::
        exin: using external_internal files
        pmat: using pmatrix files
    """
    in_img_list = []
    coords_list = []
    for im in param.img:
        if log:
            print(f'[Calculator][Judge]{im.name}w:{im.w}h:{im.h}->', end='')
        if method == 'exin':
            projected_coords = external_internal_calc(param, points, im.name)
        else:
            projected_coords = pmatrix_calc(param, points, im.name)
        coords = in_img_boundary(projected_coords, (im.w, im.h), log=log)
        if coords is not None:
            in_img_list.append(im.name)
            coords_list.append(coords)

    return in_img_list, coords_list

'''
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
'''