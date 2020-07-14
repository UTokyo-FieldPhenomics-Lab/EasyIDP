import numpy as np
from easyric.io import json


def get_another_photo_pixel(param, from_img_name, to_img_name, xu, yu, z_prime=None):
    f  = param.F  * param.img[from_img_name].w / param.w_mm
    cx = param.Px * param.img[from_img_name].w / param.w_mm
    cy = param.Py * param.img[from_img_name].h / param.h_mm

    R1 = param.img[from_img_name].cam_rot
    R2 = param.img[to_img_name].cam_rot
    T1 = param.img[from_img_name].cam_pos
    T2 = param.img[to_img_name].cam_pos

    if z_prime is None:
        z_prime = - T1[2]

    X1_prime = z_prime * np.asarray([(cx - xu) / f, (cy - yu) / f, 1])

    X = np.linalg.inv(R1).T.dot(X1_prime) + T1
    X2_prime = (X - T2).dot(R2)

    xu2 = - f * (X2_prime[0] / X2_prime[2]) + cx
    yu2 = - f * (X2_prime[1] / X2_prime[2]) + cy

    return xu2, yu2


def get_others_photo_pixels(param, from_img_name, pixels, json_path=None):
    xu, yu = pixels[:,0], pixels[:,1]

    out_dict = {}
    for img_iter in param.img:
        to_img_name = img_iter.name

        if to_img_name == from_img_name:
            continue

        xu2, yu2 = get_another_photo_pixel(param, from_img_name, to_img_name, xu, yu)

        # judge if in the pixel
        if xu2.min() < 0 or xu2.max() > img_iter.w or yu2.min() < 0 or yu2.max() > img_iter.h:
            continue
        else:
            out_dict[to_img_name] = np.vstack([xu2, yu2]).T

    if json_path:
        json.dict2json(out_dict, json_path)

    return out_dict