import numpy as np
import pandas as pd

def read_xyz(xyz_path):
    '''
    read pix4d file PROJECTNAME_offset.xyz
    :param xyz_path: the path to target offset.xyz file
    :return: x, y, z float results
    '''
    with open(xyz_path, 'r') as f:
        x, y, z = f.read().split(' ')
    return float(x), float(y), float(z)

def read_pmat(pmat_path):
    '''
    read pix4d file PROJECTNAME_pmatrix.txt
    :param pmat_path: the path of pmatrix file.type
    :return: pmat_dict.type
    pmat_dict = {"DJI_0000.JPG": nparray(3x4), ... ,"DJI_9999.JPG": nparray(3x4)}
    '''

    pmat_nb = np.loadtxt(pmat_path, dtype=float, delimiter=None, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,))
    pmat_names = np.loadtxt(pmat_path, dtype=str, delimiter=None, usecols=0)

    pmat_dict = {}

    for i, name in enumerate(pmat_names):
        pmat_dict[name] = pmat_nb[i, :].reshape((3, 4))

    return pmat_dict


def read_cicp(cicp_path):
    '''
    Read PROJECTNAME_pix4d_calibrated_internal_camera_parameters.cam file
    :param cicp_path:
    :return:
    '''
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
                # Focal Length mm assuming a sensor width of 12.82x8.55mm\n
                w_h = sp_list[8].split('x')
                cam_dict['w_mm'] = float(w_h[0])  # extract e.g. 12.82
                cam_dict['h_mm'] = float(w_h[1][:-4])  # extract e.g. 8.55
    return cam_dict

def read_ccp(ccp_path):
    '''
    Read PROJECTNAME_calibrated_camera_parameters.txt
    :param ccp_path:
    :return:
    '''
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
    for i, line in enumerate(lines):
        if i < 8:
            pass
        else:
            block_id = (i - 7) % 10
            if block_id == 1:  # [line]: fileName imageWidth imageHeight
                file_name, w, h = line[:-1].split()  # ignore \n character
                img_configs[file_name] = {'w': int(w), 'h': int(h)}
            elif block_id == 2:
                cam_mat_line1 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 3:
                cam_mat_line2 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 4:
                cam_mat_line3 = np.fromstring(line, dtype=float, sep=' ')
                cam_mat = np.vstack([cam_mat_line1, cam_mat_line2, cam_mat_line3])
                img_configs[file_name]['cam_matrix'] = cam_mat
            elif block_id == 5:
                img_configs[file_name]['rad_distort'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 6:
                img_configs[file_name]['tan_distort'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 7:
                img_configs[file_name]['cam_pos'] = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 8:
                cam_rot_line1 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 9:
                cam_rot_line2 = np.fromstring(line, dtype=float, sep=' ')
            elif block_id == 0:
                cam_rot_line3 = np.fromstring(line, dtype=float, sep=' ')
                cam_rot = np.vstack([cam_rot_line1, cam_rot_line2, cam_rot_line3])
                img_configs[file_name]['cam_rot'] = cam_rot

    return img_configs

def read_cam_pos(cam_pos_path):
    return pd.read_csv(cam_pos_path, index_col=0, header=None, names=["name", "x", "y", "z"])