import shapefile
import os
import warnings
import numpy as np

def read_shp(shp_path, correct_coord=None):
    """
    convert shape file to corrected numpy ndarray
    :param shp_path: string
    :param correct_coord: (x, y, z) tuple
    :return: dict with polygon name as keys
    """
    shp = shapefile.Reader(shp_path)
    shp_dict = {}

    if correct_coord is None:
        correct_coord = (0, 0, 0)

    for i, shape in enumerate(shp.shapes()):
        plot_name = shp.records()[i][-1]
        if isinstance(plot_name, str):
            plot_name = plot_name.replace(r'/', '_')
            plot_name = plot_name.replace(r'\\', '_')
        else:
            plot_name = str(plot_name)
        coord_np = np.asarray(shape.points)
        # correct
        if correct_coord is not None:
            coord_np[:, 0] -= correct_coord[0]
            coord_np[:, 1] -= correct_coord[1]
        coord_np = np.insert(coord_np, 2, 0, axis=1)

        shp_dict[plot_name] = coord_np

    return shp_dict

def read_shps(shp_list, correct_coord=None, rename=True):
    """
    read several shp file into one dict with corrected numpy
    :param shp_list:
    :param correct_coord: list of tuples of each shp file, or one tuple for all shp files
    :param rename: if true, shp file name will be added in the front of polygon name, otherwise, if two shp files have
                   the same name, it will be overwrited.
                   e.g. if you ensure all polygon name is totally different, set rename=False
                        if polygon name start from 0 - n in each shp file, set rename=True
    :return: dict with polygon name as keys
    """
    shp_dict = {}

    shp_num = len(shp_list)
    if correct_coord is None:
        correct_coord = [(0, 0, 0)] * shp_num
    elif isinstance(correct_coord, tuple):
        correct_coord = [correct_coord] * shp_num
    else:
        coord_num = len(correct_coord)
        if shp_num != coord_num:
            raise ValueError(f"The number of shp files ({shp_num}) doesn't match the number of correct_coord ({coord_num}) given")

    total_len = 0
    for i, shp_path in enumerate(shp_list):
        folder, shp_name = os.path.split(os.path.abspath(shp_path))
        temp_shp_dict = read_shp(shp_path, correct_coord[i])
        total_len += len(temp_shp_dict)
        for plot_name in temp_shp_dict.keys():
            if rename:
                shp_dict[f'{shp_name[:-4]}_{plot_name}'] = temp_shp_dict[plot_name]
            else:
                shp_dict[plot_name] = temp_shp_dict[plot_name]

    # check if number correct
    if len(shp_dict) != total_len:
        warnings.warn(f"the total polygon number({len(shp_dict)}) is not same as the sum of all shp files({total_len})")

    return shp_dict
    
def read_xyz(file_path):
    with open(file_path, 'r') as f:
        x, y, z = f.read().split(' ')
    return (float(x), float(y), float(z))