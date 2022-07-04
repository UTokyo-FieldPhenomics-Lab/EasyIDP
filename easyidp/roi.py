import os
import numpy as np

def read_cc_txt(txt_path):
    """Read the point cloud annotation made by cloudcompare

    Parameters
    ----------
    txt_path : str
        The path to cloudcompare annotation txt file
    """
    if os.path.exists(txt_path):
        try:
            # try to analysis 'x,y,z' type 
            test_data = np.loadtxt(txt_path, delimiter=',')
        except ValueError as e:
            # means it is 'label, x, y, z' type
            # line i -> Point #0, x, y, z
            # ValueError: could not convert string to float: 'Point '
            # it also view # as comment sign.
            test_data = np.loadtxt(txt_path, delimiter=',', comments="@", usecols=[1,2,3])
    else:
        raise FileNotFoundError(f"Could not find file [{txt_path}]")

    # get points A,B,C,D, but polygon needs A,B,C,D,A
    poly_data = np.append(test_data, test_data[0,:][None,:], axis = 0)

    return poly_data