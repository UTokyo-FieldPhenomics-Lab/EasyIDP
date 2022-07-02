import os
import re
import pytest
import numpy as np

import easyidp as idp

def test_poly2mask_type_int():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    
    xy = np.array([x,y]).T # int type

    width = 11
    height = 10

    mask_shp = idp.cvtools.poly2mask((width, height), xy, engine='shapely')
    mask_pil = idp.cvtools.poly2mask((width, height), xy)

    # the same results from skinage.polygon > 0.18.3
    wanted_shp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)


    wanted_pil = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    np.testing.assert_equal(mask_shp, wanted_shp)
    np.testing.assert_equal(mask_pil, wanted_pil)


def test_poly2mask_type_float():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    xy = np.array([x,y], dtype=np.float16).T   # get float type

    mask_shp = idp.cvtools.poly2mask((10, 10), xy, engine="shapely")
    mask_pil = idp.cvtools.poly2mask((10, 10), xy)

    wanted_shp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    wanted_pil = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    np.testing.assert_equal(mask_shp, wanted_shp)

def test_poly2mask_out_of_bound():
    x=[1,7,4,1]   # horizontal coord
    y=[1,2,8,1]   # vertical coord
    xy = np.array([x,y]).T   

    with pytest.raises(ValueError, match=re.escape("The polygon coords (1, 1, 7, 8) is out of mask boundary [0, 0, 6, 6]")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

def test_poly2mask_wrong_type():
    xy = [[1, 2],[3, 4]]

    with pytest.raises(TypeError, match=re.escape("The `poly_coord` only accept numpy ndarray integer and float types")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([1, 2, 3, 4])
    with pytest.raises(AttributeError, match=re.escape("Only nx2 ndarray are accepted")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(AttributeError, match=re.escape("Only nx2 ndarray are accepted")):
        mask = idp.cvtools.poly2mask((6, 6), xy)

    xy = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    with pytest.raises(TypeError, match=re.escape(f"The `poly_coord` only accept numpy ndarray integer and float types")):
        mask = idp.cvtools.poly2mask((6, 6), xy)
