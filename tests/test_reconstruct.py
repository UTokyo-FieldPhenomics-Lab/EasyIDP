import enum
import re
import pytest
import numpy as np
import easyidp as idp

def test_class_recons_output_io():
    # init return None
    recons = idp.reconstruct.Recons()

    assert recons.pcd is None
    assert recons.dom is None
    assert recons.dsm is None

    # raise error if given false type
    # dom test
    with pytest.raises(TypeError, match=re.escape("Please either specify DOM file ")):
        recons.dom = []
    
    with pytest.raises(FileNotFoundError, match=re.escape("Given DOM file")):
        recons.dom = "tests/not_exist.tiff"

    # dsm test
    with pytest.raises(TypeError, match=re.escape("Please either specify DSM file ")):
        recons.dsm = []
    
    with pytest.raises(FileNotFoundError, match=re.escape("Given DSM file")):
        recons.dsm = "tests/not_exist.tiff"

    # pcd test
    with pytest.raises(TypeError, match=re.escape("Please either specify pointcloud file ")):
        recons.pcd = []
    
    with pytest.raises(FileNotFoundError, match=re.escape("Given pointcloud file")):
        recons.pcd = "tests/not_exist.ply"

    # pass if give correct file
    pcd = "tests\data\pix4d\lotus_tanashi_full\hasu_tanashi_20170525_Ins1RGB_30m_group1_densified_point_cloud.ply"
    dsm = "tests\data\pix4d\lotus_tanashi_full\hasu_tanashi_20170525_Ins1RGB_30m_dsm.tif"
    dom = "tests\data\pix4d\lotus_tanashi_full\hasu_tanashi_20170525_Ins1RGB_30m_transparent_mosaic_group1.tif"

    recons.dom = dom
    recons.dsm = dsm
    recons.pcd = pcd

    assert isinstance(recons.dom, idp.GeoTiff)
    assert isinstance(recons.dsm, idp.GeoTiff)
    assert isinstance(recons.pcd, idp.PointCloud)


def test_class_sensor_in_img_boundary():
    # ignore = None, x, y
    # polygon in bounds / outof bounds
    s = idp.reconstruct.Sensor()
    s.width = 4000
    s.height = 3000

    # x and y all out of boundary
    polygon = np.array([
        [-30, 400],
        [400, -30],
        [700, 800],
        [5000, 2500],
        [2700, 6000],
    ])

    assert s.in_img_boundary(polygon) is None
    assert s.in_img_boundary(polygon, ignore="x") is None
    assert s.in_img_boundary(polygon, ignore="y") is None

    # ====================
    polygon_x_in = np.array([
        [30, 400],
        [400, -30],
        [700, 800],
        [3000, 2500],
        [2700, 6000],
    ])

    polygon_y_in = np.array([
        [-30, 400],
        [400, 30],
        [700, 800],
        [5000, 2500],
        [2700, 2000],
    ])

    ix = s.in_img_boundary(polygon_y_in, ignore="x")
    iy = s.in_img_boundary(polygon_x_in, ignore="y")

    ix_expect = np.array([
        [0, 400],
        [400, 30],
        [700, 800],
        [4000, 2500],
        [2700, 2000],
    ])
    
    iy_expect = np.array([
        [30, 400],
        [400, 0],
        [700, 800],
        [3000, 2500],
        [2700, 3000],
    ])

    np.testing.assert_almost_equal(ix, ix_expect)
    np.testing.assert_almost_equal(iy, iy_expect)


def test_class_calibration_calibrate_error():
    # ============ pix4d =============
    c1 = idp.reconstruct.Calibration()
    c1.software = "pix4d_e"
    c1.type = "frame"
    with pytest.raises(
        TypeError, 
        match=re.escape("Could only handle [pix4d | metashape] projects")):
        c1.calibrate(1,2)

    c2 = idp.reconstruct.Calibration()
    c2.software = "pix4d_e"
    c2.type = "fisheye"
    with pytest.raises(
        TypeError, 
        match=re.escape("Could only handle [pix4d | metashape] projects")):
        c2.calibrate(1,2)

    c3 = idp.reconstruct.Calibration()
    c3.software = "pix4d"
    c3.type = "frame"

    c4 = idp.reconstruct.Calibration()
    c4.software = "pix4d"
    c4.type = "fisheye"
    with pytest.raises(
        NotImplementedError, 
        match=re.escape("Can not calibrate camera type [fisheye],")):
        c4.calibrate(1,2)

    # ========== Metashape ===========
    c5 = idp.reconstruct.Calibration()
    c5.software = "metashape_e"
    c5.type = "frame"
    with pytest.raises(
        TypeError, 
        match=re.escape("Could only handle [pix4d | metashape] projects")):
        c5.calibrate(1,2)

    c6 = idp.reconstruct.Calibration()
    c6.software = "metashape_e"
    c6.type = "fisheye"
    with pytest.raises(
        TypeError, 
        match=re.escape("Could only handle [pix4d | metashape] projects")):
        c6.calibrate(1,2)

    c7 = idp.reconstruct.Calibration()
    c7.software = "metashape"
    c7.type = "frame"

    c8 = idp.reconstruct.Calibration()
    c8.software = "metashape"
    c8.type = "fisheye"
    with pytest.raises(
        NotImplementedError, 
        match=re.escape("Can not calibrate camera type [fisheye],")):
        c4.calibrate(1,2)


def test_class_container():
    # for i in c, 
    # for i in c.keys, 
    # for i in c.values, 
    # for i, j in c.items()
    ctn = idp.reconstruct.Container()

    k = [str(_) for _ in range(5)]

    v = []
    for _ in range(6, 11):
        p = idp.reconstruct.Photo()
        p.label = _
        v.append(p)

    val = {}
    for i, j in zip(k,v):
        ctn[i] = j
        val[i] = j
    
    assert ctn.id_item == val

    # test object iteration
    for idx, value in enumerate(ctn):
        assert value == v[idx]

    for idx, value in ctn.items():
        assert value in v

    for key in ctn.keys():
        assert key in k

    for value in ctn.values():
        assert value in v

def test_func_filter_cloest_img():
    pass